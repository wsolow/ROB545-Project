#include "joint_trajectory_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

namespace kinova_lfd {
namespace {

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kGoalTolerance = 1e-4;
constexpr double kMinMove = 1e-6;

// Kinova's Gen3 joint_limits.yaml. The URDF has no acceleration field.
constexpr std::array<double, kArmDofs> kMaxAcceleration = {1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0};

std::array<double, kArmDofs> MeasuredRadians(const RobotState& state) {
    std::array<double, kArmDofs> q{};
    for (int i = 0; i < kArmDofs; ++i) {
        q[static_cast<size_t>(i)] = static_cast<double>(state.joint_position_deg[static_cast<size_t>(i)]) * kDegToRad;
    }
    return q;
}

}  // namespace

JointTrajectoryController::JointTrajectoryController(std::string urdf_path) : model_(std::move(urdf_path)) {}

void JointTrajectoryController::Reset(const RobotState& /*state*/) {
    active_ = false;
    elapsed_ = 0.0;
    segment_ = Segment{};
    command_goal_.fill(0.0);
    command_duration_ = 0.0;
}

bool JointTrajectoryController::SameCommand(const ipc::CommandPacket& command) const {
    if (!active_ || std::abs(static_cast<double>(command.duration_s) - command_duration_) > kGoalTolerance) {
        return false;
    }
    for (int i = 0; i < kArmDofs; ++i) {
        if (std::abs(static_cast<double>(command.joints[i]) - command_goal_[static_cast<size_t>(i)]) > kGoalTolerance) {
            return false;
        }
    }
    return true;
}

void JointTrajectoryController::Plan(const ipc::CommandPacket& command, const RobotState& state) {
    const auto start = MeasuredRadians(state);
    const auto& limits = model_.limits();
    std::array<double, kArmDofs> goal{};
    for (int i = 0; i < kArmDofs; ++i) {
        const size_t index = static_cast<size_t>(i);
        command_goal_[index] = static_cast<double>(command.joints[i]);
        goal[index] = command_goal_[index];
        if (!limits.has_position_limit[index]) {
            continue;
        }
        const double lower = limits.lower_rad[index];
        const double upper = limits.upper_rad[index];
        if (goal[index] < lower || goal[index] > upper) {
            const double clamped = std::clamp(goal[index], lower, upper);
            std::cerr << "[kortex_udp] joint " << (i + 1) << " goal " << goal[index]
                      << " rad is outside [" << lower << ", " << upper << "], using " << clamped << "\n";
            goal[index] = clamped;
        }
    }
    command_duration_ = static_cast<double>(command.duration_s);

    segment_ = Segment{};
    segment_.start_rad = start;
    double v_scale = std::numeric_limits<double>::infinity();
    double a_scale = std::numeric_limits<double>::infinity();
    for (int i = 0; i < kArmDofs; ++i) {
        const size_t index = static_cast<size_t>(i);
        segment_.delta_rad[index] = goal[index] - start[index];
        const double distance = std::abs(segment_.delta_rad[index]);
        if (distance < kMinMove) {
            segment_.delta_rad[index] = 0.0;
            continue;
        }
        const double velocity = limits.velocity_rad_s[index];
        if (velocity > kMinMove) {
            v_scale = std::min(v_scale, velocity / distance);
        }
        const double acceleration = kMaxAcceleration[index];
        if (acceleration > kMinMove) {
            a_scale = std::min(a_scale, acceleration / distance);
        }
    }
    active_ = true;
    elapsed_ = 0.0;
    if (!std::isfinite(v_scale) || !std::isfinite(a_scale)) {
        return;
    }

    const double peak = std::sqrt(a_scale);
    const double minimum = peak <= v_scale ? 2.0 / peak : (1.0 / v_scale + v_scale / a_scale);
    const double requested = command_duration_ > 0.0 ? command_duration_ : 0.0;
    const double duration = std::max(requested, minimum);
    const double discriminant = std::max(0.0, (a_scale * duration) * (a_scale * duration) - 4.0 * a_scale);
    double cruise = (a_scale * duration - std::sqrt(discriminant)) / 2.0;
    cruise = std::min(cruise, v_scale);
    const double t_accel = cruise / a_scale;
    segment_.accel = a_scale;
    segment_.cruise = cruise;
    segment_.t_accel = t_accel;
    segment_.t_cruise = std::max(0.0, duration - 2.0 * t_accel);
    segment_.duration = duration;

    std::cerr << "[kortex_udp] joint trajectory";
    if (duration > requested + 1e-3) {
        std::cerr << " requested " << requested << " s, using " << duration
                  << " s to respect velocity and acceleration limits";
    } else {
        std::cerr << " duration " << duration << " s";
    }
    std::cerr << ", peak speed deg/s";
    for (int i = 0; i < kArmDofs; ++i) {
        const double peak_deg_s = std::abs(segment_.delta_rad[static_cast<size_t>(i)]) * cruise * kRadToDeg;
        std::cerr << (i == 0 ? " " : ", ") << peak_deg_s;
    }
    std::cerr << "\n";
}

double JointTrajectoryController::Progress(double t_s) const {
    if (segment_.duration <= 0.0 || t_s >= segment_.duration) {
        return 1.0;
    }
    if (t_s <= 0.0) {
        return 0.0;
    }
    const double t_accel = segment_.t_accel;
    const double t_cruise = segment_.t_cruise;
    const double accel = segment_.accel;
    const double cruise = segment_.cruise;
    if (t_s < t_accel) {
        return 0.5 * accel * t_s * t_s;
    }
    if (t_s < t_accel + t_cruise) {
        return 0.5 * accel * t_accel * t_accel + cruise * (t_s - t_accel);
    }
    const double t_decel = t_s - t_accel - t_cruise;
    const double coast = 0.5 * accel * t_accel * t_accel + cruise * t_cruise;
    return std::min(1.0, coast + cruise * t_decel - 0.5 * accel * t_decel * t_decel);
}

void JointTrajectoryController::Write(double progress, ActuatorTargets& out) const {
    for (int i = 0; i < kArmDofs; ++i) {
        const size_t index = static_cast<size_t>(i);
        const double q = segment_.start_rad[index] + segment_.delta_rad[index] * progress;
        out.position_deg[index] = static_cast<float>(q * kRadToDeg);
    }
}

void JointTrajectoryController::Compute(
    const ipc::CommandPacket& command,
    const RobotState& state,
    double dt_s,
    ActuatorTargets& out) {
    if (command.mode != static_cast<uint32_t>(ipc::CommandMode::JointPosition)) {
        out.position_deg = state.joint_position_deg;
        return;
    }
    if (!SameCommand(command)) {
        Plan(command, state);
    } else {
        elapsed_ += std::max(0.0, dt_s);
    }
    Write(Progress(elapsed_), out);
}

}  // namespace kinova_lfd
