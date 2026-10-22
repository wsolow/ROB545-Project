#include "twist_controller.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace kinova_lfd {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kSingularDamping = 1e-4;
// How far the integrated target may lead the measurement. Past this the step is
// dropped, so a joint that is not moving cannot wind the command away from the arm.
constexpr double kMaxLeadRad = 0.10;

// True when some 2π copy of angle lies inside [lower, upper]. The Gen3 limits
// are narrower than one turn, so that copy is unique.
bool CopyInsideLimits(double angle, double lower, double upper, double* canonical) {
    const double mid = 0.5 * (lower + upper);
    const double turns = std::round((mid - angle) / kTwoPi);
    const double candidate = angle + turns * kTwoPi;
    if (candidate < lower - 1e-6 || candidate > upper + 1e-6) {
        return false;
    }
    *canonical = candidate;
    return true;
}

std::array<double, kArmDofs> MeasuredRadians(const RobotState& state) {
    std::array<double, kArmDofs> q{};
    for (int i = 0; i < kArmDofs; ++i) {
        q[static_cast<size_t>(i)] = static_cast<double>(state.joint_position_deg[static_cast<size_t>(i)]) * kDegToRad;
    }
    return q;
}

}  // namespace

TwistController::TwistController(std::string urdf_path, std::string servo_config_path)
    : model_(std::move(urdf_path)), config_(LoadServoConfig(std::move(servo_config_path))) {
    const std::string resolved = model_.resolveFrame(config_.ee_frame_name);
    if (resolved != config_.ee_frame_name) {
        std::cerr << "[kortex_udp] servo frame '" << config_.ee_frame_name << "' is not in the URDF, using "
                  << resolved << "\n";
    }
    model_.setToolFrame(resolved);
    frame_ = resolved;
    std::cerr << "[kortex_udp] twist servo on " << frame_ << " timeout=" << config_.incoming_command_timeout
              << "s type=" << config_.command_in_type << "\n";
}

void TwistController::Reset(const RobotState& /*state*/) {
    filter_stage1_.setZero();
    filter_stage2_.setZero();
    command_q_rad_.fill(0.0);
    command_position_deg_.fill(0.f);
    last_command_time_ = std::chrono::steady_clock::now();
    last_stamp_us_ = 0;
    last_seq_ = 0;
    halted_ = true;
    have_command_ = false;
    logged_jacobian_ = false;
}

bool TwistController::AcceptCommand(const ipc::CommandPacket& command) {
    // The 1 kHz loop holds the latest datagram, so an unchanged twist is not a
    // stopped stream. Keyboard teleop resends that twist on every step. Each
    // send carries a new host stamp, and that stamp is what keeps the servo live.
    const auto now = std::chrono::steady_clock::now();
    const bool new_stamp = command.stamp_us != 0 && command.stamp_us != last_stamp_us_;
    const bool new_seq = command.seq != last_seq_;
    if (new_stamp || new_seq) {
        last_stamp_us_ = command.stamp_us;
        last_seq_ = command.seq;
        last_command_time_ = now;
    }
    const double age = std::chrono::duration<double>(now - last_command_time_).count();
    return age <= config_.incoming_command_timeout;
}

Eigen::Matrix<double, 6, 1> TwistController::CartesianTwist(const ipc::CommandPacket& command) const {
    Eigen::Matrix<double, 6, 1> twist;
    for (int i = 0; i < 3; ++i) {
        twist[i] = static_cast<double>(command.twist[i]);
        twist[i + 3] = static_cast<double>(command.twist[i + 3]) * kDegToRad;
    }
    if (config_.command_in_type == "unitless") {
        twist.head<3>() *= config_.scale_linear;
        twist.tail<3>() = Eigen::Matrix<double, 3, 1>::Zero();
        for (int i = 0; i < 3; ++i) {
            twist[i + 3] = static_cast<double>(command.twist[i + 3]) * config_.scale_rotational;
        }
    } else if (config_.command_in_type != "speed_units") {
        throw std::runtime_error("unsupported command_in_type " + config_.command_in_type);
    }
    if (config_.override_velocity_scaling_factor > 0.0) {
        twist *= config_.override_velocity_scaling_factor;
    }
    return twist;
}

bool TwistController::FilterSettled() const {
    // One microradian per tick is 0.001 rad/s, too small to move the tool.
    constexpr double kStopStepRad = 1e-6;
    return filter_stage1_.cwiseAbs().maxCoeff() < kStopStepRad && filter_stage2_.cwiseAbs().maxCoeff() < kStopStepRad;
}

void TwistController::Hold(const RobotState& state, ActuatorTargets& out) {
    filter_stage1_.setZero();
    filter_stage2_.setZero();
    // The measurement can still be the pre-motion reading while the actuators
    // are catching the last position command. Publishing that reading reverses
    // the twist. Hold the command that was just streamed instead.
    if (have_command_) {
        out.position_deg = command_position_deg_;
        return;
    }
    out.position_deg = state.joint_position_deg;
}

void TwistController::Compute(
    const ipc::CommandPacket& command,
    const RobotState& state,
    double dt_s,
    ActuatorTargets& out) {
    const bool twist_mode = command.mode == static_cast<uint32_t>(ipc::CommandMode::Twist);
    const bool fresh = twist_mode && AcceptCommand(command);
    // A stopped stream keeps the last joint step in the low-pass and lets it
    // decay. Clearing the filter here is what snapped the arm to a stop.
    const bool coast = twist_mode && !fresh && have_command_ && !FilterSettled();
    if (!fresh && !coast) {
        if (twist_mode && !halted_) {
            std::cerr << "[kortex_udp] twist command timed out, holding joint position\n";
        }
        halted_ = true;
        Hold(state, out);
        return;
    }
    if (fresh) {
        halted_ = false;
    } else if (!halted_) {
        std::cerr << "[kortex_udp] twist command timed out, coasting to a stop\n";
        halted_ = true;
    }

    ipc::CommandPacket applied = command;
    if (!fresh) {
        for (float& value : applied.twist) {
            value = 0.f;
        }
    }

    const double dt = dt_s > 1e-6 ? dt_s : 1e-3;
    const auto q = MeasuredRadians(state);
    const Eigen::Matrix<double, 6, 7> jacobian = model_.toolJacobian(q, config_.command_in_ee_frame);

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto singular = svd.singularValues();
    const double condition = singular(0) / std::max(singular(singular.size() - 1), 1e-12);
    double singularity_scale = 1.0;
    if (condition > config_.lower_singularity_threshold) {
        const double span = config_.hard_stop_singularity_threshold - config_.lower_singularity_threshold;
        singularity_scale = std::clamp(1.0 - (condition - config_.lower_singularity_threshold) / span, 0.0, 1.0);
    }
    if (!logged_jacobian_) {
        std::cerr << "[kortex_udp] tool Jacobian condition number " << condition << "\n";
        logged_jacobian_ = true;
    }

    Eigen::Matrix<double, 6, 1> singular_inverse;
    for (int i = 0; i < 6; ++i) {
        const double value = singular(i);
        singular_inverse(i) = value / (value * value + kSingularDamping * kSingularDamping);
    }
    const Eigen::Matrix<double, 7, 6> jacobian_inverse =
        svd.matrixV().leftCols(6) * singular_inverse.asDiagonal() * svd.matrixU().transpose();

    const Eigen::Matrix<double, 7, 1> raw_delta =
        jacobian_inverse * (CartesianTwist(applied) * dt * singularity_scale);
    // low_pass_filter_coeff is the MoveIt name. Each stage has time constant
    // τ = (1 + coeff) milliseconds. A single stage steps the acceleration on
    // the first tick. The second stage starts that acceleration at zero.
    const double filter = std::max(0.0, config_.low_pass_filter_coeff);
    const double blend = 1.0 / (1.0 + filter);
    filter_stage1_ += blend * (raw_delta - filter_stage1_);
    filter_stage2_ += blend * (filter_stage1_ - filter_stage2_);
    Eigen::Matrix<double, 7, 1> delta = filter_stage2_;

    double velocity_scale = 1.0;
    const auto& limits = model_.limits();
    for (int i = 0; i < kArmDofs; ++i) {
        const double max_step = limits.velocity_rad_s[static_cast<size_t>(i)] * dt;
        const double step = std::abs(delta[i]);
        if (max_step > 0.0 && step > max_step) {
            velocity_scale = std::min(velocity_scale, max_step / step);
        }
    }
    if (velocity_scale < 1.0) {
        filter_stage1_ *= velocity_scale;
        filter_stage2_ *= velocity_scale;
        delta *= velocity_scale;
    }

    if (!have_command_) {
        command_q_rad_ = q;
    }
    for (int i = 0; i < kArmDofs; ++i) {
        const size_t index = static_cast<size_t>(i);
        if (!limits.has_position_limit[index]) {
            continue;
        }
        const double margin = config_.joint_limit_margins[index];
        const double lower = limits.lower_rad[index] + margin;
        const double upper = limits.upper_rad[index] - margin;
        if (lower >= upper) {
            continue;
        }
        double canonical = command_q_rad_[index];
        if (!CopyInsideLimits(
                command_q_rad_[index], limits.lower_rad[index], limits.upper_rad[index], &canonical)) {
            continue;
        }
        const double q_next = std::clamp(canonical + delta[i], lower, upper);
        delta[i] = q_next - canonical;
    }
    for (int i = 0; i < kArmDofs; ++i) {
        const size_t index = static_cast<size_t>(i);
        double lead = command_q_rad_[index] + delta[i] - q[index];
        lead -= std::round(lead / kTwoPi) * kTwoPi;
        const double clamped = std::clamp(lead, -kMaxLeadRad, kMaxLeadRad);
        delta[i] -= lead - clamped;
        if (std::abs(delta[i]) + 1e-12 < std::abs(filter_stage2_[i])) {
            filter_stage1_[i] = delta[i];
        }
        filter_stage2_[i] = delta[i];
        command_q_rad_[index] += delta[i];
        command_position_deg_[index] = static_cast<float>(command_q_rad_[index] * kRadToDeg);
        out.velocity_deg_s[index] = static_cast<float>(delta[i] / dt * kRadToDeg);
    }
    have_command_ = true;
    out.position_deg = command_position_deg_;
}

}  // namespace kinova_lfd
