#include "twist_controller.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace kinova_lfd {
namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kSingularDamping = 1e-4;

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
    previous_delta_.setZero();
    last_command_time_ = std::chrono::steady_clock::now();
    last_seq_ = 0;
    halted_ = true;
    logged_jacobian_ = false;
}

bool TwistController::AcceptCommand(const ipc::CommandPacket& command) {
    const auto now = std::chrono::steady_clock::now();
    if (command.seq != last_seq_) {
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

void TwistController::Hold(const RobotState& state, ActuatorTargets& out) {
    previous_delta_.setZero();
    out.position_deg = state.joint_position_deg;
}

void TwistController::Compute(
    const ipc::CommandPacket& command,
    const RobotState& state,
    double dt_s,
    ActuatorTargets& out) {
    if (command.mode != static_cast<uint32_t>(ipc::CommandMode::Twist) || !AcceptCommand(command)) {
        if (command.mode == static_cast<uint32_t>(ipc::CommandMode::Twist) && !halted_) {
            std::cerr << "[kortex_udp] twist command timed out, holding joint position\n";
        }
        halted_ = true;
        Hold(state, out);
        return;
    }
    halted_ = false;

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

    Eigen::Matrix<double, 7, 1> delta =
        jacobian_inverse * (CartesianTwist(command) * dt * singularity_scale);
    const double filter = std::max(0.0, config_.low_pass_filter_coeff);
    delta = (delta + filter * previous_delta_) / (1.0 + filter);

    double velocity_scale = 1.0;
    const auto& limits = model_.limits();
    for (int i = 0; i < kArmDofs; ++i) {
        const double max_step = limits.velocity_rad_s[static_cast<size_t>(i)] * dt;
        const double step = std::abs(delta[i]);
        if (max_step > 0.0 && step > max_step) {
            velocity_scale = std::min(velocity_scale, max_step / step);
        }
    }
    delta *= velocity_scale;

    for (int i = 0; i < kArmDofs; ++i) {
        if (!limits.has_position_limit[static_cast<size_t>(i)]) {
            continue;
        }
        const double margin = config_.joint_limit_margins[static_cast<size_t>(i)];
        const double lower = limits.lower_rad[static_cast<size_t>(i)] + margin;
        const double upper = limits.upper_rad[static_cast<size_t>(i)] - margin;
        if (lower >= upper) {
            continue;
        }
        const double q_next = std::clamp(q[static_cast<size_t>(i)] + delta[i], lower, upper);
        delta[i] = q_next - q[static_cast<size_t>(i)];
    }
    previous_delta_ = delta;

    for (int i = 0; i < kArmDofs; ++i) {
        const double q_next = q[static_cast<size_t>(i)] + delta[i];
        out.position_deg[static_cast<size_t>(i)] = static_cast<float>(q_next * kRadToDeg);
    }
}

}  // namespace kinova_lfd
