#include "controller.hpp"

namespace kinova_lfd {
namespace {

void HoldMeasuredJoints(const RobotState& state, ActuatorTargets& out) {
    out.position_deg = state.joint_position_deg;
}

}  // namespace

void HoldController::Reset(const RobotState& /*state*/) {}

void HoldController::Compute(
    const ipc::CommandPacket& /*command*/,
    const RobotState& state,
    double /*dt_s*/,
    ActuatorTargets& out) {
    HoldMeasuredJoints(state, out);
}

void JointTrajectoryController::Reset(const RobotState& /*state*/) {
    has_goal_ = false;
    goal_rad_.fill(0.f);
}

void JointTrajectoryController::Compute(
    const ipc::CommandPacket& command,
    const RobotState& state,
    double /*dt_s*/,
    ActuatorTargets& out) {
    if (command.mode == static_cast<uint32_t>(ipc::CommandMode::JointPosition)) {
        for (int i = 0; i < ipc::kArmDofs; ++i) {
            goal_rad_[i] = command.joints[i];
        }
        has_goal_ = true;
    }

    // Placeholder: do not interpolate toward goal_rad_. Commanding the measured
    // position keeps the arm still until a trajectory generator fills `out`.
    HoldMeasuredJoints(state, out);
}

void TwistController::Reset(const RobotState& /*state*/) {
    has_twist_ = false;
    twist_.fill(0.f);
}

void TwistController::Compute(
    const ipc::CommandPacket& command,
    const RobotState& state,
    double /*dt_s*/,
    ActuatorTargets& out) {
    if (command.mode == static_cast<uint32_t>(ipc::CommandMode::Twist)) {
        for (int i = 0; i < ipc::kTwistDofs; ++i) {
            twist_[i] = command.twist[i];
        }
        has_twist_ = true;
    }

    // Placeholder: do not integrate twist_ into a joint trajectory.
    HoldMeasuredJoints(state, out);
}

}  // namespace kinova_lfd
