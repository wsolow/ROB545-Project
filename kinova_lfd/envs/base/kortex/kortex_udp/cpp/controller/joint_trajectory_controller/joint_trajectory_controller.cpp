#include "joint_trajectory_controller.hpp"

#include <cstddef>

namespace kinova_lfd {

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
            goal_rad_[static_cast<size_t>(i)] = command.joints[i];
        }
        has_goal_ = true;
    }
    out.position_deg = state.joint_position_deg;
}

}  // namespace kinova_lfd
