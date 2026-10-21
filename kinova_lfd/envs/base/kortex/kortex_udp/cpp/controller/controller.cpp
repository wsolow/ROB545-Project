#include "controller.hpp"

namespace kinova_lfd {

void HoldController::Reset(const RobotState& /*state*/) {}

void HoldController::Compute(
    const ipc::CommandPacket& /*command*/,
    const RobotState& state,
    double /*dt_s*/,
    ActuatorTargets& out) {
    out.position_deg = state.joint_position_deg;
}

}  // namespace kinova_lfd
