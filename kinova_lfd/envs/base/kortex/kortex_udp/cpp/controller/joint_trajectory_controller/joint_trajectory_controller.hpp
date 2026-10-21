// Joint-position command slot.
//
// The bridge still selects this controller for joint-position setpoints. It records
// the goal and commands the measured joint angles. Trajectory generation is not
// implemented yet.

#ifndef KINOVA_LFD_KORTEX_UDP_JOINT_TRAJECTORY_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_JOINT_TRAJECTORY_CONTROLLER_HPP

#include "../controller.hpp"

#include <array>

namespace kinova_lfd {

class JointTrajectoryController : public Controller {
 public:
    void Reset(const RobotState& state) override;
    void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) override;

    bool has_goal() const { return has_goal_; }
    const std::array<float, ipc::kArmDofs>& goal_rad() const { return goal_rad_; }

 private:
    std::array<float, ipc::kArmDofs> goal_rad_{};
    bool has_goal_ = false;
};

}  // namespace kinova_lfd

#endif
