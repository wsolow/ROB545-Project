// controller.hpp
//
// 1 kHz arm-command API. The realtime loop calls Compute every millisecond with the
// latest host setpoint (refreshed at about 30 Hz and then held).
//
// Compute must write the seven actuator position commands, in degrees, that will be
// sent on the Kortex cyclic UDP socket. It must not block, and it must not talk to
// the robot. Gripper position is applied by the cyclic loop itself.
//
// JointTrajectoryController and TwistController currently store the setpoint and
// command the measured joint position, so the arm holds. Trajectory interpolation
// and twist integration belong in Compute and are intentionally not implemented.

#ifndef KINOVA_LFD_KORTEX_UDP_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_CONTROLLER_HPP

#include "ipc.hpp"

#include <array>

namespace kinova_lfd {

struct RobotState {
    std::array<float, ipc::kArmDofs> joint_position_deg{};
    std::array<float, ipc::kArmDofs> joint_velocity_deg_s{};
    std::array<float, ipc::kArmDofs> joint_torque_nm{};
    std::array<float, 3> tool_xyz_m{};
    std::array<float, 3> tool_theta_deg{};
    std::array<float, 3> tool_twist_linear{};
    std::array<float, 3> tool_twist_angular{};
    std::array<float, 3> wrench_force{};
    std::array<float, 3> wrench_torque{};
    float gripper_percent = 0.f;
    float gripper_velocity = 0.f;
    float gripper_current = 0.f;
};

struct ActuatorTargets {
    std::array<float, ipc::kArmDofs> position_deg{};
};

class Controller {
 public:
    virtual ~Controller() = default;

    // Called when this controller becomes the active one.
    virtual void Reset(const RobotState& state) = 0;

    // dt_s is the cyclic period (0.001 when the loop is on time).
    virtual void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) = 0;
};

// Holds the measured joint position. Used before any host setpoint arrives.
class HoldController : public Controller {
 public:
    void Reset(const RobotState& state) override;
    void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) override;
};

// Future joint-space trajectory tracker.
// command.joints is the goal in radians. command.gripper is unused here.
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

// Future tool-frame twist tracker.
// command.twist is linear m/s (0:3) and angular deg/s (3:6).
class TwistController : public Controller {
 public:
    void Reset(const RobotState& state) override;
    void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) override;

    bool has_twist() const { return has_twist_; }
    const std::array<float, ipc::kTwistDofs>& twist() const { return twist_; }

 private:
    std::array<float, ipc::kTwistDofs> twist_{};
    bool has_twist_ = false;
};

}  // namespace kinova_lfd

#endif
