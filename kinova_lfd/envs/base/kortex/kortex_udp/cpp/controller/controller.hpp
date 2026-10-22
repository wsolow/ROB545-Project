// Base types for the 1 kHz arm controllers.
//
// Compute writes seven actuator commands in degrees and deg/s. It must not block
// and it must not talk to the robot. The gripper is applied by the cyclic loop.

#ifndef KINOVA_LFD_KORTEX_UDP_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_CONTROLLER_HPP

#include "../ipc.hpp"

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
    // Feedforward for the low-level cyclic command. Zero tells each actuator to
    // brake while the position setpoint is still moving, which jerks the arm
    // along an otherwise smooth trajectory.
    std::array<float, ipc::kArmDofs> velocity_deg_s{};
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

}  // namespace kinova_lfd

#endif
