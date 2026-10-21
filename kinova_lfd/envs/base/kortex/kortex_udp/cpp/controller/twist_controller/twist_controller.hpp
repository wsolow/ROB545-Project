// MoveIt Servo-style twist tracker for the Kinova Gen3.
//
// Each 1 ms tick converts the latest tool twist into a joint-position step:
//   1. Drop the command when it is older than incoming_command_timeout.
//   2. Scale unitless commands. speed_units are used as m/s and deg/s.
//   3. Map the twist through the damped pseudo-inverse of the tool Jacobian.
//   4. Slow down near a singularity, using the Jacobian condition number.
//   5. Low-pass the joint step, then enforce velocity and position margins.
//   6. Add the step to the measured joint position.
//
// The Jacobian comes from Pinocchio and gen3_2f85.urdf. Collision checking from
// MoveIt Servo is not included.

#ifndef KINOVA_LFD_KORTEX_UDP_TWIST_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_TWIST_CONTROLLER_HPP

#include "../controller.hpp"
#include "../gen3_model.hpp"
#include "servo_config.hpp"

#include <chrono>
#include <string>

namespace kinova_lfd {

class TwistController : public Controller {
 public:
    TwistController(std::string urdf_path, std::string servo_config_path);

    void Reset(const RobotState& state) override;
    void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) override;

 private:
    bool AcceptCommand(const ipc::CommandPacket& command);
    Eigen::Matrix<double, 6, 1> CartesianTwist(const ipc::CommandPacket& command) const;
    void Hold(const RobotState& state, ActuatorTargets& out);

    Gen3Model model_;
    ServoConfig config_;
    std::string frame_;
    Eigen::Matrix<double, 7, 1> previous_delta_ = Eigen::Matrix<double, 7, 1>::Zero();
    std::chrono::steady_clock::time_point last_command_time_{};
    uint32_t last_seq_ = 0;
    bool halted_ = true;
    bool logged_jacobian_ = false;
};

}  // namespace kinova_lfd

#endif
