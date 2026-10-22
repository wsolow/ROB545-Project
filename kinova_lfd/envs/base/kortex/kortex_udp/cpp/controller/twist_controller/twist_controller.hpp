// MoveIt Servo-style twist tracker for the Kinova Gen3.
//
// Each 1 ms tick converts the latest tool twist into a joint-position step:
//   1. Drop the command when no newly stamped packet has arrived within
//      incoming_command_timeout. Repeating the same twist refreshes the stamp.
//   2. Scale unitless commands. speed_units are used as m/s and deg/s.
//   3. Map the twist through the damped pseudo-inverse of the tool Jacobian.
//   4. Slow down near a singularity, using the Jacobian condition number.
//   5. Low-pass the joint step twice, then enforce velocity and position margins.
//   6. Integrate that step onto the last commanded position.
//
// The Jacobian is taken at the latest measurement, so the model stays on the arm.
// The position target is not. Replacing the target with the measurement every
// tick copies shoulder sag into the command: the next target is already lower,
// and both a linear twist and an angular twist walk the tool down. The smoothed
// integral holds the joint that the twist does not ask to move, which gives the
// position loop an error to servo instead of a target that follows the sag.
// Kinova angles a full turn outside the URDF interval are kept on that turn.
// A timeout feeds a zero twist through the same low-pass, so the arm slows on
// that ramp and then holds the last target. It does not reverse.
//
// The Jacobian comes from Pinocchio and gen3_2f85.urdf. Collision checking from
// MoveIt Servo is not included.

#ifndef KINOVA_LFD_KORTEX_UDP_TWIST_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_TWIST_CONTROLLER_HPP

#include "../controller.hpp"
#include "../gen3_model.hpp"
#include "servo_config.hpp"

#include <array>
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
    bool FilterSettled() const;
    Eigen::Matrix<double, 6, 1> CartesianTwist(const ipc::CommandPacket& command) const;
    void Hold(const RobotState& state, ActuatorTargets& out);

    Gen3Model model_;
    ServoConfig config_;
    std::string frame_;
    // Two cascaded low-pass states of the joint step. One stage would step the
    // acceleration when a twist appears. The second stage lets acceleration rise
    // from zero. Both are joint-space radians per tick.
    Eigen::Matrix<double, 7, 1> filter_stage1_ = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> filter_stage2_ = Eigen::Matrix<double, 7, 1>::Zero();
    std::array<double, kArmDofs> command_q_rad_{};
    std::array<float, kArmDofs> command_position_deg_{};
    std::chrono::steady_clock::time_point last_command_time_{};
    uint64_t last_stamp_us_ = 0;
    uint32_t last_seq_ = 0;
    bool halted_ = true;
    bool have_command_ = false;
    bool logged_jacobian_ = false;
};

}  // namespace kinova_lfd

#endif
