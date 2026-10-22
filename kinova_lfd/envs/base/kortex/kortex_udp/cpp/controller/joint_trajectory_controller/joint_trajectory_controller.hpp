// Synchronized joint-position trajectory for the Kinova Gen3.
//
// ROS 2's joint_trajectory_controller samples a path whose timestamps are already
// filled in, and it interpolates position-only waypoints linearly. This controller
// has one position goal and a requested duration, and no velocity input, so it
// builds the timing itself.
//
// Every joint follows the same progress curve s(t) from 0 to 1. Joints that travel
// farther therefore move faster, and they all start and stop together. s(t) is a
// trapezoid (or a triangle, when the move is short): accelerate, cruise, decelerate.
// The cruise speed and acceleration are reduced so no joint exceeds its URDF
// velocity limit or its acceleration limit. If the requested duration is too short
// for those limits, the duration is extended and a message is printed.
//
// The sampled position then passes through two low-pass stages. A trapezoid steps
// its acceleration at the start, at cruise, and at the stop. The second stage lets
// that acceleration rise and fall. The filter state is the position last sent to
// the arm. A new goal continues from that position. Restarting the filter from the
// measured pose would jump back along the move already commanded, because feedback
// still lags the target.
//
// A goal is an angle. Kinova's cyclic feedback is not wrapped into the URDF
// interval, so the pose the arm is already in can differ from a wrapped command
// by a whole turn. The planner puts the goal on the reported turn before timing
// the move. A command equal to the measured angle, or to that angle plus any
// number of turns, holds. A limited joint still follows a long goal that stays
// inside its range.

#ifndef KINOVA_LFD_KORTEX_UDP_JOINT_TRAJECTORY_CONTROLLER_HPP
#define KINOVA_LFD_KORTEX_UDP_JOINT_TRAJECTORY_CONTROLLER_HPP

#include "../controller.hpp"
#include "../gen3_model.hpp"

#include <array>
#include <string>

namespace kinova_lfd {

class JointTrajectoryController : public Controller {
 public:
    explicit JointTrajectoryController(std::string urdf_path);

    void Reset(const RobotState& state) override;
    void Compute(
        const ipc::CommandPacket& command,
        const RobotState& state,
        double dt_s,
        ActuatorTargets& out) override;

    bool has_goal() const { return active_; }
    double duration() const { return segment_.duration; }

 private:
    struct Segment {
        std::array<double, kArmDofs> start_rad{};
        std::array<double, kArmDofs> delta_rad{};
        double accel = 0.0;
        double cruise = 0.0;
        double t_accel = 0.0;
        double t_cruise = 0.0;
        double duration = 0.0;
    };

    bool SameCommand(const ipc::CommandPacket& command) const;
    void Plan(const ipc::CommandPacket& command, const RobotState& state);
    double Progress(double t_s) const;
    void Write(double progress, double dt_s, ActuatorTargets& out);

    Gen3Model model_;
    Segment segment_{};
    std::array<double, kArmDofs> command_goal_{};
    std::array<double, kArmDofs> filter_stage1_rad_{};
    std::array<double, kArmDofs> filter_stage2_rad_{};
    double command_duration_ = 0.0;
    double elapsed_ = 0.0;
    bool active_ = false;
    bool have_command_ = false;
};

}  // namespace kinova_lfd

#endif
