// Pinocchio model of the Gen3 arm, shared by the twist servo and the dry-run simulator.

#ifndef KINOVA_LFD_KORTEX_UDP_GEN3_MODEL_HPP
#define KINOVA_LFD_KORTEX_UDP_GEN3_MODEL_HPP

#include <array>
#include <memory>
#include <string>

#include <Eigen/Core>

namespace kinova_lfd {

inline constexpr int kArmDofs = 7;

struct ArmLimits {
    std::array<double, kArmDofs> lower_rad{};
    std::array<double, kArmDofs> upper_rad{};
    std::array<double, kArmDofs> velocity_rad_s{};
    std::array<bool, kArmDofs> has_position_limit{};
};

struct ArmSample {
    std::array<double, kArmDofs> q_rad{};
    std::array<double, kArmDofs> qdot_rad_s{};
    Eigen::Vector3d translation_m = Eigen::Vector3d::Zero();
    Eigen::Vector3d rpy_rad = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_m_s = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_rad_s = Eigen::Vector3d::Zero();
};

// Loads gen3_2f85.urdf. Arm joints are gen3_joint_1 .. gen3_joint_7.
// The tool frame defaults to gen3_tool_frame when the requested name is absent.
class Gen3Model {
 public:
    explicit Gen3Model(const std::string& urdf_path);
    ~Gen3Model();

    Gen3Model(const Gen3Model&) = delete;
    Gen3Model& operator=(const Gen3Model&) = delete;

    const std::string& toolFrame() const;
    std::string resolveFrame(const std::string& requested) const;
    void setToolFrame(const std::string& name);
    const ArmLimits& limits() const;

    // Jacobian of the tool frame. in_tool_frame selects Pinocchio LOCAL, which maps
    // joint velocity to a twist expressed in the tool frame. Otherwise the twist is
    // expressed in the world frame, aligned with the arm base.
    Eigen::Matrix<double, 6, kArmDofs> toolJacobian(
        const std::array<double, kArmDofs>& q_rad,
        bool in_tool_frame);

    void reset(const std::array<double, kArmDofs>& q_rad);
    ArmSample current();

    // q_next = integrate(q, (q_cmd - q) / dt * dt). For the revolute Gen3 joints this
    // lands on q_cmd, and the returned sample is the forward kinematics of that pose.
    ArmSample integrateToward(const std::array<double, kArmDofs>& q_cmd_rad, double dt);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kinova_lfd

#endif
