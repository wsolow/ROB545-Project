#include "gen3_model.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace kinova_lfd {
namespace {

Eigen::Vector3d RpyFromMatrix(const Eigen::Matrix3d& rotation) {
    Eigen::Vector3d rpy;
    rpy.y() = std::atan2(-rotation(2, 0), std::hypot(rotation(0, 0), rotation(1, 0)));
    rpy.x() = std::atan2(rotation(2, 1), rotation(2, 2));
    rpy.z() = std::atan2(rotation(1, 0), rotation(0, 0));
    return rpy;
}

}  // namespace

struct Gen3Model::Impl {
    pinocchio::Model model;
    std::unique_ptr<pinocchio::Data> data;
    std::array<pinocchio::JointIndex, kArmDofs> joint_ids{};
    std::array<int, kArmDofs> q_index{};
    std::array<int, kArmDofs> v_index{};
    pinocchio::FrameIndex tool_frame = 0;
    std::string tool_frame_name;
    ArmLimits limits;
    Eigen::VectorXd q;
    std::array<double, kArmDofs> arm_q{};
    bool has_q = false;

    explicit Impl(const std::string& urdf_path) {
        pinocchio::urdf::buildModel(urdf_path, model);
        data = std::make_unique<pinocchio::Data>(model);
        q = pinocchio::neutral(model);

        for (int i = 0; i < kArmDofs; ++i) {
            const std::string name = "gen3_joint_" + std::to_string(i + 1);
            if (!model.existJointName(name)) {
                throw std::runtime_error("URDF is missing " + name);
            }
            const auto joint_id = model.getJointId(name);
            joint_ids[static_cast<size_t>(i)] = joint_id;
            const int nq = model.nqs[joint_id];
            const int nv = model.nvs[joint_id];
            if (nv != 1 || (nq != 1 && nq != 2)) {
                throw std::runtime_error(
                    name + " has nq=" + std::to_string(nq) + " nv=" + std::to_string(nv));
            }
            q_index[static_cast<size_t>(i)] = model.idx_qs[joint_id];
            v_index[static_cast<size_t>(i)] = model.idx_vs[joint_id];

            const int iq = q_index[static_cast<size_t>(i)];
            const int iv = v_index[static_cast<size_t>(i)];
            const double lower = model.lowerPositionLimit[iq];
            const double upper = model.upperPositionLimit[iq];
            limits.lower_rad[static_cast<size_t>(i)] = lower;
            limits.upper_rad[static_cast<size_t>(i)] = upper;
            limits.velocity_rad_s[static_cast<size_t>(i)] = model.upperVelocityLimit[iv];
            limits.has_position_limit[static_cast<size_t>(i)] =
                nq == 1 && std::isfinite(lower) && std::isfinite(upper) && (upper - lower) < 100.0;
        }

        tool_frame_name = resolve("gen3_tool_frame");
        tool_frame = model.getFrameId(tool_frame_name);
    }

    std::string resolve(const std::string& requested) const {
        if (model.existFrame(requested)) {
            return requested;
        }
        const std::string prefixed = "gen3_" + requested;
        if (model.existFrame(prefixed)) {
            return prefixed;
        }
        if (requested != "gen3_tool_frame" && model.existFrame("gen3_tool_frame")) {
            return "gen3_tool_frame";
        }
        throw std::runtime_error("URDF has no frame named " + requested);
    }

    void writeArm(const std::array<double, kArmDofs>& q_arm) {
        for (int i = 0; i < kArmDofs; ++i) {
            const size_t index = static_cast<size_t>(i);
            const int iq = q_index[index];
            const double angle = q_arm[index];
            if (model.nqs[joint_ids[index]] == 1) {
                q[iq] = angle;
            } else {
                q[iq] = std::cos(angle);
                q[iq + 1] = std::sin(angle);
            }
            arm_q[index] = angle;
        }
        has_q = true;
    }

    std::array<double, kArmDofs> readArm() const { return arm_q; }

    ArmSample sampleAt(const Eigen::VectorXd& velocity) {
        pinocchio::forwardKinematics(model, *data, q, velocity);
        pinocchio::updateFramePlacements(model, *data);
        const pinocchio::Motion twist =
            pinocchio::getFrameVelocity(model, *data, tool_frame, pinocchio::LOCAL_WORLD_ALIGNED);
        const pinocchio::SE3& pose = data->oMf[tool_frame];

        ArmSample sample;
        sample.q_rad = readArm();
        for (int i = 0; i < kArmDofs; ++i) {
            sample.qdot_rad_s[static_cast<size_t>(i)] = velocity[v_index[static_cast<size_t>(i)]];
        }
        sample.translation_m = pose.translation();
        sample.rpy_rad = RpyFromMatrix(pose.rotation());
        sample.linear_m_s = twist.linear();
        sample.angular_rad_s = twist.angular();
        return sample;
    }
};

Gen3Model::Gen3Model(const std::string& urdf_path) : impl_(std::make_unique<Impl>(urdf_path)) {}

Gen3Model::~Gen3Model() = default;

const std::string& Gen3Model::toolFrame() const { return impl_->tool_frame_name; }

std::string Gen3Model::resolveFrame(const std::string& requested) const { return impl_->resolve(requested); }

void Gen3Model::setToolFrame(const std::string& name) {
    if (!impl_->model.existFrame(name)) {
        throw std::runtime_error("URDF has no frame named " + name);
    }
    impl_->tool_frame_name = name;
    impl_->tool_frame = impl_->model.getFrameId(name);
}

const ArmLimits& Gen3Model::limits() const { return impl_->limits; }

Eigen::Matrix<double, 6, kArmDofs> Gen3Model::toolJacobian(
    const std::array<double, kArmDofs>& q_rad,
    bool in_tool_frame) {
    impl_->writeArm(q_rad);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, impl_->model.nv);
    const auto reference = in_tool_frame ? pinocchio::LOCAL : pinocchio::LOCAL_WORLD_ALIGNED;
    pinocchio::computeFrameJacobian(
        impl_->model, *impl_->data, impl_->q, impl_->tool_frame, reference, jacobian);

    Eigen::Matrix<double, 6, kArmDofs> arm = Eigen::Matrix<double, 6, kArmDofs>::Zero();
    for (int i = 0; i < kArmDofs; ++i) {
        arm.col(i) = jacobian.col(impl_->v_index[static_cast<size_t>(i)]);
    }
    return arm;
}

void Gen3Model::reset(const std::array<double, kArmDofs>& q_rad) { impl_->writeArm(q_rad); }

ArmSample Gen3Model::current() {
    return impl_->sampleAt(Eigen::VectorXd::Zero(impl_->model.nv));
}

ArmSample Gen3Model::integrateToward(const std::array<double, kArmDofs>& q_cmd_rad, double dt) {
    if (!impl_->has_q) {
        reset(q_cmd_rad);
        return current();
    }
    const double step = dt > 1e-6 ? dt : 1e-3;
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(impl_->model.nv);
    const auto q_now = impl_->readArm();
    for (int i = 0; i < kArmDofs; ++i) {
        const double qdot = (q_cmd_rad[static_cast<size_t>(i)] - q_now[static_cast<size_t>(i)]) / step;
        velocity[impl_->v_index[static_cast<size_t>(i)]] = qdot;
    }
    impl_->q = pinocchio::integrate(impl_->model, impl_->q, velocity * step);
    for (int i = 0; i < kArmDofs; ++i) {
        impl_->arm_q[static_cast<size_t>(i)] = q_cmd_rad[static_cast<size_t>(i)];
    }
    return impl_->sampleAt(velocity);
}

}  // namespace kinova_lfd
