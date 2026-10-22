// Fields read from config/servo.yaml. Topic names and command_out_type are ignored:
// this process publishes joint positions on the Kortex cyclic socket, not on ROS topics.

#ifndef KINOVA_LFD_KORTEX_UDP_SERVO_CONFIG_HPP
#define KINOVA_LFD_KORTEX_UDP_SERVO_CONFIG_HPP

#include <array>
#include <string>

namespace kinova_lfd {

struct ServoConfig {
    std::string command_in_type = "speed_units";
    double scale_linear = 0.9;
    double scale_rotational = 1.5;
    double low_pass_filter_coeff = 40.0;
    std::string ee_frame_name = "gen3_tool_frame";
    bool command_in_ee_frame = true;
    double incoming_command_timeout = 0.5;
    double lower_singularity_threshold = 17.0;
    double hard_stop_singularity_threshold = 30.0;
    std::array<double, 7> joint_limit_margins{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    double override_velocity_scaling_factor = 0.0;
};

ServoConfig LoadServoConfig(const std::string& path);

}  // namespace kinova_lfd

#endif
