#include "servo_config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace kinova_lfd {
namespace {

std::string Trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string Unquote(std::string text) {
    text = Trim(text);
    if (text.size() >= 2 && ((text.front() == '"' && text.back() == '"') ||
                             (text.front() == '\'' && text.back() == '\''))) {
        return text.substr(1, text.size() - 2);
    }
    return text;
}

bool ParseBool(const std::string& text) {
    const std::string value = Unquote(text);
    if (value == "true" || value == "True" || value == "1") {
        return true;
    }
    if (value == "false" || value == "False" || value == "0") {
        return false;
    }
    throw std::runtime_error("expected a boolean, got '" + text + "'");
}

void ParseMargins(const std::string& text, std::array<double, 7>& margins) {
    const auto open = text.find('[');
    const auto close = text.find(']');
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        throw std::runtime_error("joint_limit_margins must be a list of 7 numbers");
    }
    std::stringstream stream(text.substr(open + 1, close - open - 1));
    std::string item;
    int count = 0;
    while (std::getline(stream, item, ',')) {
        if (count >= 7) {
            throw std::runtime_error("joint_limit_margins has more than 7 values");
        }
        margins[static_cast<size_t>(count++)] = std::stod(Trim(item));
    }
    if (count != 7) {
        throw std::runtime_error("joint_limit_margins must contain 7 values");
    }
}

}  // namespace

ServoConfig LoadServoConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open servo config " + path);
    }

    ServoConfig config;
    bool in_scale = false;
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        if (Trim(line).empty()) {
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = Trim(line.substr(0, colon));
        const std::string value = Trim(line.substr(colon + 1));
        const bool nested = !line.empty() && std::isspace(static_cast<unsigned char>(line.front()));

        if (key == "scale" && value.empty()) {
            in_scale = true;
            continue;
        }
        if (!nested) {
            in_scale = key == "scale";
        }

        if (in_scale && key == "linear") {
            config.scale_linear = std::stod(value);
        } else if (in_scale && key == "rotational") {
            config.scale_rotational = std::stod(value);
        } else if (key == "command_in_type") {
            config.command_in_type = Unquote(value);
        } else if (key == "low_pass_filter_coeff") {
            config.low_pass_filter_coeff = std::stod(value);
        } else if (key == "ee_frame_name") {
            config.ee_frame_name = Unquote(value);
        } else if (key == "apply_twist_commands_about_ee_frame") {
            config.command_in_ee_frame = ParseBool(value);
        } else if (key == "incoming_command_timeout") {
            config.incoming_command_timeout = std::stod(value);
        } else if (key == "lower_singularity_threshold") {
            config.lower_singularity_threshold = std::stod(value);
        } else if (key == "hard_stop_singularity_threshold") {
            config.hard_stop_singularity_threshold = std::stod(value);
        } else if (key == "joint_limit_margins") {
            ParseMargins(value, config.joint_limit_margins);
        } else if (key == "override_velocity_scaling_factor") {
            config.override_velocity_scaling_factor = std::stod(value);
        }
    }
    if (config.hard_stop_singularity_threshold <= config.lower_singularity_threshold) {
        throw std::runtime_error("hard_stop_singularity_threshold must exceed lower_singularity_threshold");
    }
    return config;
}

}  // namespace kinova_lfd
