// ipc.hpp
//
// Localhost datagram layout shared with kinova_lfd/envs/base/kortex_udp/ipc.py.
// Both sides are little-endian. Packing is explicit so the sizes stay stable.

#ifndef KINOVA_LFD_KORTEX_UDP_IPC_HPP
#define KINOVA_LFD_KORTEX_UDP_IPC_HPP

#include <cstdint>

namespace kinova_lfd {
namespace ipc {

inline constexpr uint32_t kMagic = 0x4B555044;  // 'KUPD' as a little-endian uint32
inline constexpr int kArmDofs = 7;
inline constexpr int kTwistDofs = 6;

enum class PacketType : uint32_t {
    Command = 1,
    Feedback = 2,
    Control = 3,
};

// What the 30 Hz host is asking for. The 1 kHz loop holds the latest packet.
enum class CommandMode : uint32_t {
    Hold = 0,
    JointPosition = 1,
    Twist = 2,
};

enum class ControlOp : uint32_t {
    Stop = 1,
    SetServoingMode = 2,
};

inline constexpr uint32_t kFlagLowLevel = 1u;

#pragma pack(push, 1)

// Joints are radians. Gripper is [0, 1]. Twist is tool-frame linear m/s and angular deg/s,
// matching Kortex SendTwistCommand. duration_s is the requested joint-trajectory time.
// stamp_us is the host monotonic clock at send time, in microseconds. A held key
// repeats the same twist with a new stamp. The twist watchdog uses that stamp.
struct CommandPacket {
    uint32_t magic;
    uint32_t type;
    uint32_t seq;
    uint32_t mode;
    float joints[kArmDofs];
    float gripper;
    float twist[kTwistDofs];
    float duration_s;
    uint64_t stamp_us;
};

struct ControlPacket {
    uint32_t magic;
    uint32_t type;
    uint32_t seq;
    uint32_t op;
    uint32_t value;
};

// Feedback uses Kortex cyclic units: joint degrees, tool meters, tool theta degrees,
// twist linear m/s, twist angular deg/s, gripper percent.
struct FeedbackPacket {
    uint32_t magic;
    uint32_t type;
    uint32_t seq;
    uint32_t actuator_count;
    float tool_xyz[3];
    float tool_theta_deg[3];
    float tool_twist_linear[3];
    float tool_twist_angular[3];
    float wrench_force[3];
    float wrench_torque[3];
    float joint_position_deg[kArmDofs];
    float joint_velocity_deg_s[kArmDofs];
    float joint_torque_nm[kArmDofs];
    float gripper_position;
    float gripper_velocity;
    float gripper_current;
    uint32_t command_seq;
    uint32_t command_mode;
    uint32_t flags;
};

#pragma pack(pop)

static_assert(sizeof(CommandPacket) == 84, "CommandPacket size drifted from ipc.py");
static_assert(sizeof(ControlPacket) == 20, "ControlPacket size drifted from ipc.py");
static_assert(sizeof(FeedbackPacket) == 196, "FeedbackPacket size drifted from ipc.py");

}  // namespace ipc
}  // namespace kinova_lfd

#endif
