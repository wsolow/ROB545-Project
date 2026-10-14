from enum import IntEnum


class ActionSpec(IntEnum):
    TWIST_TCP = 0
    MOVE_TCP = 1
    JOINTS_VEL = 3
    JOINTS_POS = 4
    JOINT_ADMITTANCE = 5
    CARTESIAN_ADMITTANCE = 6
    NULLSPACE_ADMITTANCE = 7
    NONE_ADMITTANCE = 8
