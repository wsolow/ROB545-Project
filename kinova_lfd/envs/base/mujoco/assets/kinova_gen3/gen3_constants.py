"""Kinova Gen3 robot configuration for Mujoco."""

from pathlib import Path
from typing import Any

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def update_assets(
    assets: dict[str, Any],
    path: str | Path,
    meshdir: str | None = None,
    glob: str = "*",
    recursive: bool = False,
) -> None:
    """Update assets dictionary with files from a directory.

    This function reads files from a directory and adds them to an assets dictionary,
    with keys formatted to include the meshdir prefix when specified.

    Args:
      assets: Dictionary to update with file contents. Keys are asset paths, values are
        file contents as bytes.
      path: Path to directory containing asset files.
      meshdir: Optional mesh directory prefix, typically `spec.meshdir`. If provided,
        will be prepended to asset keys (e.g., "mesh.obj" becomes "custom_dir/mesh.obj").
      glob: Glob pattern for file matching. Defaults to "*" (all files).
      recursive: If True, recursively search subdirectories.

    """
    for f in Path(path).glob(glob):
        if f.is_file():
            asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
            assets[asset_key] = f.read_bytes()
        elif f.is_dir() and recursive:
            update_assets(assets, f, meshdir, glob, recursive)


# MJCF and assets.
##

_HERE = Path(__file__).parent
KINOVA_GEN3_GRIPPER_XML: Path = _HERE / "xmls" / "gen3_gripper.xml"


def get_gen3_assets(meshdir: str) -> dict[str, bytes]:
    """Load Kinova Gen3 mesh assets."""
    assets: dict[str, bytes] = {}
    update_assets(assets, KINOVA_GEN3_GRIPPER_XML.parent / "assets", meshdir)
    return assets


def get_gen3_spec() -> mujoco.MjSpec:
    """Load Kinova Gen3 with Robotiq 2F-85 gripper for position control.

    Includes the 7-DOF arm with position actuators plus the parallel gripper mechanism.
    """
    spec = mujoco.MjSpec.from_file(str(KINOVA_GEN3_GRIPPER_XML))
    spec.assets = get_gen3_assets(spec.meshdir)
    # Position actuators clip at forcerange. The XML kv (250 / 120) spends that
    # budget on damping, so a 0.15 m/s tool command saturates and the arm sags.
    for act in spec.actuators:
        if act.name.startswith("joint_"):
            kv = 40.0 if abs(act.forcerange[1]) >= 30.0 else 20.0
            act.biasprm[2] = -kv
    # Attach keeps the parent scene's <option> and warns on conflicts. The XML
    # requests implicitfast; MujocoCfg applies that integrator after compile.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER

    for key in list(spec.keys):
        spec.delete(key)
    return spec


def get_gen3_robot_cfg() -> EntityCfg:
    """Get a fresh Kinova Gen3 robot configuration instance for lift task.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # Arm joints - ready pose for lifting
                "joint_1": 0.0,
                "joint_2": 0.2274,
                "joint_3": -0.0692,
                "joint_4": 1.5998,
                "joint_5": 0.0160,
                "joint_6": 1.3141,
                "joint_7": -1.6493,
                # Gripper joints - open position
                "right_driver_joint": 0.0,
                "left_driver_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        collisions=(),  # Use collisions from XML
        spec_fn=get_gen3_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    # Arm joints only. ".*" also matches the fingers_actuator tendon
                    # and pinch_site, which MjLab warns about.
                    target_names_expr=("joint_.*",),
                    transmission_type=TransmissionType.JOINT,
                ),
                XmlActuatorCfg(
                    target_names_expr=("fingers_actuator",),
                    transmission_type=TransmissionType.TENDON,
                    command_field="position",
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class Gen3ClosedCfg(EntityCfg):
    """Kinova Gen3 configuration with gripper closed.

    The gripper starts closed and the fingers_actuator ctrl defaults to 255
    (closed) via the init state. Used for tasks like peg-in-hole where the
    robot holds an object throughout the episode.
    """

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints - same ready pose
            "joint_1": 0.0,  # 0°
            "joint_2": 0.3490658504,  # 20°
            "joint_3": 0.0,  # 0°
            "joint_4": 1.7453292519,  # 100°
            "joint_5": 0.0,  # 0°
            "joint_6": -0.5235987756,  # -30°
            "joint_7": -1.5707963268,  # -90°
            # Gripper joints - closed position (0.8 = fully closed)
            "right_driver_joint": 0.8,
            "left_driver_joint": 0.8,
        },
        joint_vel={".*": 0.0},
    )
    collisions = ()
    spec_fn = get_gen3_spec
    articulation = (
        EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=("joint_.*",),
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class Gen3ClosedPegCfg(EntityCfg):
    """Get Kinova Gen3 config for peg-in-hole task.

    Arm is rotated 90° at joint_1 to face the workspace side.
    Gripper is closed (driver joints at 1.0) to hold the peg.
    """

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints
            "joint_1": 1.5707963268,  # 90°
            "joint_2": 0.5235987756,  # 30°
            "joint_3": 0.0,  # 0°
            "joint_4": 1.5707963268,  # 90°
            "joint_5": 0.0,  # 0°
            "joint_6": 1.0471975512,  # 60°
            "joint_7": -1.5707963268,  # -90°
            # Gripper joints - 60% closed (consistent 4-bar linkage state)
            "right_driver_joint": 0.503,
            "right_coupler_joint": 0.001,
            "right_spring_link_joint": 0.505,
            "right_follower_joint": -0.485,
            "left_driver_joint": 0.503,
            "left_coupler_joint": 0.001,
            "left_spring_link_joint": 0.505,
            "left_follower_joint": -0.485,
        },
        joint_vel={".*": 0.0},
    )
    collisions = ()
    spec_fn = get_gen3_spec
    articulation = (
        EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=("joint_.*",),
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
