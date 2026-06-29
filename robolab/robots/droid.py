# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass, noise

from robolab.constants import ROBOTS_DIR

# Offset of the end-effector control frame relative to base_link. Used by:
#   - DroidCfg.frames "eef_frame" (FrameTransformer publishes this pose for downstream code)
#   - examples/run_abs_ik_demo.py (converts eef_frame targets → base_link IK actions)
# Kept here so all code agrees on what eef_frame is.
EEF_OFFSET_POS: tuple[float, float, float] = (0.0, 0.0, 0.0)
EEF_OFFSET_ROT: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)

_frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
_frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

_WRIST_CAM = TiledCameraCfg(
    # Deliberately named wrist_cam (not wrist_camera) to avoid collision with the
    # wrist_camera prim baked into the robot USD, which has different intrinsics.
    # We spawn our own sensor here with policy-calibrated focal_length 2.8 to match
    # pi05 / DreamZero training.
    prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    # Offset reproduces the baked USD `wrist_camera_flipped` prim
    # (.../base_link/wrist_camera_flipped): look ~ (0.333, 0.011, -0.943) steep down-forward
    # at the workspace. This is the policy-calibrated DROID wrist mount; the prior committed
    # rot pointed the camera sideways (+Y) in isaacsim6 because base_link loads re-oriented.
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.011, -0.031, -0.074), rot=(-0.5813, -0.5733, 0.4157, 0.4007), convention="opengl"
    ),
)


@configclass
class DroidCfg:
    """Cfg class that adds robot articulation to scene configurations."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(ROBOTS_DIR, "franka_robotiq_2f_85_flattened_mimicfix.usd"),  # mimic-damping fix (stops linkage divergence under arm motion)
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            # Un-inverts the arm under isaaclab3. The conventional wxyz identity (1,0,0,0)
            # loads the flattened robot upside-down (gripper below the table, out of reach) --
            # isaaclab3 reads InitialStateCfg.rot as xyzw, so (0,0,0,1) is the upright identity.
            # Verified load-bearing: (1,0,0,0) scores 0/6 on BananaInBowl & GrabAFruit vs 6/6.
            rot=(0, 0, 0, 1),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_inner.*": 0.0,
                "right_inner.*": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            # Arm actuators ported VERBATIM from the official isaaclab Franka+Robotiq cfg:
            #   isaaclab_assets/robots/franka.py :: FRANKA_ROBOTIQ_GRIPPER_CFG
            # i.e. shoulder effort_limit_sim=5200 / stiffness=1100 / damping=80, forearm
            # 720 / 1000 / 80. isaaclab3's actuator limits only reach the PhysX solver when
            # given as the *_sim variants (effort_limit_sim / velocity_limit_sim); the
            # pre-upgrade gains here were the Panda-hand values (effort_limit 87/12,
            # stiffness 400) which stall the arm under gravity in isaacsim6. Verified
            # load-bearing: reverting droid.py to base -> 0/6 on BananaInBowl & GrabAFruit.
            # (We keep RoboLab's own init_state DROID home pose + the mimic-damped DROID
            # USD, not the upstream cfg's franka.usd/init pose.)
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=5200.0,
                velocity_limit_sim=2.175,
                stiffness=1100.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=720.0,
                velocity_limit_sim=2.61,
                stiffness=1000.0,
                damping=80.0,
            ),
            # Gripper actuators also ported verbatim from FRANKA_ROBOTIQ_GRIPPER_CFG
            # (isaaclab_assets/robots/franka.py): the 2F-85 close-loop linkage is modelled
            # as three groups -- driven finger_joint (drive), the parallel inner fingers
            # (finger), and the passive knuckle joints (passive, PD=0). The pre-upgrade
            # config drove finger_joint alone with stiffness=2000, ~100x too stiff for the
            # mimic linkage -> it oscillated (finger_joint swung +/-pi) and corrupted the
            # gripper_pos proprio. Values: drive 1650/17/0.02, finger 50/0.2/0.001,
            # passive 1/0/0 (effort_limit_sim/stiffness/damping).
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=1650.0,
                velocity_limit_sim=10.0,
                stiffness=17.0,
                damping=0.02,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=0.2,
                damping=0.001,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
                effort_limit_sim=1.0,
                velocity_limit_sim=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    wrist_cam = _WRIST_CAM

    # Per-link frame visualization for debugging. EE pose still comes from articulation
    # body state (faster); this sensor is purely for debug rendering. Flip debug_vis to
    # True to render RGB axes at every tracked link in the viewport.
    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/panda_link{i}",
                name=f"panda_link{i}",
            )
            for i in range(8)
        ] + [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                name="gripper_base",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                name="eef_frame",
                offset=OffsetCfg(pos=EEF_OFFSET_POS, rot=EEF_OFFSET_ROT),
            ),
        ],
    )


@configclass
class WristCameraCfg:
    """Introspection wrapper so the wrist camera can be passed to generate_image_obs_from_cameras.
    The scene's wrist_cam is still sourced from DroidCfg; this wrapper only exposes the name.
    """
    wrist_cam = _WRIST_CAM

########################################################
# Contact gripper
########################################################

# IsaacLab ContactSensor requires exactly one prim per env for filter_prim_paths_expr
# (force_matrix_w) to work. .*_inner_finger matches 2 bodies (left + right) per env,
# breaking filtered contact detection. Use one finger only.
contact_gripper = {"gripper": "{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/left_inner_finger"}

########################################################
# Definitions
########################################################


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """ Returns gripper position as 0 for open and 1 for closed.
    """
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


def ee_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector position (x, y, z) in the env-local frame."""
    robot = env.scene[asset_cfg.name]
    # Get the body index for the end effector link
    ee_body_name = "base_link"  # Robotiq gripper base link
    body_idx = robot.data.body_names.index(ee_body_name)
    # Return position (shape: [num_envs, 3])
    return robot.data.body_pos_w[:, body_idx, :] - env.scene.env_origins[:, 0:3]


def ee_quat(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Returns the end effector orientation as quaternion (w, x, y, z) in the world frame."""
    robot = env.scene[asset_cfg.name]
    # Get the body index for the end effector link
    ee_body_name = "base_link"  # Robotiq gripper base link
    body_idx = robot.data.body_names.index(ee_body_name)
    # Return quaternion (shape: [num_envs, 4])
    return robot.data.body_quat_w[:, body_idx, :]


def eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame position (x, y, z) in the env-local frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_pos_w[:, idx, :] - env.scene.env_origins[:, 0:3]


def eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame orientation as quaternion (w, x, y, z) in the world frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_quat_w[:, idx, :]

########################################################
# Actions
########################################################

class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction
@configclass
class DroidJointPositionActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
        # Clamp arm joint-position targets to the Franka Panda limits. isaaclab applies this
        # in JointAction.process_actions (torch.clamp to [min,max]) -- the supported way to
        # bound targets -- so we do NOT need a manual clamp in the eval loop. Needed because
        # an out-of-range target (pi0.5 occasionally overshoots, e.g. panda_joint5) otherwise
        # winds the joint up to tens of radians and corrupts the proprio fed back to the policy.
        # Values are the official Franka Emika Panda joint position limits (rad), from the
        # panda URDF / datasheet.
        clip={
            "panda_joint1": (-2.8973, 2.8973),
            "panda_joint2": (-1.7628, 1.7628),
            "panda_joint3": (-2.8973, 2.8973),
            "panda_joint4": (-3.0718, -0.0698),
            "panda_joint5": (-2.8973, 2.8973),
            "panda_joint6": (-0.0175, 3.7525),
            "panda_joint7": (-2.8973, 2.8973),
        },
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class DroidIKActionCfg:
    """Absolute end-effector pose control via differential IK.

    Tracks base_link directly (no body_offset rotation). If a policy wants to
    command poses in eef_frame's coordinates, it must convert before sending:
    target_base_quat = target_eef_quat ⊗ R_eef_in_base⁻¹. We don't use
    body_offset.rot because IsaacLab's DifferentialIK computes the orientation
    error in root frame but multiplies the rotational Jacobian by R_offset,
    leaving the bases inconsistent — the IK reaches position cleanly, then
    drifts in orientation and diverges. (See run_abs_ik_demo.py for the
    command-side conversion.) The relative IK path is unaffected, so
    DroidRelIKActionCfg keeps body_offset.rot.

    Note:
        if self.cfg.command_type == "position", action_dim = 3, (x, y, z)
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode, action_dim = 6, (dx, dy, dz, droll, dpitch, dyaw)
        if self.cfg.command_type == "pose" and not self.cfg.use_relative_mode, action_dim = 7, (x, y, z, qw, qx, qy, qz)
    """
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base_link",  # Robotiq 2F-85 base flange (gripper mount); matches ee_pos/ee_quat helpers
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        # Robotiq 2F-85 max height base flange -> fingertip is 162.8mm (per Robotiq spec).
        # Uncomment to control the fingertip plane instead of the base flange.
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1628]),
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class DroidRelIKActionCfg:
    """Relative end-effector pose control via differential IK.

    Note:
        if self.cfg.command_type == "position", action_dim = 3, (x, y, z)
        if self.cfg.command_type == "pose" and self.cfg.use_relative_mode, action_dim = 6, (dx, dy, dz, droll, dpitch, dyaw)
        if self.cfg.command_type == "pose" and not self.cfg.use_relative_mode, action_dim = 7, (x, y, z, qw, qx, qy, qz)
    """
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base_link",  # Robotiq 2F-85 base flange (gripper mount); matches ee_pos/ee_quat helpers
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=[0.0, 0.0, 0.0],
            # rot=(0.5, -0.5, 0.5, -0.5),  # Match eef_frame: rotates base_link to the EE control frame.
        ),
        # Robotiq 2F-85 max height base flange -> fingertip is 162.8mm (per Robotiq spec).
        # Uncomment to control the fingertip plane instead of the base flange.
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1628]),
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

########################################################
# Observations
########################################################
@configclass
class ProprioceptionObservationCfg(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos)
    gripper_pos = ObsTerm(
        func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )
    # ee_*: base_link (gripper mount flange). eef_*: eef_frame (EE control frame, R_offset rotated).
    ee_pos = ObsTerm(func=ee_pos)
    ee_quat = ObsTerm(func=ee_quat)
    eef_pos = ObsTerm(func=eef_pos)
    eef_quat = ObsTerm(func=eef_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False # must include
        self.concatenate_terms = False # must include
