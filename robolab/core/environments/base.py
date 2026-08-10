# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base configuration classes for RoboLab environments.

This module contains all the base configuration classes used to define
environment configurations, including observations, actions, events,
rewards, terminations, and the main RobolabDefaultEnvCfg.
"""


import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robolab.constants
from robolab.constants import get_output_dir
from robolab.core.events.basic_recorders import (
    InitialStateRecorderCfg,
    PostStepBBoxRecorderCfg,
    PostStepEndEffectorPoseRecorderCfg,
    PostStepRobotRootPoseRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderCfg


@configclass
class ObservationCfg:
    """Observation terms for the MDP."""

@configclass
class ActionCfg:
    """Observation terms for the MDP."""

@configclass
class BaseEventCfg:
    """Configuration for events."""
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class BaseRecorderManagerCfg(RecorderManagerBaseCfg):
    """Base recorder configuration with common settings. By default, proprio data is recorded.

    Note: Camera extrinsics are recorded automatically as part of initial_state under
    the 'cameras' key. Use InitialStateRecorderCfg.camera_names to filter which cameras
    are recorded (default: all cameras).
    """
    record_initial_state: InitialStateRecorderCfg = InitialStateRecorderCfg()
    record_states: PostStepStatesRecorderCfg = PostStepStatesRecorderCfg()
    record_actions: PreStepActionsRecorderCfg = PreStepActionsRecorderCfg()
    record_ee_pose: PostStepEndEffectorPoseRecorderCfg = PostStepEndEffectorPoseRecorderCfg()
    record_robot_root_pose: PostStepRobotRootPoseRecorderCfg = PostStepRobotRootPoseRecorderCfg()
    record_bbox: PostStepBBoxRecorderCfg = PostStepBBoxRecorderCfg()
    record_policy_observations: PreStepFlatPolicyObservationsRecorderCfg | None = None
    record_subtask_completion: SubtaskCompletionRecorderCfg | None = None
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL

def create_recorder_config(
    include_policy_observations: bool = False,
    include_subtask_tracking: bool = False,
    export_dir: str | None = None,
    filename: str = "data.hdf5"
) -> RecorderManagerBaseCfg:
    """Factory function to create appropriate recorder configuration.

    Args:
        include_policy_observations: Whether to record policy observations (images, etc.)
        include_subtask_tracking: Whether to track subtask completion
        export_dir: Directory to export data to
        filename: Name of the output file

    Returns:
        Configured RecorderManagerBaseCfg instance
    """
    # Create base config
    config = BaseRecorderManagerCfg(
        export_in_record_pre_reset=False,
        dataset_export_dir_path=export_dir,
        dataset_filename=filename
    )

    # Conditionally add policy observations
    if include_policy_observations:
        config.record_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()

    # Conditionally add subtask tracking
    if include_subtask_tracking:
        config.record_subtask_completion = SubtaskCompletionRecorderCfg()

    return config

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""

@configclass
class RobolabDefaultEnvCfg(ManagerBasedRLEnvCfg):
    observations = None
    actions = None
    rewards = None
    commands = None
    events = None
    curriculum = None
    recorders = None
    rerender_on_reset: bool = True
    seed: int | None = 0
    subtasks: list[dict[str, dict]] | None = None

    def __post_init__(self):
        if self.observations is None:
            self.observations = ObservationCfg()
        if self.actions is None:
            self.actions = ActionCfg()
        if self.rewards is None:
            self.rewards = RewardsCfg()
        if self.commands is None:
            self.commands = CommandsCfg()
        if self.events is None:
            self.events = BaseEventCfg()
        if self.curriculum is None:
            self.curriculum = CurriculumCfg()
        if self.recorders is None:
            # Determine recorder configuration based on flags
            enable_subtask_tracking = robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING and self.subtasks is not None
            enable_policy_observations = robolab.constants.RECORD_IMAGE_DATA

            # Log configuration
            print(f"[INFO] Subtask progress checking {'ON' if enable_subtask_tracking else 'OFF'}")
            print(f"[INFO] Image observations recording {'ON' if enable_policy_observations else 'OFF'}")

            # Create recorder configuration
            self.recorders = create_recorder_config(
                include_policy_observations=enable_policy_observations,
                include_subtask_tracking=enable_subtask_tracking,
                export_dir=get_output_dir(),
                filename="data.hdf5"
            )

        self.viewer.cam_prim_path = "/OmniverseKit_Persp"
        self.viewer.eye = (1.5, 0.0, 1.0)
        self.viewer.lookat = (0.2, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 8
        self.scene.env_spacing = 2.0
        self.sim.use_fabric = True

        # PhysX settings. Isaac Lab 3 stores backend config under
        # ``sim.physics`` instead of the Isaac Lab 2 ``sim.physx`` field.
        if self.sim.physics is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_temp_buffer_capacity = 2**30
        self.sim.physics.gpu_heap_capacity = 2**30
        self.sim.physics.gpu_collision_stack_size = 2**30
        self.sim.physics.enable_ccd = True
        self.sim.physics.max_position_iteration_count = 32
        self.sim.physics.max_velocity_iteration_count = 1
        self.sim.physics.bounce_threshold_velocity = 0.2
        self.sim.physics.solver_type = 1
        # Floor for solver position iterations (PhysX clamps each actor's count to [min, max]).
        # isaacsim5 forced 32 [1]; isaaclab3 default is 1 [2] -> scene objects under-solve and the
        # gripper sinks into soft contacts. Restore the 32 floor.
        #   [1] https://github.com/petkokp/RoboLab/blob/7d45d74/robolab/core/environments/base.py#L171
        #   [2] https://github.com/isaac-sim/IsaacLab/blob/c372ae9/source/isaaclab/isaaclab/sim/simulation_cfg.py#L47
        self.sim.physics.min_position_iteration_count = 32
        # NOT a restore: this flag did not exist in isaacsim5 -- added in isaaclab v2.3 / Isaac Sim
        # 5.1+, default False [1]. Enabled as the vendor's purpose-built fix for our exact symptom:
        # the compliant finger's joint drive overrides the contact each solver iteration and re-enters
        # the grasped object; solving articulation contacts last prevents that [2][3]. Correctness here
        # is EMPIRICAL (True-vs-False ablation), not historical -- see the gripper ablation results.
        #   [1] https://github.com/isaac-sim/IsaacLab/blob/c372ae9/source/isaaclab/isaaclab/sim/simulation_cfg.py#L186
        #   [2] https://github.com/isaac-sim/IsaacLab/pull/3502
        #   [3] https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order
        self.sim.physics.solve_articulation_contact_last = True
