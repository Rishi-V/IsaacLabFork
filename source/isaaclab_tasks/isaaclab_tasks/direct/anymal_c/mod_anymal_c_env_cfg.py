# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import torch


@configclass
class EventCfg:
    """Configuration for randomization."""

    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    pass

######################################################################
####################### Command Configurations ########################
#region
@configclass
class CommandCfg:
    prob_sit = 0.5
    
#endregion
######################### End Command Configurations ####################

######################################################################
####################### Reward Configurations ########################
#region
#### Simple ones
@configclass
class WalkingRewardCfg:
    # reward scales
    # lin_vel_reward_scale = 0.02
    # yaw_rate_reward_scale = 0.01
    # z_vel_reward_scale = -4e-4
    # ang_vel_reward_scale = -2e-5
    # joint_torque_reward_scale = -2e-5
    # joint_accel_reward_scale = -5e-9
    # action_rate_reward_scale = -2e-3
    # feet_air_time_reward_scale = 0 # Keep but change based on task
    # undesired_contact_reward_scale = -0.02 # RVMod: Originally -1.0
    # flat_orientation_reward_scale = 0
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 1.0
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-05
    joint_accel_reward_scale = -2.5e-07
    action_rate_reward_scale = -0.0001
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -10.0
    flat_orientation_reward_scale = -1

@configclass
class SitUnsitRewardCfg:
    # reward scales
    lin_vel_reward_scale = 0.2
    yaw_rate_reward_scale = 0.2
    z_is_vel = False
    z_reward_scale = 5.0 # Change to z-height with positive reward
    ang_vel_reward_scale = -0.005
    joint_torque_reward_scale = -2e-5
    joint_accel_reward_scale = -5e-9
    action_rate_reward_scale = -0.0001
    undesired_contact_reward_scale = -10.0 # RVMod: Originally -1.0
    flat_orientation_reward_scale = -0.5

### More complex ones
# @configclass
# class WalkingRewardCfg:
#     # reward scales
#     lin_vel_reward_scale = 1.0
#     yaw_rate_reward_scale = 0.5
#     z_vel_reward_scale = -2.0 # Reward z-axis velocity reward
#     ang_vel_reward_scale = -0.05
#     joint_torque_reward_scale = -2.5e-5
#     joint_accel_reward_scale = -2.5e-7
#     action_rate_reward_scale = -0.01
#     feet_air_time_reward_scale = 0.5
#     undesired_contact_reward_scale = -10.0 # RVMod: Originally -1.0
#     flat_orientation_reward_scale = -5.0
    
# @configclass
# class SitUnsitRewardCfg:
#     # reward scales
#     lin_vel_reward_scale = 0.5
#     yaw_rate_reward_scale = 0.5
#     z_track_reward_scale = 10.0
#     # Maybe switch to reach target height with positive error term
#     # exp(-0.5 * (z_target - z_current)^2)
#     ang_vel_reward_scale = -0.05
#     joint_torque_reward_scale = -2.5e-5
#     joint_accel_reward_scale = -2.5e-7
#     action_rate_reward_scale = -0.01
#     undesired_contact_reward_scale = -1.0 # RVMod: Originally -1.0
#     flat_orientation_reward_scale = -5.0
#endregion
######################### End Reward Configurations ####################
########################################################################


@configclass
class ModAnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    # observation_space = 48+37+1 #RVMod: Was 48, now +1 as adding additional command
    observation_space = 48+1 #RVMod: Was 48, now +1 as adding additional command
    # observation_space = 236 # If including raycaster (cannot though)
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    # events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    
    # static other anymal
    # static_anymal: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/StaticAnymal")
    # static_anymal.init_state.pos = (0.0, 0.8, 0.6)
    
    # we add a height scanner for perceptive locomotion
    # height_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     # mesh_prim_paths=["/World/ground"],
    #     mesh_prim_paths=["/World/envs/env_.*/StaticAnymal"],
    # )


    # reward scales
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = 2.0 # Reward z-axis velocity reward
    # # z_pos_reward_scale = -5.0 # RVMod: Want to track z-axis position
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    # undesired_contact_reward_scale = -10.0 # RVMod: Originally -1.0
    # flat_orientation_reward_scale = -5.0
    command_cfg: CommandCfg = CommandCfg()
    walking_reward_cfg: WalkingRewardCfg = WalkingRewardCfg()
    situnsit_reward_cfg: SitUnsitRewardCfg = SitUnsitRewardCfg()