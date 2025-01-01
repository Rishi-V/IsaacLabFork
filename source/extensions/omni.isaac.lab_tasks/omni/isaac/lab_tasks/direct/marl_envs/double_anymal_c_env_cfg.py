# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectMARLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG, ANYDRIVE_3_LSTM_ACTUATOR_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    # physics_material1 = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot1", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )
    # physics_material2 = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot2", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass1 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot1", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    # add_base_mass2 = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot2", body_names="base"),
    #         "mass_distribution_params": (-5.0, 5.0),
    #         "operation": "add",
    #     },
    # )
    pass

ANYMAL_C_CFG_COLOR = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)

@configclass
class DoubleAnymalCFlatEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    possible_agents = ["robot1", "robot2"]
    action_spaces = {"robot1": 12, "robot2": 12}
    action_scales = {"robot1": 0.5, "robot2": 0.5}
    observation_spaces = {"robot1": 48, "robot2": 48}
    state_space = -1

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
    events: EventCfg = EventCfg()

    # robot
    robot_cfg1: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    robot_cfg1.init_state.pos = (-0.7, -0.7, 0.6)
    contact_sensor1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot1/.*", history_length=3, update_period=0.005, track_air_time=True,
        debug_vis=False # RVMod
    )
    robot_cfg2: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot2")
    robot_cfg2.init_state.pos = (0.7, 0.7, 0.6)
    contact_sensor2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot2/.*", history_length=3, update_period=0.005, track_air_time=True,
        debug_vis=True # RVMod
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0

