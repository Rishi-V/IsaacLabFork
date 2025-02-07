# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .mod_anymal_c_env_cfg import ModAnymalCFlatEnvCfg, WalkingRewardCfg, SitUnsitRewardCfg

## Visualizations
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
# from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

"""taskset -c 80-120 python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Mod-Flat-Anymal-C-Direct-v0 \
--headless --video --video_length=200 --video_interval=2000 --num_envs=1024"""

class CustomCommandManager:
    def __init__(self, num_envs: int, device: torch.device, z_is_vel: bool):
        self._num_envs = num_envs
        self._device = device
        self.z_is_vel = z_is_vel
        self.SITTING_HEIGHT = 0.05
        self.WALKING_HEIGHT = 0.6
        self.PROB_SIT = 0.5
        self.MAX_Z_VEL = 0.1
        
        self._high_level_commands = torch.zeros(size=(self._num_envs,), device=self._device) # (N); -1=sit, 1=unsit, 0=walk
        self._raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device) # (N,4); (x,y,yaw,z) velocities or positions
        self._time_doing_action = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent doing action
        
    def update_commands(self, robot: Articulation):
        ### Sitting commands
        sitting_envs = self._high_level_commands == -1 # (N)
        sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.SITTING_HEIGHT) < 0.05 # (N)
        successful_sits = torch.logical_and(sitting_envs, sitting_robots) # (N)
        self._time_doing_action[successful_sits] += 1
        
        ### Unsitting commands
        unsitting_envs = self._high_level_commands == 1 # (N)
        unsitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.WALKING_HEIGHT) < 0.05 # (N)
        successful_unsits = torch.logical_and(unsitting_envs, unsitting_robots) # (N)
        self._time_doing_action[successful_unsits] += 1
        
        ### Walking commands
        walking_envs = self._high_level_commands == 0 # (N)
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
        successful_walks = torch.logical_and(lin_vel_error, yaw_rate_error) # (N)
        self._time_doing_action[successful_walks] += 1
        
        ### Update high_level actions
        finished_sitting_envs = torch.logical_and(sitting_envs, self._time_doing_action > 50) # (N), 1 second
        finished_unsitting_envs = torch.logical_and(unsitting_envs, self._time_doing_action > 50) # (N), 1 second
        finished_walking_envs = torch.logical_and(walking_envs, self._time_doing_action > 300) # (N), 10 seconds
        self._high_level_commands[finished_sitting_envs] = 1 # Make them unsit
        self._high_level_commands[finished_unsitting_envs] = 0 # Make them sit walk
        self._high_level_commands[finished_walking_envs] = -1 # Make them sit
        self._time_doing_action[finished_sitting_envs] = 0
        self._time_doing_action[finished_unsitting_envs] = 0
        self._time_doing_action[finished_walking_envs] = 0
        
        ### Update raw commands
        # Sitting raw commands
        sitting_envs = self._high_level_commands == -1 # (N)
        self._raw_commands[sitting_envs,:3] = 0.0
        if self.z_is_vel:
            error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_envs,2] # negative if robot is above sitting height
            self._raw_commands[sitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        else:
            self._raw_commands[sitting_envs,3] = self.SITTING_HEIGHT
        # Unsitting raw commands
        unsitting_envs = self._high_level_commands == 1 # (N)
        self._raw_commands[unsitting_envs,:3] = 0.0
        if self.z_is_vel:
            error = self.WALKING_HEIGHT - robot.data.root_com_pos_w[unsitting_envs,2] # positive if robot is below walking height
            self._raw_commands[unsitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        else:
            self._raw_commands[unsitting_envs,3] = self.WALKING_HEIGHT
        # Walking raw commands
        # new_walking_envs = torch.logical_and(walking_envs, self._time_doing_action > 100) # (N)
        new_walking_envs = torch.where(finished_unsitting_envs)[0] # Needs to be list of indices
        self.set_random_walk_commands(new_walking_envs)
        
    def reset_commands(self, env_ids: torch.Tensor, robot: Articulation):
        """env_ids: (E)
        Note: env_ids should be a tensor of indices, not boolean mask"""
        ## Split new envs into walking and sitting
        prob_sit = torch.rand(size=(len(env_ids),), device=self._device) # (E)
        bool_list = torch.where(prob_sit < self.PROB_SIT, 0, 1) # (E)
        walking_inds = env_ids[torch.where(bool_list == 1)[0]] # (E)
        sitting_inds = env_ids[torch.where(bool_list == 0)[0]] # (E)
        # self._high_level_commands[walking_inds] = torch.where(prob_sit < self.PROB_SIT, -1, 0) # (E)
        
        ## Set random walking commands
        self._high_level_commands[walking_inds] = 0
        self.set_random_walk_commands(walking_inds)
        
        ## Set sitting commands
        self._high_level_commands[sitting_inds] = -1
        self._raw_commands[sitting_inds, :3] = 0.0
        if self.z_is_vel:
            self._raw_commands[sitting_inds, 3] = -self.MAX_Z_VEL
        else:
            self._raw_commands[sitting_inds, 3] = self.SITTING_HEIGHT
        # error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_inds,2] # negative if robot is above sitting height
        # self._raw_commands[sitting_envs,3] = torch.clamp(error, -0.1, 0.1)
        
        ## Reset time doing action
        self._time_doing_action[env_ids] = 0
        
    def set_random_walk_commands(self, env_ids: torch.Tensor):
        """env_ids: (E)
        Note: env_ids should be a tensor of indices, not boolean mask"""
        
        self._raw_commands[env_ids] = torch.zeros(size=(len(env_ids), 4), device=self._device).uniform_(-1.0, 1.0)
        self._raw_commands[env_ids,3] = 0.0 # z-axis velocity is 0.0
        
    def get_commands(self) -> torch.Tensor:
        return self._raw_commands
    
    def get_high_level_commands(self) -> torch.Tensor:
        return self._high_level_commands
    
    def set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "command_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy() # Blue denotes moving
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = f"/Visuals/Command/robot/blue_arrow"
                self.command_visualizer = VisualizationMarkers(marker_cfg)
                
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy() # Red denotes sitting
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = f"/Visuals/Command/robot/red_arrow"
                self._marker_cfg = marker_cfg
                self.sit_unsit_visualizer = VisualizationMarkers(marker_cfg)
                # marker_cfg.markers["arrow"].size = (0.05, 0.05, 0.05)
                # marker_cfg = CUBOID_MARKER_CFG.copy()
                # marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                # set their visibility to true
                self.command_visualizer.set_visibility(True)
                self.sit_unsit_visualizer.set_visibility(True)
        else:
            if hasattr(self, "command_visualizer"):
                self.command_visualizer.set_visibility(False)
                self.sit_unsit_visualizer.set_visibility(False)
        
    def debug_vis_callback(self, robot: Articulation):
        target_loc = robot.data.root_com_pos_w.clone()  # (N,3)
        target_loc[:, 2] += 0.5
        
        walking_envs = self._high_level_commands == 0
        sit_or_unsit_envs = self._high_level_commands != 0
        xyz_commands = self._raw_commands[:, [0,1,3]].clone()
        if not self.z_is_vel: 
            # If it is position, then sit_or_unsit_envs should have the z-value adjusted
            xyz_commands[sit_or_unsit_envs, 2] = robot.data.root_com_pos_w[sit_or_unsit_envs, 2] - xyz_commands[sit_or_unsit_envs, 2]
            
        arrow_scale, arrow_quat = get_arrow_settings(self._marker_cfg, xyz_commands, robot, self._device)
        if walking_envs.sum() > 0:
            self.command_visualizer.visualize(translations=target_loc[walking_envs], orientations=arrow_quat[walking_envs], scales=arrow_scale[walking_envs])
        if sit_or_unsit_envs.sum() > 0:
            self.sit_unsit_visualizer.visualize(translations=target_loc[sit_or_unsit_envs], orientations=arrow_quat[sit_or_unsit_envs], scales=arrow_scale[sit_or_unsit_envs])
        

def get_arrow_settings(arrow_cfg: VisualizationMarkersCfg, xyz_velocity: torch.Tensor, 
                       robot: Articulation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts the XYZ base velocity command to arrow direction rotation."""
    # obtain default scale of the marker
    default_scale = arrow_cfg.markers["arrow"].scale
    # arrow-scale
    arrow_scale = torch.tensor(default_scale, device=device).repeat(xyz_velocity.shape[0], 1)
    arrow_scale[:, 0] *= torch.clamp(torch.linalg.norm(xyz_velocity, dim=1), min=0.2) * 3.0
    # arrow-direction
    heading_angle = torch.atan2(xyz_velocity[:, 1], xyz_velocity[:, 0])
    zeros = torch.zeros_like(heading_angle)
    pitch_angle = torch.atan2(xyz_velocity[:, 2], torch.linalg.norm(xyz_velocity[:,:2], dim=1)) # Add negative sign to z-axis?
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, heading_angle)
    # convert everything back from base to world frame
    base_quat_w = robot.data.root_quat_w
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

    return arrow_scale, arrow_quat

class CustomRewardManager:
    def __init__(self, num_envs: int, device: torch.device, command_manager: CustomCommandManager,
                 walk_reward_weights: WalkingRewardCfg, sit_unsit_reward_weights: SitUnsitRewardCfg,
                 z_is_vel: bool):
        self._num_envs = num_envs
        self._device = device
        self._rewards = torch.zeros(size=(self._num_envs,), device=self._device)
        self._command_manager = command_manager
        self.walk_reward_weights = walk_reward_weights
        self.sit_unsit_reward_weights = sit_unsit_reward_weights
        self._z_is_vel = z_is_vel
        self.episode_sums = {}
        
        walking_reward_components = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2",
                                        "dof_acc_l2", "action_rate_l2", "feet_air_time", "undesired_contacts", "flat_orientation_l2"]
        for key in walking_reward_components:
            self.episode_sums[f"walking/{key}"] = torch.zeros(size=(self._num_envs,), device=self._device)
        sit_unsit_reward_components = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "z_error", "ang_vel_xy_l2", "dof_torques_l2",
                                        "dof_acc_l2", "action_rate_l2", "undesired_contacts", "flat_orientation_l2"]
        for key in sit_unsit_reward_components:
            self.episode_sums[f"sit_or_unsit/{key}"] = torch.zeros(size=(self._num_envs,), device=self._device)
    
    def compute_rewards(self, robot: Articulation, 
                                actions: torch.Tensor, previous_actions: torch.Tensor,
                                contact_sensor: ContactSensor,
                                step_dt,
                                feet_ids, undesired_contact_body_ids: list[int]) -> torch.Tensor:
        commands = self._command_manager.get_commands()
        walking_mask = self._command_manager.get_high_level_commands() == 0
        
        ### Compute reward components
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(robot.data.root_lin_vel_b[:, 2]) # RVMod
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(actions - previous_actions), dim=1)
        # feet air time
        first_contact = contact_sensor.compute_first_contact(step_dt)[:, feet_ids]
        last_air_time = contact_sensor.data.last_air_time[:, feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces: torch.Tensor = contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

        ### Compute walking rewards
        walking_rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.walk_reward_weights.lin_vel_reward_scale * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.walk_reward_weights.yaw_rate_reward_scale * step_dt,
            "lin_vel_z_l2": z_vel_error * self.walk_reward_weights.z_vel_reward_scale * step_dt, # RVMod
            # "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            # "track_z_pos_l2": z_pos_error * self.cfg.z_pos_reward_scale * self.step_dt, # RVMod: z-axis position tracking
            "ang_vel_xy_l2": ang_vel_error * self.walk_reward_weights.ang_vel_reward_scale * step_dt,
            "dof_torques_l2": joint_torques * self.walk_reward_weights.joint_torque_reward_scale * step_dt,
            "dof_acc_l2": joint_accel * self.walk_reward_weights.joint_accel_reward_scale * step_dt,
            "action_rate_l2": action_rate * self.walk_reward_weights.action_rate_reward_scale * step_dt,
            "feet_air_time": air_time * self.walk_reward_weights.feet_air_time_reward_scale * step_dt,
            "undesired_contacts": contacts * self.walk_reward_weights.undesired_contact_reward_scale * step_dt,
            "flat_orientation_l2": flat_orientation * self.walk_reward_weights.flat_orientation_reward_scale * step_dt,
        }
        walking_reward = torch.sum(torch.stack(list(walking_rewards.values()))[:, walking_mask], dim=0) # (N_walking)
        # Logging
        for key, value in walking_rewards.items():
            self.episode_sums[f"walking/{key}"] += value

        ### Compute sitting and unsitting rewards
        if self._z_is_vel:
            z_error = torch.square(robot.data.root_lin_vel_b[:, 2] - commands[:, 3]) # RVMod
        else:
            z_error = torch.square(robot.data.root_com_pos_w[:, 2] - commands[:, 3])
        z_error_mapped = torch.exp(-z_error / 0.25) # RVMod
        sit_or_unsit_rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.sit_unsit_reward_weights.lin_vel_reward_scale * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.sit_unsit_reward_weights.yaw_rate_reward_scale * step_dt,
            "z_error": z_error_mapped * self.sit_unsit_reward_weights.z_reward_scale * step_dt, # RVMod
            "ang_vel_xy_l2": ang_vel_error * self.sit_unsit_reward_weights.ang_vel_reward_scale * step_dt,
            "dof_torques_l2": joint_torques * self.sit_unsit_reward_weights.joint_torque_reward_scale * step_dt,
            "dof_acc_l2": joint_accel * self.sit_unsit_reward_weights.joint_accel_reward_scale * step_dt,
            "action_rate_l2": action_rate * self.sit_unsit_reward_weights.action_rate_reward_scale * step_dt,
            "undesired_contacts": contacts * self.sit_unsit_reward_weights.undesired_contact_reward_scale * step_dt,
            "flat_orientation_l2": flat_orientation * self.sit_unsit_reward_weights.flat_orientation_reward_scale * step_dt,
        }
        sit_or_unsit_reward = torch.sum(torch.stack(list(sit_or_unsit_rewards.values()))[:, ~walking_mask], dim=0) # (N_sit_unsit)
        # Logging
        for key, value in sit_or_unsit_rewards.items():
            self.episode_sums[f"sit_or_unsit/{key}"] += value
            
        ### Aggregate final rewards
        reward = torch.zeros(size=(self._num_envs,), device=self._device)
        reward[walking_mask] = walking_reward
        reward[~walking_mask] = sit_or_unsit_reward
        return reward



class ModAnymalCEnv(DirectRLEnv):
    cfg: ModAnymalCFlatEnvCfg

    def __init__(self, cfg: ModAnymalCFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device) # (N,12)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        ) # (N,12)

        # Command manager
        z_is_vel = cfg.situnsit_reward_cfg.z_is_vel
        self.command_manager = CustomCommandManager(self.num_envs, self.device, z_is_vel=z_is_vel)
        self._commands = self.command_manager.get_commands()

        # Reward manager
        self.reward_manager = CustomRewardManager(self.num_envs, self.device, self.command_manager, 
                                                  cfg.walking_reward_cfg, cfg.situnsit_reward_cfg,
                                                  z_is_vel)

        # Logging
        # self._episode_sums = {
        #     key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #     for key in [
        #         "track_lin_vel_xy_exp",
        #         "track_ang_vel_z_exp",
        #         "lin_vel_z_l2",
        #         # "track_z_pos_l2", # RVMod: z-axis position tracking
        #         "ang_vel_xy_l2",
        #         "dof_torques_l2",
        #         "dof_acc_l2",
        #         "action_rate_l2",
        #         "feet_air_time",
        #         "undesired_contacts",
        #         "flat_orientation_l2",
        #     ]
        # }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self.set_debug_vis(debug_vis=cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # Add height scanner
        # self._height_scanner = RayCaster(self.cfg.height_scanner)
        # self.scene.sensors["height_scanner"] = self._height_scanner
        
        ### Add static anymal
        # self._static_anymal = Articulation(self.cfg.static_anymal)
        # self.scene.articulations["static_anymal"] = self._static_anymal
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self.command_manager.update_commands(self._robot) # Update actions before getting observations
        self._previous_actions = self._actions.clone()
        # height_data = (
        #     self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        # ).clip(-1.0, 1.0)
        obs = torch.cat([self._robot.data.root_lin_vel_b, # (N,3): Remove from actor (critic is okay)
                    self._robot.data.root_ang_vel_b, # (N,3)
                    self._robot.data.projected_gravity_b, # (N,3)
                    # self._commands, # (N,4)
                    self.command_manager.get_commands(), # (N,4)
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,12)
                    self._robot.data.joint_vel, # (N,12)
                    # height_data,
                    self._actions, # (N,12)
                    # self.get_static_anymal_obs(), # (N,37)
                    ], dim=-1)
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        self._commands = self.command_manager.get_commands()
        rewards = self.reward_manager.compute_rewards(self._robot, self._actions, self._previous_actions, self._contact_sensor, 
                                            self.step_dt, self._feet_ids, self._undesired_contact_body_ids)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        tipping_threshold = 0.8  # Define a tipping threshold, note that torch.norm(projected_gravity_b) is 1.0
        died = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > tipping_threshold
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        ### Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        ### Reset static anymal
        # random_position = torch.rand(size=(len(env_ids), 3), device=self.device) * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        # default_root_state = self._static_anymal.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids] #+ random_position
        # self._static_anymal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # joint_vel = self._static_anymal.data.default_joint_vel[env_ids]
        # self._static_anymal.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._static_anymal.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        ### Sample new commands
        self.command_manager.reset_commands(env_ids, self._robot)
        
        # Logging
        extras = dict()
        for key in self.reward_manager.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.reward_manager.episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self.reward_manager.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)


    def get_static_anymal_obs(self) -> torch.Tensor:
        static_pos = self._static_anymal.data.root_com_pos_w # (N,3)
        static_lin_vel = self._static_anymal.data.root_lin_vel_b # (N,3)
        static_ang_vel = self._static_anymal.data.root_ang_vel_b # (N,3)
        static_joint_pos = self._static_anymal.data.joint_pos - self._static_anymal.data.default_joint_pos # (N,12)
        static_joint_vel = self._static_anymal.data.joint_vel # (N,12)
        
        cur_pos = self._robot.data.root_com_pos_w # (N,3)
        cur_orienation = self._robot.data.root_quat_w # (N,4)
        static_orientation = self._static_anymal.data.root_quat_w # (N,4)
        relative_pos = static_pos - cur_pos # (N,3)
        relative_orientation = math_utils.quat_mul(cur_orienation, math_utils.quat_inv(static_orientation)) # (N,4)
        relative_lin_vel = static_lin_vel - self._robot.data.root_lin_vel_b # (N,3)
        relative_ang_vel = static_ang_vel - self._robot.data.root_ang_vel_b # (N,3)
        
        obs = torch.cat([relative_pos, # (N,3)
                        relative_lin_vel, # (N,3)
                        relative_ang_vel, # (N,3)
                        relative_orientation, # (N,4)
                        static_joint_pos, # (N,12)
                        static_joint_vel], # (N,12)
                    dim=-1) # (3+3+3+4+12+12) = (N,37)
        return obs # (N,37)


    def _set_debug_vis_impl(self, debug_vis: bool):
        self.command_manager.set_debug_vis_impl(debug_vis)
        
    def _debug_vis_callback(self, event):
        self.command_manager.debug_vis_callback(self._robot)

        