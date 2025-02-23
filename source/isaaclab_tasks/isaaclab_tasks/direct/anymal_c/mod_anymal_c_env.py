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

from .mod_anymal_c_env_cfg import ModAnymalCFlatEnvCfg #, WalkingRewardCfg, SitUnsitRewardCfg
from .mod_anymal_command_manager import DynamicSkillManager
# from .mod_anymal_reward_manager import CustomRewardManager

## Visualizations
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
# import isaaclab.utils.math as math_utils
# # from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

"""taskset -c 40-79 python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Mod-Flat-Anymal-C-Direct-v0 \
--headless --video --video_length=600 --video_interval=10000 --num_envs=1024"""




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
        # z_is_vel = cfg.situnsit_reward_cfg.z_is_vel
        # self.command_manager = CustomCommandManager(self.num_envs, self.device, cfg.command_cfg, z_is_vel=z_is_vel)
        # self._commands = self.command_manager.get_commands()
        self.skill_manager = DynamicSkillManager(self.num_envs, self.device)
        self.skill_manager.parse_cfg(cfg.dynamic_skill_cfg)

        # Reward manager
        # self.reward_manager = CustomRewardManager(self.num_envs, self.device, self.command_manager, 
        #                                           cfg.walking_reward_cfg, cfg.situnsit_reward_cfg,
        #                                           z_is_vel)

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
        self.skill_manager.update(self._robot)
        raw_commands = self.skill_manager.get_raw_commands()
        # self.command_manager.update_commands(self._robot) # Update actions before getting observations
        self._previous_actions = self._actions.clone()
        # height_data = (
        #     self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        # ).clip(-1.0, 1.0)
        obs = torch.cat([self._robot.data.root_lin_vel_b, # (N,3): Remove from actor (critic is okay)
                    self._robot.data.root_ang_vel_b, # (N,3)
                    self._robot.data.projected_gravity_b, # (N,3)
                    # self.command_manager.get_commands(), # (N,4)
                    raw_commands, # (N,4)
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,12)
                    self._robot.data.joint_vel, # (N,12)
                    # height_data,
                    self._actions, # (N,12)
                    # self.get_static_anymal_obs(), # (N,37)
                    ], dim=-1)
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        # self._commands = self.command_manager.get_commands()
        # rewards = self.reward_manager.compute_rewards(self._robot, self._actions, self._previous_actions, self._contact_sensor, 
        #                                     self.step_dt, self._feet_ids, self._undesired_contact_body_ids)
        rewards = self.skill_manager.compute_rewards(self._robot, self._actions, self._previous_actions, self._contact_sensor,
                                                        self.step_dt, self._feet_ids, self._undesired_contact_body_ids)
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # # died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        # tipping_threshold = 0.8  # Define a tipping threshold, note that torch.norm(projected_gravity_b) is 1.0
        # died = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > tipping_threshold
        # return died, time_out
        return self.skill_manager.get_should_reset(self._robot), time_out

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
        # self.command_manager.reset_commands(env_ids, self._robot)
        # self.command_manager.reset_commands2(env_ids, self._robot)
        self.skill_manager.reset(env_ids, self._robot)
        
        # Logging
        extras = dict()
        # for key in self.reward_manager.episode_sums.keys():
        #     episodic_sum_avg = torch.mean(self.reward_manager.episode_sums[key][env_ids])
        #     extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
        #     self.reward_manager.episode_sums[key][env_ids] = 0.0
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
        # self.command_manager.set_debug_vis_impl(debug_vis)
        self.skill_manager.set_debug_vis_impl(debug_vis)
        
    def _debug_vis_callback(self, event):
        # self.command_manager.debug_vis_callback(self._robot)
        self.skill_manager.debug_vis_callback(self._robot)

        