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

from .mod_anymal_c_env_cfg import ModAnymalCFlatEnvCfg

## Visualizations
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
# from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

"""python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Mod-Flat-Anymal-C-Direct-v0 \
--headless --video --video_length 200 --video_interval 1000 --num_envs 1024"""
class ModAnymalCEnv(DirectRLEnv):
    cfg: ModAnymalCFlatEnvCfg

    def __init__(self, cfg: ModAnymalCFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device) # (N,12)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        ) # (N,12)

        # X/Y linear velocity and yaw angular velocity commands
        # RVMod: Add z-axis target position command
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._commands[:,3] = 0.6
        self._SIT_HEIGHT = 0.1
        self._timesteps_before_switch = torch.zeros(self.num_envs, 4, device=self.device)+1000 # Start at 1000

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                # "lin_vel_z_l2",
                "track_z_pos_l2", # RVMod: z-axis position tracking
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
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
        self._static_anymal = Articulation(self.cfg.static_anymal)
        self.scene.articulations["static_anymal"] = self._static_anymal
        
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
        self._previous_actions = self._actions.clone()
        # height_data = (
        #     self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        # ).clip(-1.0, 1.0)
        obs = torch.cat([self._robot.data.root_lin_vel_b, # (N,3)
                    self._robot.data.root_ang_vel_b, # (N,3)
                    self._robot.data.projected_gravity_b, # (N,3)
                    self._commands, # (N,4)
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,12)
                    self._robot.data.joint_vel, # (N,12)
                    # height_data,
                    self._actions, # (N,12)
                    self.get_static_anymal_obs(), # (N,37)
                    ], dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # # z velocity tracking
        # z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # RVMod: z-axis position tracking
        z_pos_error = torch.square(self._robot.data.root_com_pos_w[:, 2] - self._commands[:, 3])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            # "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "track_z_pos_l2": z_pos_error * self.cfg.z_pos_reward_scale * self.step_dt, # RVMod: z-axis position tracking
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        tipping_threshold = 0.5  # Define a tipping threshold
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
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids,3] = 0.6
        # num_envs_to_sample = int(0.2 * len(env_ids))
        # sampled_envs = torch.randperm(len(env_ids))[:num_envs_to_sample]
        # self._commands[env_ids[sampled_envs], :3] = 0.0
        # self._commands[env_ids[sampled_envs], 3] = 0.05
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
        default_root_state = self._static_anymal.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids] #+ random_position
        self._static_anymal.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        joint_vel = self._static_anymal.data.default_joint_vel[env_ids]
        self._static_anymal.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._static_anymal.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
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

    def update_commands(self):
        # normal_envs = torch.where(self._commands[:,3] == 0.6) # (N1=number of environments where z-axis target position is 0.6)
        sit_command_envs = (self._commands[:, 3] == self._SIT_HEIGHT) # Binary vector of size N
        sit_actual_envs = (self._robot.data.root_com_pos_w[:, 2] < self._SIT_HEIGHT) # Binary vector of size N
        successful_sits = torch.logical_and(sit_command_envs, sit_actual_envs) # Binary vector of size N
        # Update z-axis target position command for environments where sit was successful
        self._commands[successful_sits,3] = 0.6
        
        pass


    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "command_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                # marker_cfg.markers["arrow"].size = (0.05, 0.05, 0.05)
                # marker_cfg = CUBOID_MARKER_CFG.copy()
                # marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = f"/Visuals/Command/robot/goal_position"
                self.command_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.command_visualizer.set_visibility(True)
        else:
            if hasattr(self, "command_visualizer"):
                self.command_visualizer.set_visibility(False)
        
    def _debug_vis_callback(self, event):
        target_loc = self._robot.data.root_com_pos_w.clone()  # (N,3)
        target_loc[:, 2] += 0.5
        
        # RVMod: Add orientation visualization
        # xdir = self._commands[:, 0]
        # ydir = self._commands[:, 1]
        # zdir = self._commands[:, 3] # 0.6 for standing, 0.0 for sitting
        # zdir = torch.where(self._commands[:, 3] == 0.6, 0.0, -1.0) # 0.0 for standing, 1.0 for sitting
        # quat = direction_to_quaternion(xdir, ydir, zdir)
        tmp = self._commands[:, [0,1,3]].clone()
        tmp[:, 2] = torch.where(tmp[:, 2] == 0.6, 0.0, -1.0)
        arrow_scale, arrow_quat = self._resolve_xyz_velocity_to_arrow(tmp)
        self.command_visualizer.visualize(translations=target_loc, orientations=arrow_quat, scales=arrow_scale)
        

    def _resolve_xyz_velocity_to_arrow(self, xyz_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XYZ base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.command_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xyz_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xyz_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xyz_velocity[:, 1], xyz_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        pitch_angle = torch.atan2(-xyz_velocity[:, 2], torch.linalg.norm(xyz_velocity[:,:2], dim=1)) # Note adding a negative sign to z-axis
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
        
def direction_to_quaternion(xdir, ydir, zdir):
    """
    Converts a direction vector (x, y, z) to a quaternion.
    
    Args:
    xdir (torch.Tensor): x direction component.
    ydir (torch.Tensor): y direction component.
    zdir (torch.Tensor): z direction component.
    
    Returns:
    torch.Tensor: Quaternion representing the orientation.
    """
    quat = torch.zeros((xdir.shape[0], 4), device=xdir.device)
    quat[:, 0] = 2 * (xdir * ydir + zdir)
    quat[:, 1] = 1 - 2 * (xdir ** 2 + zdir ** 2)
    quat[:, 2] = 2 * (ydir * zdir - xdir)
    quat[:, 3] = 1
    
    return quat