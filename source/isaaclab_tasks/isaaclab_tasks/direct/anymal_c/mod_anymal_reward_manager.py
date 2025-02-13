import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

from .mod_anymal_c_env_cfg import WalkingRewardCfg, SitUnsitRewardCfg
from .mod_anymal_command_manager import CustomCommandManager

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
        # z_vel_error = torch.square(robot.data.root_lin_vel_b[:, 2]) # RVMod
        if self._z_is_vel:
            z_error = torch.square(robot.data.root_lin_vel_b[:, 2] - commands[:, 3]) # RVMod
        else:
            z_error = torch.square(robot.data.root_com_pos_w[:, 2] - commands[:, 3])
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
            "lin_vel_z_l2": z_error * self.walk_reward_weights.z_vel_reward_scale * step_dt, # RVMod
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
        ## Original reward calculation
        walking_reward = torch.sum(torch.stack(list(walking_rewards.values()))[:, walking_mask], dim=0) # (N_walking)
        # task_keys = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp"]
        # aux_keys = [key for key in walking_rewards.keys() if key not in task_keys]
        # walking_reward = self.compute_walk_these_ways_reward(walking_rewards, walking_mask, task_keys, aux_keys)
        # r_walking_task = torch.sum(torch.stack([walking_rewards[key] for key in task_keys])[:, walking_mask], dim=0)
        # r_walking_aux = torch.sum(torch.stack([walking_rewards[key] for key in aux_keys])[:, walking_mask], dim=0)
        # r_walking = r_walking_task * torch.exp(r_walking_aux * 0.02)
        
        # Logging
        for key, value in walking_rewards.items():
            self.episode_sums[f"walking/{key}"] += value

        ### Compute sitting and unsitting rewards
        z_error_mapped = torch.exp(-z_error / 0.25) # RVMod
        flat_orientation_mapped = torch.exp(-flat_orientation / 0.25) # RVMod
        sit_or_unsit_rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.sit_unsit_reward_weights.lin_vel_reward_scale * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.sit_unsit_reward_weights.yaw_rate_reward_scale * step_dt,
            "z_error": z_error_mapped * self.sit_unsit_reward_weights.z_reward_scale * step_dt, # RVMod
            "ang_vel_xy_l2": ang_vel_error * self.sit_unsit_reward_weights.ang_vel_reward_scale * step_dt,
            "dof_torques_l2": joint_torques * self.sit_unsit_reward_weights.joint_torque_reward_scale * step_dt,
            "dof_acc_l2": joint_accel * self.sit_unsit_reward_weights.joint_accel_reward_scale * step_dt,
            "action_rate_l2": action_rate * self.sit_unsit_reward_weights.action_rate_reward_scale * step_dt,
            "undesired_contacts": contacts * self.sit_unsit_reward_weights.undesired_contact_reward_scale * step_dt,
            # "flat_orientation_l2": flat_orientation * self.sit_unsit_reward_weights.flat_orientation_reward_scale * step_dt,
            "flat_orientation_l2": flat_orientation_mapped * self.sit_unsit_reward_weights.flat_orientation_reward_scale * step_dt,
        }
        ## Original reward calculation
        sit_or_unsit_reward = torch.sum(torch.stack(list(sit_or_unsit_rewards.values()))[:, ~walking_mask], dim=0) # (N_sit_unsit)
        # task_keys = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "z_error"]
        # aux_keys = [key for key in sit_or_unsit_rewards.keys() if key not in task_keys]
        # sit_or_unsit_reward = self.compute_walk_these_ways_reward(sit_or_unsit_rewards, ~walking_mask, task_keys, aux_keys)
        
        # Logging
        for key, value in sit_or_unsit_rewards.items():
            self.episode_sums[f"sit_or_unsit/{key}"] += value
            
        ### Aggregate final rewards
        reward = torch.zeros(size=(self._num_envs,), device=self._device)
        reward[walking_mask] = walking_reward
        reward[~walking_mask] = sit_or_unsit_reward
        return reward
    
    def compute_walk_these_ways_reward(self, reward_dict: dict[str, torch.Tensor], mask, task_keys, aux_keys) -> torch.Tensor:
        """Computes the reward for the walk_these_ways task."""
        r_task = torch.sum(torch.stack([reward_dict[key] for key in task_keys])[:, mask], dim=0)
        r_aux = torch.sum(torch.stack([reward_dict[key] for key in aux_keys])[:, mask], dim=0)
        r_total = r_task * torch.exp(r_aux * 0.02)
        return r_total
