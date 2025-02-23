import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

## Visualizations
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
# from .mod_anymal_c_env_cfg import CustomCommandCfg, DynamicSkillCfg
# from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker
from abc import ABC, abstractmethod
from isaaclab.utils import configclass
import pdb

class AbstractSkill(ABC):
    WALKING_HEIGHT = 0.6
    SITTING_HEIGHT = 0.2
    
    def __init__(self, timeout: float, dts_memory=100):
        self._num_envs: int
        self._device: torch.device
        self._timeout = timeout
        self._alpha = 1
        self._beta = 1
        self._C = dts_memory # Max sum of alpha and beta for DTS
        
    def set_non_params(self, num_envs: int, device: torch.device):
        self._num_envs = num_envs
        self._device = device
    
    @abstractmethod
    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        """Sets the new internals for the given env_ids

        Args:
            env_ids (torch.Tensor): (E) indices
            robot (Articulation): Robot
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Returns the (E,4) raw command tensor for the given env_ids

        Args:
            env_ids (torch.Tensor): (N) boolean mask

        Returns:
            torch.Tensor: (E,4) raw command tensor
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_failures(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        """Returns a (N) boolean vector of envs that have failed the skill

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot (Articulation): Robot

        Returns:
            torch.Tensor: (E) boolean vector of envs that have failed the skill
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        """Returns a (N) boolean vector of envs that have completed the skill

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot (Articulation): Robot

        Returns:
            torch.Tensor: (E) boolean vector of envs that have completed the skill
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        """Updates the internals for the given env_ids, e.g., timesteps, raw_commands, etc.

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot (Articulation): Robot
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def debug_vis_callback(self, env_ids: torch.Tensor, robot: Articulation):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def compute_rewards(self, env_ids: torch.Tensor, robot: Articulation, 
                                actions: torch.Tensor, previous_actions: torch.Tensor,
                                contact_sensor: ContactSensor, step_dt: float,
                                feet_ids: list[int], undesired_contact_body_ids: list[int]) -> torch.Tensor:
        """Returns the (E) reward tensor for the given env_ids. Also logs the reward components.

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot (Articulation): Robot

        Returns:
            torch.Tensor: reward vector
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def update_success_rate(self, success: int, failures: int):
        """Updates the success rate of the skill

        Args:
            success (int): Number of successes
            failures (int): Number of failures
        """
        self._alpha += success
        self._beta += failures
        if self._alpha + self._beta > self._C:
            self._alpha *= self._C / (self._alpha + self._beta)
            self._beta *= self._C / (self._alpha + self._beta)
            
    def get_success_rate(self) -> float:
        """Returns the success rate of the skill

        Returns:
            float: Success rate of the skill
        """
        return self._alpha / (self._alpha + self._beta)

    def __repr__(self):
        IGNORED_PARAMS = ["_num_envs", "_device", "_timeout", "_alpha", "_beta", "_C"]
        params = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if k not in IGNORED_PARAMS)
        return f"{self.__class__.__name__}({params}, success_rate={self.get_success_rate():.2f})"

@configclass
class WalkSkillRewardCfg:
    lin_vel_reward_scale = 2.0
    yaw_rate_reward_scale = 1.0
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-05
    joint_accel_reward_scale = -2.5e-07
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1 #-1.0
    flat_orientation_reward_scale = -1

@configclass
class WalkSkillCfg:
    reward_cfg = WalkSkillRewardCfg()
    timeout = 400
    dir = (0.0, 0.0, 0.0)
    holdtime = 50
    randomize = True
    dts_memory = 100

class WalkSkill(AbstractSkill):
    def __init__(self, reward_cfg: WalkSkillRewardCfg, timeout: float, dir: tuple[float, float, float], 
                 holdtime: int, randomize: bool, dts_memory=100):
        """
        dir: (x,y,yaw) direction to walk in
        """
        super().__init__(timeout, dts_memory)
        self.dir = dir
        self._holdtime = holdtime
        self._randomize = randomize
        
        ## Internals that get updated
        self._current_timestep: torch.Tensor # = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands: torch.Tensor # = torch.zeros(size=(num_envs, 4), device=self._device) #(x,y,yaw,z)
        self._reward_cfg: WalkSkillRewardCfg = reward_cfg
        
    def set_non_params(self, num_envs, device):
        super().set_non_params(num_envs, device)
        self._current_timestep = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands = torch.zeros(size=(num_envs, 4), device=self._device) #(x,y,yaw,z)
        self._raw_commands[:, 3] = AbstractSkill.WALKING_HEIGHT
    
    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        if self._randomize:
            # Randomly sample from [-1,1] for x,y,yaw
            self._raw_commands[env_ids, :3] = torch.rand(size=(len(env_ids), 3), device=self._device) * 2 - 1
        else:
            self._raw_commands[env_ids, :3] = torch.tensor(self.dir, device=self._device).repeat(len(env_ids), 1)
        # Note: Don't need to set the z-axis command as it is always the same from __init__
        self._current_timestep[env_ids] = 0
        
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._raw_commands[env_ids]
    
    def get_failures(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        tipping_threshold = 0.8  # Define a tipping threshold, note that torch.norm(projected_gravity_b) is 1.0
        died = torch.norm(robot.data.projected_gravity_b[env_ids, :2], dim=1) > tipping_threshold
        return died | (self._current_timestep[env_ids] > self._timeout)
    
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        return self._current_timestep[env_ids] > self._holdtime
    
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
        successful_walks = env_ids & lin_vel_error & yaw_rate_error # (N)
        self._current_timestep[successful_walks] += 1
    
    def set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_visualizer_marker"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy() # Blue denotes moving
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = f"/Visuals/Command/robot/walk_arrow"
                self._visualizer_marker = VisualizationMarkers(marker_cfg)
                self._visualizer_marker.set_visibility(True)
                self._marker_cfg = marker_cfg
        else:
            if hasattr(self, "_visualizer_marker"):
                self._visualizer_marker.set_visibility(False)
                
    def debug_vis_callback(self, env_ids: torch.Tensor, robot: Articulation):
        target_loc = robot.data.root_com_pos_w.clone()  # (N,3)
        target_loc[:, 2] += 0.5
        
        xyz_commands = self._raw_commands[:, [0,1,3]].clone()
            
        arrow_scale, arrow_quat = get_arrow_settings(self._marker_cfg, xyz_commands, robot, self._device)
        self._visualizer_marker.visualize(translations=target_loc[env_ids], orientations=arrow_quat[env_ids], scales=arrow_scale[env_ids])     
    
    def compute_rewards(self, env_ids: torch.Tensor, robot: Articulation, 
                                actions: torch.Tensor, previous_actions: torch.Tensor,
                                contact_sensor: ContactSensor, step_dt: float,
                                feet_ids: list[int], undesired_contact_body_ids: list[int]) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z position tracking
        z_error = torch.square(robot.data.root_com_pos_w[:, 2] - self._raw_commands[:, 3])
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
        last_air_time = contact_sensor.data.last_air_time[:, feet_ids] # Ignore red squiggles
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._raw_commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces: torch.Tensor = contact_sensor.data.net_forces_w_history # Ignore red squiggles
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

        ### Compute rewards
        rewards_dict = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self._reward_cfg.lin_vel_reward_scale * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self._reward_cfg.yaw_rate_reward_scale * step_dt,
            "lin_vel_z_l2": z_error * self._reward_cfg.z_vel_reward_scale * step_dt, # RVMod
            "ang_vel_xy_l2": ang_vel_error * self._reward_cfg.ang_vel_reward_scale * step_dt,
            "dof_torques_l2": joint_torques * self._reward_cfg.joint_torque_reward_scale * step_dt,
            "dof_acc_l2": joint_accel * self._reward_cfg.joint_accel_reward_scale * step_dt,
            "action_rate_l2": action_rate * self._reward_cfg.action_rate_reward_scale * step_dt,
            "feet_air_time": air_time * self._reward_cfg.feet_air_time_reward_scale * step_dt,
            "undesired_contacts": contacts * self._reward_cfg.undesired_contact_reward_scale * step_dt,
            "flat_orientation_l2": flat_orientation * self._reward_cfg.flat_orientation_reward_scale * step_dt,
        }
        rewards = torch.sum(torch.stack(list(rewards_dict.values()))[:, env_ids], dim=0) # (E)
        return rewards
    
    
@configclass
class ReachZSkillRewardCfg:
    lin_vel_reward_scale = 0.2
    yaw_rate_reward_scale = 0.2
    z_reward_scale = 1.0 # Change to z-height with positive reward
    flat_orientation_reward_scale = 0.5 # Change to positive reward
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2e-5
    joint_accel_reward_scale = -5e-9
    action_rate_reward_scale = -0.01
    undesired_contact_reward_scale = -1 #-1.0 # RVMod: Originally -1.0
    
@configclass
class ReachZSkillCfg:
    reward_cfg = ReachZSkillRewardCfg()
    timeout = 400
    holdtime = 50
    ztarget_type = "random"
    dts_memory = 100
    
class ReachZSkill(AbstractSkill):
    def __init__(self, reward_cfg: ReachZSkillRewardCfg, timeout: float, 
                 holdtime: int, ztarget_type: str, dts_memory=100):
        super().__init__(timeout, dts_memory)
        self._holdtime = holdtime
        self._ztarget_type = ztarget_type
        assert ztarget_type in ["random", "sitting", "walking"]
        # self.MAX_Z_VEL = 0.1 # Not needed as z is not velocity
        
        ## Internals that get updated
        self._current_timestep: torch.Tensor # = torch.zeros(size=(num_envs,), device=self._device)
        self._sitting_height: torch.Tensor # = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands: torch.Tensor # = torch.zeros(size=(num_envs, 4), device=self._device) #(x,y,yaw,z)
        self._reward_cfg: ReachZSkillRewardCfg = reward_cfg
        
    def set_non_params(self, num_envs, device):
        super().set_non_params(num_envs, device)
        self._current_timestep = torch.zeros(size=(num_envs,), device=self._device)
        self._sitting_height = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands = torch.zeros(size=(num_envs, 4), device=self._device)

    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        """env_ids: (E)"""
        if self._ztarget_type == "random":
            self._sitting_height[env_ids] = AbstractSkill.SITTING_HEIGHT + torch.rand(size=(len(env_ids),), device=self._device) * 0.2
        elif self._ztarget_type == "sitting":
            self._sitting_height[env_ids] = AbstractSkill.SITTING_HEIGHT
        elif self._ztarget_type == "walking":
            self._sitting_height[env_ids] = AbstractSkill.WALKING_HEIGHT
        else:
            raise ValueError(f"Unknown ztarget_type: {self._ztarget_type}")
        self._raw_commands[env_ids, 3] = self._sitting_height[env_ids]
        self._current_timestep[env_ids] = 0
        
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._raw_commands[env_ids]
    
    def get_failures(self, env_ids, robot) -> torch.Tensor:
        tipping_threshold = 0.8
        died = torch.norm(robot.data.projected_gravity_b[env_ids, :2], dim=1) > tipping_threshold
        return died | (self._current_timestep[env_ids] > self._timeout)
    
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        return self._current_timestep[env_ids] > self._holdtime
    
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self._sitting_height) < 0.1 # (N)
        successful_sits = env_ids & sitting_robots # (N)
        self._current_timestep[successful_sits] += 1
    
    def set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_visualizer_marker"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy() # Ignore red squiggles
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                marker_cfg.prim_path = f"/Visuals/Command/robot/sit_arrow"
                self._visualizer_marker = VisualizationMarkers(marker_cfg)
                self._visualizer_marker.set_visibility(True)
                self._marker_cfg = marker_cfg
        else:
            if hasattr(self, "_visualizer_marker"):
                self._visualizer_marker.set_visibility(False)
                
    def debug_vis_callback(self, env_ids: torch.Tensor, robot: Articulation):
        target_loc = robot.data.root_com_pos_w.clone()
        target_loc[:, 2] += 0.5
        
        xyz_commands = self._raw_commands[:, [0,1,3]].clone()
        
        arrow_scale, arrow_quat = get_arrow_settings(self._marker_cfg, xyz_commands, robot, self._device)
        self._visualizer_marker.visualize(translations=target_loc[env_ids], orientations=arrow_quat[env_ids], scales=arrow_scale[env_ids])
        
    def compute_rewards(self, env_ids: torch.Tensor, robot: Articulation,
                                actions: torch.Tensor, previous_actions: torch.Tensor,
                                contact_sensor: ContactSensor, step_dt: float,
                                feet_ids: list[int], undesired_contact_body_ids: list[int]) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z position tracking
        z_error = torch.square(robot.data.root_com_pos_w[:, 2] - self._raw_commands[:, 3])
        z_error_mapped = torch.exp(-z_error / 0.25) # RVMod
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(actions - previous_actions), dim=1)
        # undesired contacts
        net_contact_forces: torch.Tensor = contact_sensor.data.net_forces_w_history # Ignore red squiggles
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
        flat_orientation_mapped = torch.exp(-flat_orientation / 0.25) # RVMod

        ### Compute walking rewards
        reward_dict = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self._reward_cfg.lin_vel_reward_scale * step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self._reward_cfg.yaw_rate_reward_scale * step_dt,
            "z_error": z_error_mapped * self._reward_cfg.z_reward_scale * step_dt, # RVMod
            "ang_vel_xy_l2": ang_vel_error * self._reward_cfg.ang_vel_reward_scale * step_dt,
            "dof_torques_l2": joint_torques * self._reward_cfg.joint_torque_reward_scale * step_dt,
            "dof_acc_l2": joint_accel * self._reward_cfg.joint_accel_reward_scale * step_dt,
            "action_rate_l2": action_rate * self._reward_cfg.action_rate_reward_scale * step_dt,
            "undesired_contacts": contacts * self._reward_cfg.undesired_contact_reward_scale * step_dt,
            # "flat_orientation_l2": flat_orientation * self._reward_cfg.flat_orientation_reward_scale * step_dt,
            "flat_orientation_l2": flat_orientation_mapped * self._reward_cfg.flat_orientation_reward_scale * step_dt,
        }
        ## Original reward calculation
        rewards = torch.sum(torch.stack(list(reward_dict.values()))[:, env_ids], dim=0) # (E)
        return rewards
    
    
    
class SequenceOfSkills(AbstractSkill):
    def __init__(self, timeout: float, skill_sequence: list[AbstractSkill], dts_memory=100):
        super().__init__(timeout, dts_memory)
        self._skill_sequence = skill_sequence
        # self._env_to_skill_index stores the index of the current skill for each env
        self._env_to_skill_index: torch.Tensor # = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.long)
        
    def set_non_params(self, num_envs, device):
        super().set_non_params(num_envs, device)
        for skill in self._skill_sequence:
            skill.set_non_params(num_envs, device)
        # Initialize the env_to_skill_index to 0 for all envs
        self._env_to_skill_index = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.long)
        
    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        """env_ids: (E)"""
        self._env_to_skill_index[env_ids] = 0
        self._skill_sequence[0].set_new_internals(env_ids, robot)
            
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device)
        for i, skill in enumerate(self._skill_sequence):
            skill_env_ids = env_ids & (self._env_to_skill_index == i) # (N)
            if skill_env_ids.any():
                raw_commands[skill_env_ids] = skill.get_raw_command(skill_env_ids)
        return raw_commands[env_ids]
    
    def get_failures(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        failures = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.bool)
        for i, skill in enumerate(self._skill_sequence):
            skill_env_ids = env_ids & (self._env_to_skill_index == i) # (N)
            if skill_env_ids.any():
                failures[skill_env_ids] = skill.get_failures(skill_env_ids, robot) # (E) boolean
        return failures[env_ids]
    
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        successes = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.bool) # (N)
        last_index = len(self._skill_sequence) - 1
        skill_env_ids = env_ids & (self._env_to_skill_index == last_index) # (N)
        if skill_env_ids.any():
            successes[skill_env_ids] = self._skill_sequence[last_index].get_successes(skill_env_ids, robot) # (E)
        return successes[env_ids]
    
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        increment_envs = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.bool)
        for i, skill in enumerate(self._skill_sequence):
            skill_env_ids = env_ids & (self._env_to_skill_index == i)
            if skill_env_ids.any():
                skill.update(skill_env_ids, robot)
                # If skill is finished, move to next skill
                finished_subskill = skill.get_successes(skill_env_ids, robot) # (N) boolean
                increment_envs |= finished_subskill
                
        ### Increment envs that finished a skill and set new internals for the next skill
        # Note: Finishing the last skill will increment the index but will not set new internals
        self._env_to_skill_index[increment_envs] += 1
        for i, skill in enumerate(self._skill_sequence):
            new_skill_envs = increment_envs & (self._env_to_skill_index == i) # (N)
            if new_skill_envs.any():
                skill.set_new_internals(new_skill_envs, robot)
        
        # Clamp the index to the last skill
        self._env_to_skill_index[env_ids] = torch.clamp(self._env_to_skill_index[env_ids], 0, len(self._skill_sequence) - 1)
    
    def set_debug_vis_impl(self, debug_vis: bool):
        for skill in self._skill_sequence:
            skill.set_debug_vis_impl(debug_vis)
            
    def debug_vis_callback(self, env_ids: torch.Tensor, robot: Articulation):
        for i, skill in enumerate(self._skill_sequence):
            skill_env_ids = env_ids & (self._env_to_skill_index == i)
            if skill_env_ids.any():
                skill.debug_vis_callback(skill_env_ids, robot)
                
    def compute_rewards(self, env_ids: torch.Tensor, robot: Articulation,
                                actions: torch.Tensor, previous_actions: torch.Tensor,
                                contact_sensor: ContactSensor, step_dt: float,
                                feet_ids: list[int], undesired_contact_body_ids: list[int]) -> torch.Tensor:
        rewards = torch.zeros(size=(self._num_envs,), device=self._device)
        for i, skill in enumerate(self._skill_sequence):
            skill_env_ids = env_ids & (self._env_to_skill_index == i)
            if skill_env_ids.any():
                rewards[skill_env_ids] = skill.compute_rewards(skill_env_ids, robot, actions, previous_actions,
                                                                contact_sensor, step_dt, feet_ids, undesired_contact_body_ids)
        assert torch.all(rewards != 0), "All rewards should be non-zero"
        return rewards[env_ids]
    
    
@configclass
class DynamicSkillCfg:
    skills: list[tuple[str, configclass, float]] = [
        ("WalkSkill", WalkSkillCfg(), 0.5),
        ("ReachZSkill", ReachZSkillCfg(), 0.5)]
    
class DynamicSkillManager:
    def __init__(self, num_envs: int, device: torch.device):
        self._num_envs = num_envs
        self._device = device
        self._skills: list[AbstractSkill] = []
        self._probs: list[float] = []
        
    def parse_cfg(self, skills_cfg: DynamicSkillCfg):
        self._skills.clear()
        self._probs.clear()
        for skill_name, skill_cfg, prob in skills_cfg.skills:
            # Note: For some reason skill_cfg is a dict, so we need to convert it to a configclass
            if skill_name == "WalkSkill":
                skill_cfg["reward_cfg"] = WalkSkillRewardCfg(**skill_cfg["reward_cfg"])
                skill = WalkSkill(**skill_cfg)
            elif skill_name == "ReachZSkill":
                skill_cfg["reward_cfg"] = ReachZSkillRewardCfg(**skill_cfg["reward_cfg"])
                skill = ReachZSkill(**skill_cfg)
            else:
                raise ValueError(f"Unknown skill name: {skill_name}")
            skill.set_non_params(self._num_envs, self._device)
            self._skills.append(skill)
            self._probs.append(prob)
            
        self._skill_indices = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.long)
        self._prob_tensor = torch.tensor(self._probs, device=self._device)
        
    def get_should_reset(self, robot: Articulation) -> torch.Tensor:
        """Returns a (N,) boolean vector of envs that should_be_reset"""
        should_be_reset = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.bool)
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                failures = skill.get_failures(env_ids, robot) # (E)
                successes = skill.get_successes(env_ids, robot) # (E)
                should_be_reset[env_ids] = failures | successes
                skill.update_success_rate(int(successes.sum().item()), int(failures.sum().item()))
        return should_be_reset
        
    def reset(self, env_ids: torch.Tensor, robot: Articulation):
        """Reset via sampling from commands
        env_ids: (E) indices"""
        self._skill_indices[env_ids] = torch.multinomial(self._prob_tensor, len(env_ids), replacement=True) # (E)
        for i, skill in enumerate (self._skills):
            new_skill_envs = env_ids[self._skill_indices[env_ids] == i] # (E)
            if len(new_skill_envs) > 0:
                skill.set_new_internals(new_skill_envs, robot)
            
    def get_raw_commands(self) -> torch.Tensor:
        raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device)
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                raw_commands[env_ids] = skill.get_raw_command(env_ids) # (E,4)
        return raw_commands
            
    def update(self, robot: Articulation):
        """Update the commands, called in get_observations"""
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                skill.update(env_ids, robot)
            
    def set_debug_vis_impl(self, debug_vis: bool):
        for skill in self._skills:
            skill.set_debug_vis_impl(debug_vis)
            
    def debug_vis_callback(self, robot: Articulation):
        for i, skill in enumerate(self._skills):
            skill_env_ids = self._skill_indices == i # (N)
            if skill_env_ids.any():
                skill.debug_vis_callback(skill_env_ids, robot)
                
    def compute_rewards(self, robot: Articulation, actions: torch.Tensor, previous_actions: torch.Tensor,
                         contact_sensor: ContactSensor, step_dt: float,
                         feet_ids: list[int], undesired_contact_body_ids: list[int]) -> torch.Tensor:
        """Returns a (N,) reward vector"""
        rewards = torch.zeros(size=(self._num_envs,), device=self._device)
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i
            if env_ids.any():
                rewards[env_ids] = skill.compute_rewards(env_ids, robot, actions, previous_actions,
                                                         contact_sensor, step_dt, feet_ids, undesired_contact_body_ids)
        assert torch.all(rewards != 0), "All rewards should be non-zero"
        return rewards
    


# class CustomCommandManager:
#     def __init__(self, num_envs: int, device: torch.device, cmd_cfg: CustomCommandCfg, z_is_vel: bool):
#         self._num_envs = num_envs
#         self._device = device
#         self.cmd_cfg = cmd_cfg
#         self.z_is_vel = z_is_vel
#         self.SITTING_HEIGHT = 0.10
#         self.WALKING_HEIGHT = 0.60
#         # self.PROB_SIT = cmd_cfg.prob_sit #0.5
#         self.MAX_Z_VEL = 0.1
        
#         self._high_level_commands = torch.zeros(size=(self._num_envs,), device=self._device) # (N); -1=sit, 1=unsit, 0=walk
#         self._raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device) # (N,4); (x,y,yaw,z) velocities or positions
#         self._time_doing_action = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent doing action
#         self._time_trying_command = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent trying to do action
        
#         self.parse_cfg_to_custom_command_sequence(cmd_cfg)
        
#     def parse_cfg_to_custom_command_sequence(self, cmd_cfg: CustomCommandCfg):
#         """high_level_command_sequence: (N,T,8)
#         - 8: (high_level_index, timeout, sample_or_done, x_vel, y_vel, yaw_vel, z_height)
#         - high_level_index: -1=sit, 1=unsit, 0=walk
#         - timeout: Maximum of timesteps to try to do action
#         - timehold: Number of timesteps to hold action
#         - sample_or_done: 0=sample raw commands, 1=don't sample, 2=done
#         - x_vel, y_vel, yaw_vel, z_height: Only use if don't sample (sample_or_done=1)
#         """
#         ## Custom command sequence
#         num_max_commands = cmd_cfg.max_cmd_length + 1 # +1 for done command
#         self._hl_sequence = torch.zeros(size=(self._num_envs, num_max_commands, 8), device=self._device) # (N,T,8)
#         self._hl_indices = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.long) # (N)
        
#         self.CC_IND_HL = 0
#         self.CC_IND_TIMEOUT = 1
#         self.CC_IND_TIMEHOLD = 2
#         self.CC_IND_SAMPLE_OR_DONE = 3
#         self.CC_IND_RAW_ACTIONS = 4
        
#         max_cmd_length = cmd_cfg.max_cmd_length # This is Tmax-1, we add the done command later
#         cmd_list = cmd_cfg.cmd_list
        
#         cmd_idx_to_tensor = torch.zeros(size=(len(cmd_list), max_cmd_length+1, 8), device=self._device) # (NumCommands,Tmax,8)
#         self._sample_probs = torch.zeros(size=(len(cmd_list),), device=self._device) # (NumCommands)
#         for i, cmd_seq_with_prob in enumerate(cmd_list):
#             cmd_seq, prob = cmd_seq_with_prob
#             self._sample_probs[i] = prob
#             cmd_seq = [self.map_str_to_tensor(cmd) for cmd in cmd_seq] # (Tcur-1,8)
#             cmd_seq.append(self.map_str_to_tensor("done")) # (Tcur<Tmax,8)
#             cmd_seq = torch.stack(cmd_seq) # Tensor (Tcur,8)
#             cmd_idx_to_tensor[i, :len(cmd_seq), :] = cmd_seq # (Tcur,8)
#         self._cmd_idx_to_tensor = cmd_idx_to_tensor # (NumCommands,Tmax,8)
    
#     def map_str_to_tensor(self, cmd_str: str):
#         """cmd_str: [sit, unsit, walk, r_sit, r_unsit, r_walk, done]
#         Returns: (high_level_index, timeout, timehold, sample_or_done, x_vel, y_vel, yaw_vel, z_height)
#         """
#         if cmd_str == "sit":
#             return torch.tensor([-1, 400, 50, 1, 0, 0, 0, self.SITTING_HEIGHT], device=self._device)
#         elif cmd_str == "r_sit":
#             return torch.tensor([-1, 400, 50, 0, 0, 0, 0, self.SITTING_HEIGHT], device=self._device)
#         elif cmd_str == "unsit":
#             return torch.tensor([1, 400, 50, 1, 0, 0, 0, self.WALKING_HEIGHT], device=self._device)
#         elif cmd_str == "r_unsit":
#             return torch.tensor([1, 400, 50, 0, 0, 0, 0, self.WALKING_HEIGHT], device=self._device)
#         elif cmd_str == "walk":
#             return torch.tensor([0, 400, 400, 1, 1, 0, 0, self.WALKING_HEIGHT], device=self._device)
#         elif cmd_str == "r_walk":
#             return torch.tensor([0, 400, 400, 0, 0, 0, 0, 0], device=self._device)
#         elif cmd_str == "done":
#             return torch.tensor([0, 0, 0, 2, 0, 0, 0, 0], device=self._device)
#         else:
#             raise ValueError(f"Unknown command: {cmd_str}")
        
#     def resample_hl_sequence(self, env_ids: torch.Tensor, robot: Articulation):
#         """env_ids: (E)
#         Note: env_ids should be a tensor of indices, not boolean mask"""
#         self._hl_indices[env_ids] = 0
#         self._time_doing_action[env_ids] = 0
        
#         ### Sample cmd indices
#         cmd_inds = torch.multinomial(self._sample_probs, len(env_ids), replacement=True) # (E)
#         self._hl_sequence[env_ids] = self._cmd_idx_to_tensor[cmd_inds] # (E,T,8)
        
#         # ### Resample hl_sequence based on cmd_cfg
#         # rand_sequence = (-1, 1, 0) # ["sit", "unsit", "walk"]
#         # self._hl_sequence[env_ids] = torch.zeros(size=(len(env_ids), 10, 8), device=self._device) # (E,T,8)
#         # self._hl_sequence[env_ids, :3, self.CC_IND_HL] = torch.tensor(rand_sequence, device=self._device).repeat(len(env_ids), 1)
#         # self._hl_sequence[env_ids, :3, self.CC_IND_SAMPLE_OR_DONE] = 0 # 0=sample raw commands
#         # self._time_trying_command[env_ids] = self._hl_sequence[env_ids, 0, self.CC_IND_TIMEOUT]
        
#     def set_custom_command_sequence(self, high_level_command_sequence: torch.Tensor):
#         self._hl_sequence = high_level_command_sequence # (N,T,8)
#         self._hl_indices = torch.zeros(size=(self._num_envs,), device=self._device) # (N)
        
#     def update_time_doing_action2(self, robot: Articulation):
#         self._time_trying_command -= 1
#         ### Sitting commands
#         sitting_envs = self._high_level_commands == -1 # (N)
#         sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.SITTING_HEIGHT) < 0.1 # (N)
#         successful_sits = torch.logical_and(sitting_envs, sitting_robots) # (N)
#         self._time_doing_action[successful_sits] += 1
        
#         ### Unsitting commands
#         unsitting_envs = self._high_level_commands == 1 # (N)
#         unsitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.WALKING_HEIGHT) < 0.1 # (N)
#         successful_unsits = torch.logical_and(unsitting_envs, unsitting_robots) # (N)
#         self._time_doing_action[successful_unsits] += 1
        
#         ### Walking commands
#         walking_envs = self._high_level_commands == 0 # (N)
#         lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
#         yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
#         successful_walks = torch.logical_and(walking_envs, torch.logical_and(lin_vel_error, yaw_rate_error)) # (N)
#         self._time_doing_action[successful_walks] += 1
        
#     def get_finished_envs2(self) -> torch.Tensor:
#         """Returns a boolean mask of environments that have finished their current high-level command"""
#         finished_envs = self._time_trying_command < 0 # (N)
#         finished_envs |= self._time_doing_action > self._hl_sequence[:, self._hl_indices, self.CC_IND_TIMEHOLD] # (N)
#         return finished_envs        

#     def update_hl_commands2(self, env_inds: torch.Tensor, robot: Articulation, finished_not_reset: bool):
#         """finished_env_inds: (N) or (E) depending on boolmask"""
#         if finished_not_reset:
#             env_inds = torch.where(env_inds)[0] # (E)
            
#             self._hl_indices[env_inds] += 1
#             next_hl_inds = self._hl_indices[env_inds] # (E)
    
#             ### Restart sequence for done envsl must be done before setting raw commands
#             sample_or_done = self._hl_sequence[env_inds, next_hl_inds, self.CC_IND_SAMPLE_OR_DONE] # (E)
#             done_envs = env_inds[sample_or_done == 2] # (E_done)
#             if len(done_envs) > 0:
#                 self.resample_hl_sequence(done_envs, robot=robot)
#         else:
#             self.resample_hl_sequence(env_inds, robot=robot)
            
#         next_hl_inds = self._hl_indices[env_inds] # (E) # Update next_hl_inds as some would be set to 0
#         self._time_doing_action[env_inds] = 0 # (E)
#         self._time_trying_command[env_inds] = self._hl_sequence[env_inds, next_hl_inds, self.CC_IND_TIMEOUT] # (E)
        
#         ### Set next high-level commands and raw commands; done after reset so will incorporate reset commands
#         next_hl_command = self._hl_sequence[env_inds, next_hl_inds, self.CC_IND_HL] # (E)
#         self._high_level_commands[env_inds] = next_hl_command # (E)
#         self._raw_commands[env_inds] = self._hl_sequence[env_inds, next_hl_inds, self.CC_IND_RAW_ACTIONS:] # (E,4)
        
#         ### Set randomly sampled raw commands. This will overwrite some of the raw commands above
#         sample_or_done = self._hl_sequence[env_inds, next_hl_inds, self.CC_IND_SAMPLE_OR_DONE] # (E)
#         sample_envs = sample_or_done == 0 # (E)
#         random_sit_envs = sample_envs & (next_hl_command == -1) # (E)
#         random_unsit_envs = sample_envs & (next_hl_command == 1) # (E)
#         random_walk_envs = sample_envs & (next_hl_command == 0) # (E)
#         self.set_random_sit_commands(env_inds[random_sit_envs], boolmask=False)
#         self.set_random_unsit_commands(env_inds[random_unsit_envs], boolmask=False)
#         self.set_random_walk_commands(env_inds[random_walk_envs], boolmask=False)
        
    
#     def update_raw_commands_based_on_robot2(self, robot: Articulation):
#         """Update raw commands based on the robot's current state.        
#         """
#         # Sitting raw commands
#         if self.z_is_vel:
#             sitting_envs = self._high_level_commands == -1 # (N)
#             self._raw_commands[sitting_envs,:3] = 0.0
#             error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_envs,2] # negative if robot is above sitting height
#             self._raw_commands[sitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
            
#         # Unsitting raw commands
#         if self.z_is_vel:
#             unsitting_envs = self._high_level_commands == 1 # (N)
#             self._raw_commands[unsitting_envs,:3] = 0.0
#             error = self.WALKING_HEIGHT - robot.data.root_com_pos_w[unsitting_envs,2] # positive if robot is below walking height
#             self._raw_commands[unsitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        
#     def update_commands2(self, robot: Articulation):
#         self.update_time_doing_action2(robot) # Updates time_doing_action based on the robot and high_level_commands
#         finished_envs = self.get_finished_envs2() # Gets envs that finished holding or timed out
#         self.update_hl_commands2(finished_envs, robot, finished_not_reset=True) # Updates high_level_commands, raw_commands, and time_trying_command based on finished_envs
#         self.update_raw_commands_based_on_robot2(robot) # Updates raw_commands based on the robot and high_level_commands
        
#     def reset_commands2(self, env_ids: torch.Tensor, robot: Articulation):
#         """env_ids: (E)
#         Note: env_ids should be a tensor of indices, not boolean mask"""
#         self.update_hl_commands2(env_ids, robot, finished_not_reset=False) # Updates high_level_commands, raw_commands, and time_trying_command based on finished_envs
#         self.update_raw_commands_based_on_robot2(robot) # Updates raw_commands based on the robot and high_level_commands
        
#     def update_commands(self, robot: Articulation):
#         self._time_trying_command -= 1
#         finished_trying_envs = self._time_trying_command < 0
        
#         ### Sitting commands
#         sitting_envs = self._high_level_commands == -1 # (N)
#         sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.SITTING_HEIGHT) < 0.1 # (N)
#         successful_sits = torch.logical_and(sitting_envs, sitting_robots) # (N)
#         self._time_doing_action[successful_sits] += 1
        
#         ### Unsitting commands
#         unsitting_envs = self._high_level_commands == 1 # (N)
#         unsitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.WALKING_HEIGHT) < 0.1 # (N)
#         successful_unsits = torch.logical_and(unsitting_envs, unsitting_robots) # (N)
#         self._time_doing_action[successful_unsits] += 1
        
#         ### Walking commands
#         walking_envs = self._high_level_commands == 0 # (N)
#         lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
#         yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
#         successful_walks = torch.logical_and(walking_envs, torch.logical_and(lin_vel_error, yaw_rate_error)) # (N)
#         self._time_doing_action[successful_walks] += 1
        
#         ### Update high_level actions
#         finished_sitting_envs = sitting_envs & ((self._time_doing_action > 50) | finished_trying_envs) # (N), 1 second
#         finished_unsitting_envs = unsitting_envs & ((self._time_doing_action > 50) | finished_trying_envs)  # (N), 1 second
#         finished_walking_envs = walking_envs & ((self._time_doing_action > 400) | finished_trying_envs) # (N), 10 seconds
#         new_sitting_envs = finished_walking_envs # finished_walking_envs # Make them unsit
#         new_unsitting_envs = finished_sitting_envs # finished_sitting_envs # Make them sit walk
#         new_walking_envs = finished_unsitting_envs # self._high_level_commands == 2 # finished_unsitting_envs # Make them sit, change command later
#         self._high_level_commands[new_unsitting_envs] = 1 
#         self._high_level_commands[new_walking_envs] = 0 
#         self._high_level_commands[new_sitting_envs] = -1 
#         self._time_doing_action[new_unsitting_envs] = 0
#         self._time_doing_action[new_walking_envs] = 0
#         self._time_doing_action[new_sitting_envs] = 0
#         self.set_time_trying_command(new_unsitting_envs, boolmask=True)
#         self.set_time_trying_command(new_walking_envs, boolmask=True)
#         self.set_time_trying_command(new_sitting_envs, boolmask=True)
        
#         ### Update raw commands
#         # Sitting raw commands
#         if self.z_is_vel:
#             sitting_envs = self._high_level_commands == -1 # (N)
#             self._raw_commands[sitting_envs,:3] = 0.0
#             error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_envs,2] # negative if robot is above sitting height
#             self._raw_commands[sitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
#         else: # Only update new sitting commands
#             self.set_random_sit_commands(new_sitting_envs, boolmask=True)
#             # self._raw_commands[sitting_envs,3] = self.SITTING_HEIGHT # (x_vel,y_vel,yaw_vel,z_height)
#         # Unsitting raw commands
#         if self.z_is_vel:
#             unsitting_envs = self._high_level_commands == 1 # (N)
#             self._raw_commands[unsitting_envs,:3] = 0.0
#             error = self.WALKING_HEIGHT - robot.data.root_com_pos_w[unsitting_envs,2] # positive if robot is below walking height
#             self._raw_commands[unsitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
#         else:
#             # self._raw_commands[unsitting_envs,3] = self.WALKING_HEIGHT
#             self.set_random_unsit_commands(new_unsitting_envs, boolmask=True)
#         # Walking raw commands
#         # new_walking_envs = torch.logical_and(walking_envs, self._time_doing_action > 100) # (N)
#         self.set_random_walk_commands(new_walking_envs, boolmask=True)
        
#     def reset_commands(self, env_ids: torch.Tensor, robot: Articulation):
#         """env_ids: (E)
#         Note: env_ids should be a tensor of indices, not boolean mask"""
#         ## Split new envs into walking and sitting
#         prob_sit = torch.rand(size=(len(env_ids),), device=self._device) # (E)
#         bool_list = torch.where(prob_sit < self.PROB_SIT, 0, 1) # (E)
#         sitting_inds = env_ids[torch.where(bool_list == 0)[0]] # (E)
#         walking_inds = env_ids[torch.where(bool_list == 1)[0]] # (E)
        
#         ## Set random walking commands
#         self._high_level_commands[walking_inds] = 0
#         self.set_random_walk_commands(walking_inds, boolmask=False)
        
#         ## Set sitting commands
#         self._high_level_commands[sitting_inds] = -1
#         self.set_random_sit_commands(sitting_inds, boolmask=False)
        
#         ## Reset time doing action
#         self._time_doing_action[env_ids] = 0
#         self.set_time_trying_command(env_ids, boolmask=False)
        
#     def set_time_trying_command(self, env_ids: torch.Tensor, boolmask: bool):
#         """env_ids: (E)
#         boolmask: bool indicating whether env_ids is a boolean mask or not"""
#         if boolmask:
#             self._time_trying_command[env_ids] = torch.zeros(size=(int(env_ids.sum().item()),), device=self._device).uniform_(400, 800)
#         else:
#             self._time_trying_command[env_ids] = torch.zeros(size=(len(env_ids),), device=self._device).uniform_(400, 800)
        
#     def set_random_walk_commands(self, env_ids: torch.Tensor, boolmask: bool):
#         """env_ids: (E)
#         boolmask: bool indicating whether env_ids is a boolean mask or not"""
#         if boolmask:
#             env_ids = torch.where(env_ids)[0]
        
#         self._raw_commands[env_ids] = torch.zeros(size=(len(env_ids), 4), device=self._device).uniform_(-1.0, 1.0)
#         if self.z_is_vel:
#             self._raw_commands[env_ids,3] = 0.0 # z-axis command is 0.0 velocity
#         else:
#             self._raw_commands[env_ids,3] = self.WALKING_HEIGHT # z-axis command is walking height
        
#     def set_random_sit_commands(self, env_ids: torch.Tensor, boolmask: bool):
#         """env_ids: (E)
#         boolmask: bool indicating whether env_ids is a boolean mask or not"""
#         if boolmask:
#             env_ids = torch.where(env_ids)[0]
        
#         self._raw_commands[env_ids, :3] = 0.0
#         if self.z_is_vel:
#             self._raw_commands[env_ids, 3] = -self.MAX_Z_VEL
#         else:
#             self._raw_commands[env_ids, 3] = torch.zeros(size=(len(env_ids),), 
#                                 device=self._device).uniform_(self.SITTING_HEIGHT, self.SITTING_HEIGHT+0.2)
            
#     def set_random_unsit_commands(self, env_ids: torch.Tensor, boolmask: bool):
#         """env_ids: (E)
#         boolmask: bool indicating whether env_ids is a boolean mask or not"""
#         if boolmask:
#             env_ids = torch.where(env_ids)[0]

#         self._raw_commands[env_ids, :3] = 0.0
#         if self.z_is_vel:
#             self._raw_commands[env_ids, 3] = self.MAX_Z_VEL
#         else:
#             self._raw_commands[env_ids, 3] = self.WALKING_HEIGHT
#             # torch.zeros(size=(len(env_ids), 4), device=self._device).uniform_(self.SITTING_HEIGHT, self.WALKING_HEIGHT)
        
#     def get_commands(self) -> torch.Tensor:
#         return self._raw_commands
    
#     def get_high_level_commands(self) -> torch.Tensor:
#         return self._high_level_commands
    
#     def set_debug_vis_impl(self, debug_vis: bool):
#         if debug_vis:
#             if not hasattr(self, "command_visualizer"):
#                 marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy() # Blue denotes moving
#                 marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
#                 marker_cfg.prim_path = f"/Visuals/Command/robot/blue_arrow"
#                 self.command_visualizer = VisualizationMarkers(marker_cfg)
                
#                 marker_cfg = RED_ARROW_X_MARKER_CFG.copy() # Red denotes sitting
#                 marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
#                 marker_cfg.prim_path = f"/Visuals/Command/robot/red_arrow"
#                 self._marker_cfg = marker_cfg
#                 self.sit_unsit_visualizer = VisualizationMarkers(marker_cfg)
#                 # marker_cfg.markers["arrow"].size = (0.05, 0.05, 0.05)
#                 # marker_cfg = CUBOID_MARKER_CFG.copy()
#                 # marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
#                 # -- goal pose
#                 # set their visibility to true
#                 self.command_visualizer.set_visibility(True)
#                 self.sit_unsit_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "command_visualizer"):
#                 self.command_visualizer.set_visibility(False)
#                 self.sit_unsit_visualizer.set_visibility(False)
        
#     def debug_vis_callback(self, robot: Articulation):
#         target_loc = robot.data.root_com_pos_w.clone()  # (N,3)
#         target_loc[:, 2] += 0.5
        
#         walking_envs = self._high_level_commands == 0
#         sit_or_unsit_envs = self._high_level_commands != 0
#         xyz_commands = self._raw_commands[:, [0,1,3]].clone()
#         if not self.z_is_vel: 
#             # If it is position, then sit_or_unsit_envs should have the z-value adjusted
#             xyz_commands[sit_or_unsit_envs, 2] = robot.data.root_com_pos_w[sit_or_unsit_envs, 2] - xyz_commands[sit_or_unsit_envs, 2]
            
#         arrow_scale, arrow_quat = get_arrow_settings(self._marker_cfg, xyz_commands, robot, self._device)
#         if walking_envs.sum() > 0:
#             self.command_visualizer.visualize(translations=target_loc[walking_envs], orientations=arrow_quat[walking_envs], scales=arrow_scale[walking_envs])
#         if sit_or_unsit_envs.sum() > 0:
#             self.sit_unsit_visualizer.visualize(translations=target_loc[sit_or_unsit_envs], orientations=arrow_quat[sit_or_unsit_envs], scales=arrow_scale[sit_or_unsit_envs])
        



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
