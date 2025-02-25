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

def assertIndicesNotBoolmask(env_ids: torch.Tensor):
    # assert env_ids.dtype == torch.long, "env_ids should be a tensor of indices, not a boolean mask"
    pass
    
def convertBoolmaskToIndices(env_ids: torch.Tensor):
    return torch.where(env_ids)[0]

class AbstractSkill(ABC):
    WALKING_HEIGHT = 0.6
    SITTING_HEIGHT = 0.1
    
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
        self._successful_timesteps = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands = torch.zeros(size=(num_envs, 4), device=self._device) #(x,y,yaw,z)
        self._raw_commands[:, 3] = AbstractSkill.WALKING_HEIGHT
    
    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        assertIndicesNotBoolmask(env_ids)
        if self._randomize:
            # Randomly sample from [-1,1] for x,y,yaw
            self._raw_commands[env_ids, :3] = torch.rand(size=(len(env_ids), 3), device=self._device) * 2 - 1
        else:
            self._raw_commands[env_ids, :3] = torch.tensor(self.dir, device=self._device).repeat(len(env_ids), 1)
        # Note: Don't need to set the z-axis command as it is always the same from 
        self._successful_timesteps[env_ids] = 0
        self._current_timestep[env_ids] = 0
        
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._raw_commands[env_ids]
    
    def get_failures(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        tipping_threshold = 0.8  # Define a tipping threshold, note that torch.norm(projected_gravity_b) is 1.0
        died = torch.norm(robot.data.projected_gravity_b[env_ids, :2], dim=1) > tipping_threshold
        return died | (self._current_timestep[env_ids] > self._timeout)
    
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        return self._successful_timesteps[env_ids] > self._holdtime
    
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        self._current_timestep[env_ids] += 1
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
        successful_walks = env_ids & lin_vel_error & yaw_rate_error # (N)
        self._successful_timesteps[successful_walks] += 1
    
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
        xyz_commands[:, 2] = xyz_commands[:, 2] - robot.data.root_com_pos_w[:, 2]
            
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
    z_reward_scale = 2.0 # Change to z-height with positive reward
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
        self._successful_timesteps = torch.zeros(size=(num_envs,), device=self._device)
        self._sitting_height = torch.zeros(size=(num_envs,), device=self._device)
        self._raw_commands = torch.zeros(size=(num_envs, 4), device=self._device)

    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        assertIndicesNotBoolmask(env_ids)
        if self._ztarget_type == "random":
            # self._sitting_height[env_ids] = torch.rand(size=(len(env_ids),), device=self._device) * (AbstractSkill.WALKING_HEIGHT - AbstractSkill.SITTING_HEIGHT) + AbstractSkill.SITTING_HEIGHT
            self._sitting_height[env_ids] = AbstractSkill.SITTING_HEIGHT + torch.rand(size=(len(env_ids),), device=self._device) * 0.4
        elif self._ztarget_type == "sitting":
            self._sitting_height[env_ids] = AbstractSkill.SITTING_HEIGHT
        elif self._ztarget_type == "walking":
            self._sitting_height[env_ids] = AbstractSkill.WALKING_HEIGHT
        else:
            raise ValueError(f"Unknown ztarget_type: {self._ztarget_type}")
        self._raw_commands[env_ids, 3] = self._sitting_height[env_ids]
        self._current_timestep[env_ids] = 0
        self._successful_timesteps[env_ids] = 0
        
    def get_raw_command(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._raw_commands[env_ids]
    
    def get_failures(self, env_ids, robot) -> torch.Tensor:
        tipping_threshold = 0.8
        died = torch.norm(robot.data.projected_gravity_b[env_ids, :2], dim=1) > tipping_threshold
        return died | (self._current_timestep[env_ids] > self._timeout)
    
    def get_successes(self, env_ids: torch.Tensor, robot: Articulation) -> torch.Tensor:
        return self._successful_timesteps[env_ids] > self._holdtime
    
    def update(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        self._current_timestep[env_ids] += 1
        sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self._sitting_height) < 0.1 # (N)
        successful_sits = env_ids & sitting_robots # (N)
        self._successful_timesteps[successful_sits] += 1
    
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
        xyz_commands[:, 2] = xyz_commands[:, 2] - robot.data.root_com_pos_w[:, 2] # Change to z direction
        
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
    
    
    
@configclass
class SequenceOfSkillsCfg:
    skill_sequence = [
        ("ReachZSkill", ReachZSkillCfg(timeout=400, ztarget_type="random", holdtime=50)),
        ("ReachZSkill", ReachZSkillCfg(timeout=400, ztarget_type="walking", holdtime=50))]
    reset_on_intermediate_failures = False
    dts_memory = 100
    
class SequenceOfSkills(AbstractSkill):
    def __init__(self, skill_sequence: list[AbstractSkill], reset_on_intermediate_failures: bool, dts_memory=100):
        """Takes in a sequence of skills and executes them in order. Note no total timeout as individual skills have their own timeouts.

        Args:
            skill_sequence (list[AbstractSkill]): Sequence of skills
            reset_on_intermediate_failures (bool): Whether to reset on intermediate failures. Recommended to be False.
            dts_memory (int, optional): Defaults to 100.
        """
        # timeout = 0
        # for skill in skill_sequence:
        #     timeout += skill._timeout
        # assert timeout > 0, "Timeout must be greater than 0"
        super().__init__(timeout=0, dts_memory=dts_memory)
        self._skill_sequence = skill_sequence
        self._reset_on_intermediate_failures = reset_on_intermediate_failures
        # self._env_to_skill_index stores the index of the current skill for each env
        self._env_to_skill_index: torch.Tensor # = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.long)
        
    def set_non_params(self, num_envs, device):
        super().set_non_params(num_envs, device)
        for skill in self._skill_sequence:
            skill.set_non_params(num_envs, device)
        # Initialize the env_to_skill_index to 0 for all envs
        self._env_to_skill_index = torch.zeros(size=(num_envs,), device=self._device, dtype=torch.long)
        
    def set_new_internals(self, env_ids: torch.Tensor, robot: Articulation) -> None:
        assertIndicesNotBoolmask(env_ids)
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
        if self._reset_on_intermediate_failures: # If reset on intermerdiate failures, check all skills
            for i, skill in enumerate(self._skill_sequence):
                skill_env_ids = env_ids & (self._env_to_skill_index == i) # (N)
                if skill_env_ids.any():
                    failures[skill_env_ids] = skill.get_failures(skill_env_ids, robot) # (E) boolean
        else: # If not, just check last skill
            last_index = len(self._skill_sequence) - 1
            skill_env_ids = env_ids & (self._env_to_skill_index == last_index)
            if skill_env_ids.any():
                failures[skill_env_ids] = self._skill_sequence[last_index].get_failures(skill_env_ids, robot)
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
                increment_envs[skill_env_ids] = skill.get_successes(skill_env_ids, robot) # (E) boolean
                if not self._reset_on_intermediate_failures: # If not reset, instead increment if failed
                    increment_envs[skill_env_ids] |= skill.get_failures(skill_env_ids, robot) # (E) boolean
                
        ### Increment envs that finished a skill and set new internals for the next skill
        # Note: Finishing the last skill will increment the index but will not set new internals
        self._env_to_skill_index[increment_envs] += 1
        for i, skill in enumerate(self._skill_sequence):
            new_skill_envs = increment_envs & (self._env_to_skill_index == i) # (N)
            if new_skill_envs.any():
                skill.set_new_internals(convertBoolmaskToIndices(new_skill_envs), robot)
        
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
        assert torch.all(rewards[env_ids] != 0), "All rewards should be non-zero"
        return rewards[env_ids]
    
    
@configclass
class DynamicSkillCfg:
    # skills: list[tuple[str, configclass, float]] = [
    #     ("WalkSkill", WalkSkillCfg(), 0.5),
    #     # ("WalkSkill", WalkSkillCfg(), 0.5),
    #     # ("ReachZSkill", ReachZSkillCfg(), 0.5)
    #     ]
    
    skills: list[tuple[str, configclass, float]] = [
        ("WalkSkill", WalkSkillCfg(), 0.5),
        ("SequenceOfSkillsCfg", SequenceOfSkillsCfg(), 1.0),
        ("ReachZSkill", ReachZSkillCfg(ztarget_type="sitting"), 0.5)]
    
def parse_cfg_skills(skill_name, skill_cfg) -> AbstractSkill:
    if skill_name == "WalkSkill":
        skill_cfg["reward_cfg"] = WalkSkillRewardCfg(**skill_cfg["reward_cfg"])
        skill = WalkSkill(**skill_cfg)
    elif skill_name == "ReachZSkill":
        skill_cfg["reward_cfg"] = ReachZSkillRewardCfg(**skill_cfg["reward_cfg"])
        skill = ReachZSkill(**skill_cfg)
    elif skill_name == "SequenceOfSkillsCfg":
        skill_cfg["skill_sequence"] = [parse_cfg_skills(skill_name, skill_cfg) for skill_name, skill_cfg in skill_cfg["skill_sequence"]]
        skill = SequenceOfSkills(**skill_cfg)
    else:
        raise ValueError(f"Unknown skill name: {skill_name}")
    return skill
    
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
            skill = parse_cfg_skills(skill_name, skill_cfg)
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
