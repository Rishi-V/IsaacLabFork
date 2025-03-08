import torch

from typing import Optional
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
from dataclasses import MISSING
import pdb

from .single_quadruped import SingleQuadruped
from .skill_manager_single import AbstractSingleAgentSkill, WalkSkill, ReachZSkill, SequenceOfSkills, parse_single_quadruped_cfg_skills

class AbstractDoubleAgentSkill(ABC):
    @staticmethod
    @abstractmethod
    def create_config_dict() -> tuple[str, dict]:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @staticmethod
    @abstractmethod
    def create_reward_config_dict() -> dict:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    ############ Member functions ############
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
        self._timeout_vec = torch.zeros(size=(num_envs,), device=self._device).uniform_(self._timeout*0.8, self._timeout)
    
    @abstractmethod
    def set_new_internals(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> None:
        """Sets the new internals for the given env_ids

        Args:
            env_ids (torch.Tensor): (E) indices
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_raw_command(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns the (E,4) raw command tensor for the given env_ids

        Args:
            env_ids (torch.Tensor): (N) boolean mask

        Returns:
            dict[str, torch.Tensor]: robot name to (E,4) raw command tensor
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_failures(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> torch.Tensor:
        """Returns a (N) boolean vector of envs that have failed the skill

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances

        Returns:
            torch.Tensor: (E) boolean vector of envs that have failed the skill
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_successes(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> torch.Tensor:
        """Returns a (N) boolean vector of envs that have completed the skill

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances

        Returns:
            torch.Tensor: (E) boolean vector of envs that have completed the skill
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def update(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> None:
        """Updates the internals for the given env_ids, e.g., timesteps, raw_commands, etc.

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def set_debug_vis_impl(self, debug_vis: bool):
        """Sets the debug visualization implementation

        Args:
            debug_vis (bool): Whether to enable debug visualization
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def debug_vis_callback(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]):
        """Callback for debug visualization

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def compute_rewards(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> dict[str, torch.Tensor]:
        """Returns the (E) reward tensor for the given env_ids. Also logs the reward components.

        Args:
            env_ids (torch.Tensor): (N) boolean mask
            robot_dict (dict[str, SingleQuadruped]): Dictionary of robot names to SingleQuadruped instances

        Returns:
            dict[str, torch.Tensor]: Reward components
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
    
# @configclass
# class DoubleAgentSkillCfg:
#     robot1_name: str = "robot1"
#     robot2_name: str = "robot2"
#     skill1 = MISSING
#     skill2 = MISSING
#     timeout = MISSING
#     dts_memory = 100

class DoubleAgentSkillsFromSingleAgentSkills(AbstractDoubleAgentSkill):
    @staticmethod
    def create_config_dict(timeout: float,
                    skill1_config_tuple: tuple[str, dict], skill2_config_tuple: tuple[str, dict],
                    robot1_name: str = "robot1", robot2_name: str = "robot2", 
                    dts_memory=100, reward_dict: Optional[dict] = None) -> tuple[str, dict]:
        if reward_dict is None:
            reward_dict = DoubleAgentSkillsFromSingleAgentSkills.create_reward_config_dict()
        else:
            reward_dict = DoubleAgentSkillsFromSingleAgentSkills.create_reward_config_dict(**reward_dict)
        return ("DoubleAgentSkillsFromSingleAgentSkills", 
                    {"robot1_name": robot1_name,
                    "robot2_name": robot2_name,
                    "skill1_config_tuple": skill1_config_tuple,
                    "skill2_config_tuple": skill2_config_tuple,
                    "timeout": timeout,
                    "dts_memory": dts_memory,
                    "reward_dict": reward_dict})
        
    @staticmethod
    def create_reward_config_dict(weight1 = 0.5, weight2 = 0.5) -> dict:
        return {
            "weight1": weight1,
            "weight2": weight2
        }
    
    def __init__(self, robot1_name: str, skill1_config_tuple: tuple[str, dict], 
                    robot2_name: str, skill2_config_tuple: tuple[str, dict], reward_dict: dict, timeout: float, dts_memory=100):
        super().__init__(timeout, dts_memory)
        self.robot1_name = robot1_name
        self.robot2_name = robot2_name
        self.skill1 = parse_single_quadruped_cfg_skills(*skill1_config_tuple)
        self.skill2 = parse_single_quadruped_cfg_skills(*skill2_config_tuple)
        self.reward_dict = reward_dict # Note this is not used currently!

    def set_non_params(self, num_envs: int, device: torch.device):
        super().set_non_params(num_envs, device)
        self.skill1.set_non_params(num_envs, device)
        self.skill2.set_non_params(num_envs, device)

    def set_new_internals(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> None:
        # env_ids: (E) indices
        self.skill1.set_new_internals(env_ids, robot_dict[self.robot1_name])
        self.skill2.set_new_internals(env_ids, robot_dict[self.robot2_name])

    def get_raw_command(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # env_ids: (N) boolean mask
        command1 = self.skill1.get_raw_command(env_ids)  # (E,4)
        command2 = self.skill2.get_raw_command(env_ids)  # (E,4)
        return {self.robot1_name: command1, self.robot2_name: command2}

    def get_failures(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> torch.Tensor:
        # env_ids: (N) boolean mask
        failures1 = self.skill1.get_failures(env_ids, robot_dict[self.robot1_name])  # (E)
        failures2 = self.skill2.get_failures(env_ids, robot_dict[self.robot2_name])  # (E)
        return failures1 | failures2  # (E)

    def get_successes(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> torch.Tensor:
        # env_ids: (N) boolean mask
        successes1 = self.skill1.get_successes(env_ids, robot_dict[self.robot1_name])  # (E)
        successes2 = self.skill2.get_successes(env_ids, robot_dict[self.robot2_name])  # (E)
        return successes1 & successes2  # (E)

    def update(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> None:
        # env_ids: (N) boolean mask
        self.skill1.update(env_ids, robot_dict[self.robot1_name])
        self.skill2.update(env_ids, robot_dict[self.robot2_name])

    def set_debug_vis_impl(self, debug_vis: bool):
        self.skill1.set_debug_vis_impl(debug_vis)
        self.skill2.set_debug_vis_impl(debug_vis)

    def debug_vis_callback(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]):
        # env_ids: (N) boolean mask
        self.skill1.debug_vis_callback(env_ids, robot_dict[self.robot1_name])
        self.skill2.debug_vis_callback(env_ids, robot_dict[self.robot2_name])

    def compute_rewards(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]) -> dict[str, torch.Tensor]:
        # env_ids: (N) boolean mask
        rewards1 = self.skill1.compute_rewards(env_ids, robot_dict[self.robot1_name])  # (E)
        rewards2 = self.skill2.compute_rewards(env_ids, robot_dict[self.robot2_name])  # (E)
        return {self.robot1_name: rewards1, self.robot2_name: rewards2}


@configclass
class DoubleAgentDynamicSkillCfg:
    skills: list[tuple[str, dict, float]] = [
        (*DoubleAgentSkillsFromSingleAgentSkills.create_config_dict(timeout=400, 
            skill1_config_tuple=WalkSkill.create_config_dict(timeout=400, dir=(0, 0, 0), holdtime=20, 
                                                             randomize=True, reward_dict=None), 
            skill2_config_tuple=WalkSkill.create_config_dict(timeout=400, dir=(0, 0, 0), holdtime=20, 
                                                             randomize=True, reward_dict=None)), 
            1.0)
    ]

def parse_cfg_skills(skill_name: str, skill_cfg: dict) -> AbstractDoubleAgentSkill:
    if skill_name == "DoubleAgentSkillsFromSingleAgentSkills":
        skill = DoubleAgentSkillsFromSingleAgentSkills(**skill_cfg)
    else:
        raise ValueError(f"Unknown skill name: {skill_name}")
    return skill


class DoubleAgentDynamicSkillManager:
    def __init__(self, robot_names, num_envs: int, device: torch.device):
        self._num_envs = num_envs
        self._device = device
        self._skills: list[AbstractDoubleAgentSkill] = []
        self._probs: list[float] = []
        self._robot_names = robot_names
        
    def parse_cfg(self, skills_cfg: DoubleAgentDynamicSkillCfg):
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
        
    def get_should_reset(self, robot_dict: dict[str, SingleQuadruped]) -> torch.Tensor:
        """Returns a (N,) boolean vector of envs that should_be_reset"""
        should_be_reset = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.bool)
            
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                failures = skill.get_failures(env_ids, robot_dict) # (E)
                successes = skill.get_successes(env_ids, robot_dict) # (E)
                should_be_reset[env_ids] = failures | successes
                skill.update_success_rate(int(successes.sum().item()), int(failures.sum().item()))
        return should_be_reset
        
    def reset(self, env_ids: torch.Tensor, robot_dict: dict[str, SingleQuadruped]):
        """Reset via sampling from commands
        env_ids: (E) indices"""
        self._skill_indices[env_ids] = torch.multinomial(self._prob_tensor, len(env_ids), replacement=True) # (E)
        for i, skill in enumerate (self._skills):
            new_skill_envs = env_ids[self._skill_indices[env_ids] == i] # (E)
            if len(new_skill_envs) > 0:
                skill.set_new_internals(new_skill_envs, robot_dict)
            
    def get_raw_commands(self) -> dict[str, torch.Tensor]:
        raw_commands_dict: dict[str, torch.Tensor] = {}
        for name in self._robot_names:
            raw_commands_dict[name] = torch.zeros(size=(self._num_envs, 4), device=self._device)
        
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                raw_command_dict = skill.get_raw_command(env_ids) # dict[str, torch.Tensor]
                for name in self._robot_names:
                    raw_commands_dict[name][env_ids] = raw_command_dict[name] # (E,4)
        return raw_commands_dict
            
    def update(self, robot_dict: dict[str, SingleQuadruped]):
        """Update the commands, called in get_observations"""
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i # (N)
            if env_ids.any():
                skill.update(env_ids, robot_dict)
            
    def set_debug_vis_impl(self, debug_vis: bool):
        for skill in self._skills:
            skill.set_debug_vis_impl(debug_vis)
            
    def debug_vis_callback(self, robot_dict: dict[str, SingleQuadruped]):
        for i, skill in enumerate(self._skills):
            skill_env_ids = self._skill_indices == i # (N)
            if skill_env_ids.any():
                skill.debug_vis_callback(skill_env_ids, robot_dict)
                
    def compute_rewards(self, robot_dict: dict[str, SingleQuadruped]) -> dict[str, torch.Tensor]:
        """Returns a dictionary (N,) reward vector"""
        rewards_dict: dict[str, torch.Tensor] = {}
        for name in self._robot_names:
            rewards_dict[name] = torch.zeros(size=(self._num_envs,), device=self._device)
            
        for i, skill in enumerate(self._skills):
            env_ids = self._skill_indices == i
            if env_ids.any():
                rewards = skill.compute_rewards(env_ids, robot_dict)
                for name in self._robot_names:
                    rewards_dict[name][env_ids] = rewards[name]
                    
        # for name in self._robot_names:
        #     assert torch.all(rewards_dict[name] != 0), "All rewards should be non-zero"
        return rewards_dict
