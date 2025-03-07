from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster

# from .mod_anymal_reward_manager import CustomRewardManager

## Visualizations
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
# import isaaclab.utils.math as math_utils
# # from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

class SingleQuadruped:
    def __init__(self, cfg, agent_name: str, robot_cfg: ArticulationCfg, 
                 contact_sensor_cfg: ContactSensorCfg, num_envs: int):
        self._cfg = cfg
        self._agent_name = agent_name
        self._robot_cfg = robot_cfg
        self._contact_sensor_cfg = contact_sensor_cfg
        self._num_envs = num_envs
        
    # def setup_scene(self):
        self._robot = Articulation(self._robot_cfg)
        self._contact_sensor = ContactSensor(self._contact_sensor_cfg)
        
    def post_setup_scene(self, device: torch.device, step_dt: float):
        self._device = device
        self._step_dt = step_dt
        self._action_dim: int = self._cfg.action_spaces[self._agent_name]
        self._action_scale: float = self._cfg.action_scales[self._agent_name]
        
        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self._num_envs, self._action_dim, device=self._device) # (N,12)
        self._previous_actions = torch.zeros(self._num_envs, self._action_dim, device=self._device) # (N,12)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        
    def pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self._action_scale * self._actions + self._robot.data.default_joint_pos
        
    def apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        
    def get_observations(self, raw_commands: torch.Tensor) -> torch.Tensor:
        """Returns observations for the agent.

        Args:
            raw_commands (torch.Tensor): (N,4) command vector

        Returns:
            torch.Tensor: (N,49) as of now
        """
        self._previous_actions = self._actions.clone()
        obs = torch.cat([self._robot.data.root_lin_vel_b, # (N,3): Remove from actor (critic is okay)
                    self._robot.data.root_ang_vel_b, # (N,3)
                    self._robot.data.projected_gravity_b, # (N,3)
                    raw_commands, # (N,4)
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,12)
                    self._robot.data.joint_vel, # (N,12)
                    self._actions, # (N,12)
                    ], dim=-1)
        return obs
    
    def reset(self, env_ids: torch.Tensor, terrain_env_origins: torch.Tensor):
        """Resets the quadruped.

        Args:
            env_ids (torch.Tensor): environment ids
            terrain_env_origins (torch.Tensor): terrain origins to add to the robot's root state
        """
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        ### Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += terrain_env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids) # Ignore red squiggles
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids) # Ignore red squiggles
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids) # Ignore red squiggles

    def get_robot(self) -> Articulation:
        return self._robot
    
    def get_contact_sensor(self) -> ContactSensor:
        return self._contact_sensor
    
    def get_name(self) -> str:
        return self._agent_name
    
    def get_actions(self) -> torch.Tensor:
        return self._actions

    def get_previous_actions(self) -> torch.Tensor:
        return self._previous_actions

    def get_step_dt(self) -> float:
        return self._step_dt

    def get_feet_ids(self) -> list[int]:
        return self._feet_ids

    def get_undesired_contact_body_ids(self) -> list[int]:
        return self._undesired_contact_body_ids