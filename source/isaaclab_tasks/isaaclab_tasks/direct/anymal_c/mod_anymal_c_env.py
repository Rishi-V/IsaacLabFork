# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.envs.common import AgentID
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster


from .mod_anymal_c_env_cfg import ModAnymalCFlatEnvCfg #, WalkingRewardCfg, SitUnsitRewardCfg
# from .mod_anymal_command_manager import DynamicSkillManager
# from .mod_anymal_reward_manager import CustomRewardManager
from .skill_manager_double import DoubleAgentDynamicSkillManager
from .single_quadruped import SingleQuadruped

## Visualizations
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
# import isaaclab.utils.math as math_utils
# # from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

"""taskset -c 40-79 python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Velocity-Mod-Flat-Anymal-C-Direct-v0 \
--headless --video --video_length=600 --video_interval=10000 --num_envs=1024"""


class ModAnymalCEnv(DirectMARLEnv):
    cfg: ModAnymalCFlatEnvCfg

    def __init__(self, cfg: ModAnymalCFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Skill manager
        self._robot_names = ["robot1", "robot2"]
        self.skill_manager = DoubleAgentDynamicSkillManager(self._robot_names, self.num_envs, self.device)
        self.skill_manager.parse_cfg(cfg.dynamic_skill_cfg)

        self._robot1.post_setup_scene(self.device, self.step_dt)
        self._robot2.post_setup_scene(self.device, self.step_dt)
        self.set_debug_vis(debug_vis=cfg.debug_vis)

    def _setup_scene(self):
        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Robots
        env_origins: torch.Tensor = self._terrain.env_origins # (N,3) locations of the environments
        self._robot1 = SingleQuadruped(self.cfg, "robot1", self.cfg.robot_cfg1, self.cfg.contact_sensor1, self.num_envs, env_origins)
        self._robot2 = SingleQuadruped(self.cfg, "robot2", self.cfg.robot_cfg2, self.cfg.contact_sensor2, self.num_envs, env_origins)
        self._all_robots = {"robot1": self._robot1, "robot2": self._robot2}
        
        self.scene.articulations["robot1"] = self._robot1.get_robot()
        self.scene.articulations["robot2"] = self._robot2.get_robot()
        self.scene.sensors["contact_sensor1"] = self._robot1.get_contact_sensor()
        self.scene.sensors["contact_sensor2"] = self._robot2.get_contact_sensor()
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        self._robot1.pre_physics_step(actions[self._robot1.get_name()])
        self._robot2.pre_physics_step(actions[self._robot2.get_name()])

    def _apply_action(self):
        self._robot1.apply_action()
        self._robot2.apply_action()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.skill_manager.update(self._all_robots)
        raw_commands = self.skill_manager.get_raw_commands() # Dictionary str: (N,4)
        observations = dict()
        observations["robot1"] = self._robot1.get_observations(raw_commands["robot1"])
        observations["robot2"] = self._robot2.get_observations(raw_commands["robot2"])
        
        # # self.command_manager.update_commands(self._robot) # Update actions before getting observations
        # self._previous_actions = self._actions.clone()
        # # height_data = (
        # #     self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        # # ).clip(-1.0, 1.0)
        # obs = torch.cat([self._robot.data.root_lin_vel_b, # (N,3): Remove from actor (critic is okay)
        #             self._robot.data.root_ang_vel_b, # (N,3)
        #             self._robot.data.projected_gravity_b, # (N,3)
        #             # self.command_manager.get_commands(), # (N,4)
        #             raw_commands, # (N,4)
        #             self._robot.data.joint_pos - self._robot.data.default_joint_pos, # (N,12)
        #             self._robot.data.joint_vel, # (N,12)
        #             # height_data,
        #             self._actions, # (N,12)
        #             # self.get_static_anymal_obs(), # (N,37)
        #             ], dim=-1)
        # observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        reward_dict = self.skill_manager.compute_rewards(self._all_robots)
        return reward_dict

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.skill_manager.get_should_reset(self._all_robots)
        terminated_dict = {agent: terminated for agent in self.cfg.possible_agents}
        timed_out_dict = {agent: timed_out for agent in self.cfg.possible_agents}
        return terminated_dict, timed_out_dict # Ignore red squiggles

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot1.get_robot()._ALL_INDICES
        self._robot1.reset(env_ids)
        self._robot2.reset(env_ids)
        super()._reset_idx(env_ids) # Ignore red squiggles
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        ### Sample new commands
        self.skill_manager.reset(env_ids, self._all_robots)
        
        # Logging
        extras = dict()
        # for key in self.reward_manager.episode_sums.keys():
        #     episodic_sum_avg = torch.mean(self.reward_manager.episode_sums[key][env_ids])
        #     extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
        #     self.reward_manager.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        # extras = dict()
        # extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # self.extras["log"].update(extras)


    def _set_debug_vis_impl(self, debug_vis: bool):
        self.skill_manager.set_debug_vis_impl(debug_vis)
        
    def _debug_vis_callback(self, event):
        self.skill_manager.debug_vis_callback(self._all_robots)

        