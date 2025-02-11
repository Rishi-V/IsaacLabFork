import torch

from isaaclab.assets import Articulation

## Visualizations
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
from .mod_anymal_c_env_cfg import CommandCfg
# from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

class CustomCommandManager:
    def __init__(self, num_envs: int, device: torch.device, cmd_cfg: CommandCfg, z_is_vel: bool):
        self._num_envs = num_envs
        self._device = device
        self.cmd_cfg = cmd_cfg
        self.z_is_vel = z_is_vel
        self.SITTING_HEIGHT = 0.2
        self.WALKING_HEIGHT = 0.60
        self.PROB_SIT = cmd_cfg.prob_sit #0.5
        self.MAX_Z_VEL = 0.1
        
        self._high_level_commands = torch.zeros(size=(self._num_envs,), device=self._device) # (N); -1=sit, 1=unsit, 0=walk
        self._raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device) # (N,4); (x,y,yaw,z) velocities or positions
        self._time_doing_action = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent doing action
        self._time_trying_command = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent trying to do action
        
    def update_commands(self, robot: Articulation):
        self._time_trying_command -= 1
        finished_trying_envs = self._time_trying_command < 0
        
        ### Sitting commands
        sitting_envs = self._high_level_commands == -1 # (N)
        sitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.SITTING_HEIGHT) < 0.1 # (N)
        successful_sits = torch.logical_and(sitting_envs, sitting_robots) # (N)
        self._time_doing_action[successful_sits] += 1
        
        ### Unsitting commands
        unsitting_envs = self._high_level_commands == 1 # (N)
        unsitting_robots = torch.abs(robot.data.root_com_pos_w[:,2] - self.WALKING_HEIGHT) < 0.1 # (N)
        successful_unsits = torch.logical_and(unsitting_envs, unsitting_robots) # (N)
        self._time_doing_action[successful_unsits] += 1
        
        ### Walking commands
        walking_envs = self._high_level_commands == 0 # (N)
        lin_vel_error = torch.sum(torch.square(self._raw_commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1) < 0.1 # (N)
        yaw_rate_error = torch.square(self._raw_commands[:, 2] - robot.data.root_ang_vel_b[:, 2]) < 0.1 # (N)
        successful_walks = torch.logical_and(walking_envs, torch.logical_and(lin_vel_error, yaw_rate_error)) # (N)
        self._time_doing_action[successful_walks] += 1
        
        ### Update high_level actions
        # finished_sitting_envs = torch.logical_and(sitting_envs, self._time_doing_action > 50) # (N), 1 second
        finished_sitting_envs = sitting_envs & ((self._time_doing_action > 50) | finished_trying_envs) # (N), 1 second
        # finished_unsitting_envs = torch.logical_and(unsitting_envs, self._time_doing_action > 50) # (N), 1 second
        finished_unsitting_envs = unsitting_envs & ((self._time_doing_action > 50) | finished_trying_envs)  # (N), 1 second
        # finished_walking_envs = torch.logical_and(walking_envs, self._time_doing_action > 400) # (N), 10 seconds
        finished_walking_envs = walking_envs & ((self._time_doing_action > 400) | finished_trying_envs) # (N), 10 seconds
        self._high_level_commands[finished_sitting_envs] = 1 # Make them unsit
        self._high_level_commands[finished_unsitting_envs] = 0 # Make them sit walk
        self._high_level_commands[finished_walking_envs] = -1 # Make them sit
        self._time_doing_action[finished_sitting_envs] = 0
        self._time_doing_action[finished_unsitting_envs] = 0
        self._time_doing_action[finished_walking_envs] = 0
        self.set_time_trying_command(finished_sitting_envs, boolmask=True)
        self.set_time_trying_command(finished_unsitting_envs, boolmask=True)
        self.set_time_trying_command(finished_walking_envs, boolmask=True)
        # self._time_trying_command[finished_sitting_envs].uniform_(300, 500) # Spend at most 500 steps trying to sit
        # self._time_trying_command[finished_unsitting_envs].uniform_(300, 500) # Spend at most 500 steps trying to unsit
        # self._time_trying_command[finished_walking_envs].uniform_(300, 500) # Spend at most 500 steps trying to walk
    
        # if torch.sum(finished_sitting_envs) > 0:
        #     print(f"Finished sitting: {torch.sum(finished_sitting_envs)}")
        
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
        sitting_inds = env_ids[torch.where(bool_list == 0)[0]] # (E)
        walking_inds = env_ids[torch.where(bool_list == 1)[0]] # (E)
        
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
        # self._time_trying_command[env_ids] = torch.zeros(size=(len(env_ids),), device=self._device).uniform_(300, 500) # Spend at most 500 steps per command
        self.set_time_trying_command(env_ids, boolmask=False)
        
    def set_time_trying_command(self, env_ids: torch.Tensor, boolmask: bool):
        """env_ids: (E)
        boolmask: bool indicating whether env_ids is a boolean mask or not"""
        if boolmask:
            self._time_trying_command[env_ids] = torch.zeros(size=(int(env_ids.sum().item()),), device=self._device).uniform_(300, 500)
        else:
            self._time_trying_command[env_ids] = torch.zeros(size=(len(env_ids),), device=self._device).uniform_(300, 500)
        
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
