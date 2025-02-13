import torch

from isaaclab.assets import Articulation

## Visualizations
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
from .mod_anymal_c_env_cfg import CommandCfg, CustomCommandCfg
# from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand # Contains example of marker

class CustomCommandManager:
    def __init__(self, num_envs: int, device: torch.device, cmd_cfg: CommandCfg, z_is_vel: bool):
        self._num_envs = num_envs
        self._device = device
        self.cmd_cfg = cmd_cfg
        self.z_is_vel = z_is_vel
        self.SITTING_HEIGHT = 0.10
        self.WALKING_HEIGHT = 0.60
        self.PROB_SIT = cmd_cfg.prob_sit #0.5
        self.MAX_Z_VEL = 0.1
        
        self._high_level_commands = torch.zeros(size=(self._num_envs,), device=self._device) # (N); -1=sit, 1=unsit, 0=walk
        self._raw_commands = torch.zeros(size=(self._num_envs, 4), device=self._device) # (N,4); (x,y,yaw,z) velocities or positions
        self._time_doing_action = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent doing action
        self._time_trying_command = torch.zeros(size=(self._num_envs,), device=self._device) # (N); Time spent trying to do action
        self._hl_sequence = torch.zeros(size=(self._num_envs,10,6), device=self._device) # (N,T,6)
        self._hl_indices = torch.zeros(size=(self._num_envs,), device=self._device, dtype=torch.int) # (N)
        
    # def parse_cfg_to_custom_command_sequence(self, cmd_cfg: CustomCommandCfg):
    #     max_cmd_length = cmd_cfg.max_cmd_length
    #     cmd_list = cmd_cfg.cmd_list
        
    #     # hl_seq_tensor = torch.zeros(size=(self._num_envs, max_cmd_length, 8), device=self._device) # (N,T,8)
    #     # for cmd_seq in cmd_list:
    #     #     cmd_seq, prob = cmd_seq
    #     #     num_envs = int(prob * self._num_envs)
    #     #     for cmd in cmd_seq: # cmd is one of ["sit", "unsit", "walk"]
                
            
    #     #     cmd_seq = [self.parse_cmd_to_custom_command(cmd) for cmd in cmd_seq]
    #     #     cmd_seq = torch.tensor(cmd_seq, device=self._device)
        
    def reset_commands2(self, env_ids: torch.Tensor, robot: Articulation):
        """env_ids: (E)
        Note: env_ids should be a tensor of indices, not boolean mask"""
        self._hl_indices[env_ids] = 0
        self._time_doing_action[env_ids] = 0
        
        ### Resample hl_sequence based on cmd_cfg
        rand_sequence = (-1, 1, 0) # ["sit", "unsit", "walk"]
        self._hl_sequence[env_ids] = torch.zeros(size=(len(env_ids), 10, 8), device=self._device) # (E,T,8)
        self._hl_sequence[env_ids, :3, self.CC_IND_HL] = torch.tensor(rand_sequence, device=self._device).repeat(len(env_ids), 1)
        self._hl_sequence[env_ids, :3, self.CC_IND_SAMPLE_OR_DONE] = 0 # 0=sample raw commands
                
        self._time_trying_command[env_ids] = self._hl_sequence[env_ids, 0, self.CC_IND_TIMEOUT]
        
    def set_custom_command_sequence(self, high_level_command_sequence: torch.Tensor):
        """high_level_command_sequence: (N,T,8)
        - 8: (high_level_index, timeout, sample_or_done, x_vel, y_vel, yaw_vel, z_height)
        - high_level_index: -1=sit, 1=unsit, 0=walk
        - timeout: Maximum of timesteps to try to do action
        - timehold: Number of timesteps to hold action
        - sample_or_done: 0=sample raw commands, 1=don't sample, 2=done
        - x_vel, y_vel, yaw_vel, z_height: Only use if don't sample (sample_or_done=1)
        """
        self.CC_IND_HL = 0
        self.CC_IND_TIMEOUT = 1
        self.CC_IND_TIMEHOLD = 2
        self.CC_IND_SAMPLE_OR_DONE = 3
        self.CC_IND_RAW_ACTIONS = 4
        
        self._hl_sequence = high_level_command_sequence # (N,T,6)
        self._hl_indices = torch.zeros(size=(self._num_envs,), device=self._device) # (N)
        
    def update_time_doing_action2(self, robot: Articulation):
        self._time_trying_command -= 1
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
        
    def get_finished_envs2(self) -> torch.Tensor:
        """Returns a boolean mask of environments that have finished their current high-level command"""
        finished_envs = self._time_trying_command < 0 # (N)
        finished_envs |= self._time_doing_action > self._hl_sequence[:, self._hl_indices, self.CC_IND_TIMEHOLD] # (N)
        return finished_envs        

    def update_hl_commands2(self, finished_env: torch.Tensor, robot: Articulation):
        """finished_env: (N), boolmask"""
        finished_env_inds = torch.where(finished_env)[0] # (E)
        self._hl_indices[finished_env_inds] += 1
        next_hl_inds = self._hl_indices[finished_env_inds] # (E)
        self._time_trying_command[finished_env_inds] = self._hl_sequence[finished_env_inds, next_hl_inds, self.CC_IND_TIMEOUT] # (E)
        
        next_hl_command = self._hl_sequence[finished_env_inds, next_hl_inds, self.CC_IND_HL] # (E)
        sample_or_done = self._hl_sequence[finished_env_inds, next_hl_inds, self.CC_IND_SAMPLE_OR_DONE] # (E)
        raw_commands = self._hl_sequence[finished_env_inds, next_hl_inds, self.CC_IND_RAW_ACTIONS:] # (E,4)
        
        ### Restart sequence for done envs
        done_envs = finished_env_inds[sample_or_done == 2] # (E_done)
        self.reset_commands(done_envs, robot=robot)
        
        ### Set fixed raw commands
        fixed_envs = finished_env_inds[sample_or_done == 1] # (E_fixed)
        self._raw_commands[fixed_envs] = raw_commands[fixed_envs] # (E_fixed,4)
        
        ### Set randomly sampled raw commands
        sample_envs = sample_or_done == 0 # (E)
        random_sit_envs = sample_envs & (next_hl_command == -1) # (E)
        random_unsit_envs = sample_envs & (next_hl_command == 1) # (E)
        random_walk_envs = sample_envs & (next_hl_command == 0) # (E)
        self.set_random_sit_commands(finished_env_inds[random_sit_envs], boolmask=False)
        self.set_random_unsit_commands(finished_env_inds[random_unsit_envs], boolmask=False)
        self.set_random_walk_commands(finished_env_inds[random_walk_envs], boolmask=False)
        
    def update_raw_commands_based_on_robot2(self, robot: Articulation):
        """Update raw commands based on the robot's current state.        
        """
        # Sitting raw commands
        if self.z_is_vel:
            sitting_envs = self._high_level_commands == -1 # (N)
            self._raw_commands[sitting_envs,:3] = 0.0
            error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_envs,2] # negative if robot is above sitting height
            self._raw_commands[sitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
            
        # Unsitting raw commands
        if self.z_is_vel:
            unsitting_envs = self._high_level_commands == 1 # (N)
            self._raw_commands[unsitting_envs,:3] = 0.0
            error = self.WALKING_HEIGHT - robot.data.root_com_pos_w[unsitting_envs,2] # positive if robot is below walking height
            self._raw_commands[unsitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        
    def update_commands2(self, robot: Articulation):
        self.update_time_doing_action2(robot) # Updates time_doing_action based on the robot and high_level_commands
        finished_envs = self.get_finished_envs2() # Gets envs that finished holding or timed out
        self.update_hl_commands2(finished_envs, robot) # Updates high_level_commands, raw_commands, and time_trying_command based on finished_envs
        self.update_raw_commands_based_on_robot2(robot) # Updates raw_commands based on the robot and high_level_commands
        
        
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
        finished_sitting_envs = sitting_envs & ((self._time_doing_action > 50) | finished_trying_envs) # (N), 1 second
        finished_unsitting_envs = unsitting_envs & ((self._time_doing_action > 50) | finished_trying_envs)  # (N), 1 second
        finished_walking_envs = walking_envs & ((self._time_doing_action > 400) | finished_trying_envs) # (N), 10 seconds
        new_sitting_envs = finished_walking_envs # finished_walking_envs # Make them unsit
        new_unsitting_envs = finished_sitting_envs # finished_sitting_envs # Make them sit walk
        new_walking_envs = finished_unsitting_envs # self._high_level_commands == 2 # finished_unsitting_envs # Make them sit, change command later
        self._high_level_commands[new_unsitting_envs] = 1 
        self._high_level_commands[new_walking_envs] = 0 
        self._high_level_commands[new_sitting_envs] = -1 
        self._time_doing_action[new_unsitting_envs] = 0
        self._time_doing_action[new_walking_envs] = 0
        self._time_doing_action[new_sitting_envs] = 0
        self.set_time_trying_command(new_unsitting_envs, boolmask=True)
        self.set_time_trying_command(new_walking_envs, boolmask=True)
        self.set_time_trying_command(new_sitting_envs, boolmask=True)
        
        ### Update raw commands
        # Sitting raw commands
        if self.z_is_vel:
            sitting_envs = self._high_level_commands == -1 # (N)
            self._raw_commands[sitting_envs,:3] = 0.0
            error = self.SITTING_HEIGHT - robot.data.root_com_pos_w[sitting_envs,2] # negative if robot is above sitting height
            self._raw_commands[sitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        else: # Only update new sitting commands
            self.set_random_sit_commands(new_sitting_envs, boolmask=True)
            # self._raw_commands[sitting_envs,3] = self.SITTING_HEIGHT # (x_vel,y_vel,yaw_vel,z_height)
        # Unsitting raw commands
        if self.z_is_vel:
            unsitting_envs = self._high_level_commands == 1 # (N)
            self._raw_commands[unsitting_envs,:3] = 0.0
            error = self.WALKING_HEIGHT - robot.data.root_com_pos_w[unsitting_envs,2] # positive if robot is below walking height
            self._raw_commands[unsitting_envs,3] = torch.clamp(error, -self.MAX_Z_VEL, self.MAX_Z_VEL)
        else:
            # self._raw_commands[unsitting_envs,3] = self.WALKING_HEIGHT
            self.set_random_unsit_commands(new_unsitting_envs, boolmask=True)
        # Walking raw commands
        # new_walking_envs = torch.logical_and(walking_envs, self._time_doing_action > 100) # (N)
        self.set_random_walk_commands(new_walking_envs, boolmask=True)
        
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
        self.set_random_walk_commands(walking_inds, boolmask=False)
        
        ## Set sitting commands
        self._high_level_commands[sitting_inds] = -1
        self.set_random_sit_commands(sitting_inds, boolmask=False)
        
        ## Reset time doing action
        self._time_doing_action[env_ids] = 0
        self.set_time_trying_command(env_ids, boolmask=False)
        
    def set_time_trying_command(self, env_ids: torch.Tensor, boolmask: bool):
        """env_ids: (E)
        boolmask: bool indicating whether env_ids is a boolean mask or not"""
        if boolmask:
            self._time_trying_command[env_ids] = torch.zeros(size=(int(env_ids.sum().item()),), device=self._device).uniform_(400, 800)
        else:
            self._time_trying_command[env_ids] = torch.zeros(size=(len(env_ids),), device=self._device).uniform_(400, 800)
        
    def set_random_walk_commands(self, env_ids: torch.Tensor, boolmask: bool):
        """env_ids: (E)
        boolmask: bool indicating whether env_ids is a boolean mask or not"""
        if boolmask:
            env_ids = torch.where(env_ids)[0]
        
        self._raw_commands[env_ids] = torch.zeros(size=(len(env_ids), 4), device=self._device).uniform_(-1.0, 1.0)
        if self.z_is_vel:
            self._raw_commands[env_ids,3] = 0.0 # z-axis command is 0.0 velocity
        else:
            self._raw_commands[env_ids,3] = self.WALKING_HEIGHT # z-axis command is walking height
        
    def set_random_sit_commands(self, env_ids: torch.Tensor, boolmask: bool):
        """env_ids: (E)
        boolmask: bool indicating whether env_ids is a boolean mask or not"""
        if boolmask:
            env_ids = torch.where(env_ids)[0]
        
        self._raw_commands[env_ids, :3] = 0.0
        if self.z_is_vel:
            self._raw_commands[env_ids, 3] = -self.MAX_Z_VEL
        else:
            self._raw_commands[env_ids, 3] = torch.zeros(size=(len(env_ids),), 
                                device=self._device).uniform_(self.SITTING_HEIGHT, self.SITTING_HEIGHT+0.2)
            
    def set_random_unsit_commands(self, env_ids: torch.Tensor, boolmask: bool):
        """env_ids: (E)
        boolmask: bool indicating whether env_ids is a boolean mask or not"""
        if boolmask:
            env_ids = torch.where(env_ids)[0]

        self._raw_commands[env_ids, :3] = 0.0
        if self.z_is_vel:
            self._raw_commands[env_ids, 3] = self.MAX_Z_VEL
        else:
            self._raw_commands[env_ids, 3] = self.WALKING_HEIGHT
            # torch.zeros(size=(len(env_ids), 4), device=self._device).uniform_(self.SITTING_HEIGHT, self.WALKING_HEIGHT)
        
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
