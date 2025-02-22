# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# config shortcuts
algorithm = args_cli.algorithm.lower()

### RVMod
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.anymal_c.mod_anymal_c_env import ModAnymalCEnv
from isaaclab_tasks.direct.anymal_c.mod_anymal_c_env_cfg import CustomCommandCfg
from isaaclab_tasks.direct.anymal_c.mod_anymal_command_manager import CustomCommandManager
from abc import ABC, abstractmethod
"""
taskset -c 80-120 python source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/mod_anymal_c_env_play.py \
    --headless --video --video_length=600 --num_envs=32 \
    --checkpoint=logs/skrl/anymal_c_mod_flat_direct/2025-02-06_18-22-26_ppo_torch/checkpoints/best_agent.pt
"""

#######################################################################
############ Commands ###############################################
# region
class AbstractCommand(ABC):
    WALKING_HEIGHT = 0.6
    SITTING_HEIGHT = 0.2
    
    def __init__(self, device: torch.device):
        self._device = device
    
    @abstractmethod
    def get_command(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def completed(self, robot: Articulation) -> bool:
        pass
    
class WalkCommand(AbstractCommand):
    def __init__(self, device: torch.device, dir: tuple[float, float, float], timesteps: int):
        """
        dir: (x,y,yaw) direction to walk in
        """
        super().__init__(device)
        self.SPEED = 1.0
        self.dir = dir
        self.timesteps = timesteps
        self._current_timestep = 0
        
    def get_command(self) -> torch.Tensor:
        return torch.tensor([self.dir[0] * self.SPEED, self.dir[1] * self.SPEED, self.dir[2], 
                             AbstractCommand.WALKING_HEIGHT])
        
    def completed(self, robot: Articulation) -> bool:
        self._current_timestep += 1
        return self._current_timestep >= self.timesteps
    
class SitUnsitCommand(AbstractCommand):
    def __init__(self, device: torch.device, sit: bool):
        super().__init__(device)
        self.sit = sit
        if self.sit:
            self._target_height = AbstractCommand.SITTING_HEIGHT
        else:
            self._target_height = AbstractCommand.WALKING_HEIGHT
        self.ACCURACY = 0.05
        self._current_timestep = 0
        self.HOLDING_TIMESTEPS = 50

    def get_command(self) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, self._target_height])
    
    def completed(self, robot: Articulation) -> bool:
        assert robot.data.root_com_pos_w.shape[0] == 1 # only works for single robot env
        self._current_timestep += bool((torch.abs(robot.data.root_com_pos_w[:, 2] - self._target_height) < self.ACCURACY)[0])
        return self._current_timestep >= self.HOLDING_TIMESTEPS
        
    
class PlayCommandManager:
    def __init__(self, command_list: list[AbstractCommand]):
        self._command_list = command_list
        self._command_at = 0

    def update_commands(self, robot: Articulation) -> torch.Tensor:
        if self._command_list[self._command_at].completed(robot):
            self._command_at += 1
        self._command_at = min(self._command_at, len(self._command_list) - 1)
        return self._command_list[self._command_at].get_command()
    
    def create_obs(self, robot: Articulation) -> torch.Tensor:
        return torch.cat([
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            self.update_commands(robot),
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel,
            robot.data.root_com_pos_w,
            robot.data.root_com_vel_w,
        ], dim=-1)
        
        
# endregion
#######################################################################


def main():
    task = "Isaac-Velocity-Mod-Flat-Anymal-C-Direct-v0"
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")
    
    # Set custom command sequence
    custom_command_cfg = CustomCommandCfg(
        max_cmd_length = 4,
        cmd_list = [(("walk", "unsit", "sit", "walk"), 1)]
    )
    
    underlying_env: ModAnymalCEnv = env._unwrapped
    underlying_env.command_manager.parse_cfg_to_custom_command_sequence(custom_command_cfg)

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    # Create command manager
    # device = underlying_env.device
    # WALK_TIME = 500
    # command_manager = PlayCommandManager([
    #     WalkCommand(device, (1.0, 0.0, 0.0), WALK_TIME),
    #     SitUnsitCommand(device, True),
    #     SitUnsitCommand(device, False),
    #     WalkCommand(device, (-1.0, 0.0, 0.0), WALK_TIME),
    #     SitUnsitCommand(device, True),
    #     SitUnsitCommand(device, False),
    # ])
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        print(f"Running: {timestep}")

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # command = command_manager.get_current_command(underlying_env._robot)
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
