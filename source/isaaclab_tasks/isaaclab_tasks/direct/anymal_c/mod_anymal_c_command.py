
from .mod_anymal_c_env import ModAnymalCEnv
import torch

def customCommands(env: ModAnymalCEnv, env_ids: torch.Tensor | None):
    if env_ids is None or len(env_ids) == env.num_envs:
            env_ids = env._robot._ALL_INDICES
    env._commands[env_ids] = torch.zeros_like(env._commands[env_ids]).uniform_(-1.0, 1.0)
    env._commands[env_ids,3] = 0.6
    