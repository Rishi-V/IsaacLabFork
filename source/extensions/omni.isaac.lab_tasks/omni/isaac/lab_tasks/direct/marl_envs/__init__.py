# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

#################################################################
################ Double Cartpole Environments ###################
gym.register(
    id="Isaac-Double-Cartpole-Direct-v0",
    entry_point=f"{__name__}.double_cartpole_env_v0:DoubleCartpoleEnvV0",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_cartpole_env_v0:DoubleCartpoleEnvCfgV0",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_double_cartpole_ppo_cfg_v0.yaml",
    },
)

gym.register(
    id="Isaac-Double-Cartpole-Direct-v1",
    entry_point=f"{__name__}.double_cartpole_env_v1:DoubleCartpoleEnvV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_cartpole_env_v1:DoubleCartpoleEnvCfgV1",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_double_cartpole_ppo_cfg_v1.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_double_cartpole_ippo_cfg_v1.yaml",
    },
)

#################################################################
################ Double Anymal Velocity Environments ###################
gym.register(
    id="Isaac-Double-Velocity-Flat-Anymal-C-Direct-v0",
    entry_point=f"{__name__}.double_anymal_c_env:DoubleAnymalCFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_anymal_c_env_cfg:DoubleAnymalCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_double_anymal_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_double_anymal_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_double_anymal_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_double_anymal_mappo_cfg.yaml",
    },
)
