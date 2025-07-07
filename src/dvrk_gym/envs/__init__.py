# Author(s): Lele Chen
# Created on: 2025-06-24
# Last modified: 2025-06-24

from gymnasium.envs.registration import register
from .needle_reach import NeedleReachEnv
from .peg_transfer import PegTransferEnv

register(
    id='NeedleReach-v0',
    entry_point='dvrk_gym.envs.needle_reach:NeedleReachEnv',
    max_episode_steps=100,
)

register(
    id='PegTransfer-v0',
    entry_point='dvrk_gym.envs.peg_transfer:PegTransferEnv',
    max_episode_steps=150,
)
