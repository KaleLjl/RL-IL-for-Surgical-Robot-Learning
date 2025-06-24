# Author(s): Lele Chen
# Created on: 2025-06-24
# Last modified: 2025-06-24

from gymnasium.envs.registration import register
from .needle_reach import NeedleReachEnv

register(
    id='dvrk_gym/NeedleReach-v0',
    entry_point='dvrk_gym.envs:NeedleReachEnv',
    max_episode_steps=100,
)
