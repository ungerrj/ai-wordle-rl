"""
Environment package for Wordle RL project.
"""

from .wordle_env import WordleEnv

# Register the environment with Gymnasium
import gymnasium as gym
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='WordleEnv-v0',
    entry_point='environment.wordle_env:WordleEnv',
    max_episode_steps=100,
)