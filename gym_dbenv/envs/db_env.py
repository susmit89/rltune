import gym
from gym import error, spaces, utils

import numpy as np


class DBENV(gym.Env):
    metadata = {
        "render.modes": ["human"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self):
        maze_size=(30, 30)
        low = np.zeros(len(maze_size), dtype=int)
        high =  np.array(maze_size, dtype=int) - np.ones(len(maze_size), dtype=int)
        self.action_space = spaces.Box(low, high)
        pass

    def __del__(self):
        pass

    def _configure(self, display=None):
        pass

    def _seed(self, seed=None):
        pass

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def is_game_over(self):
        pass

    def _render(self, mode="human", close=False):
        pass
