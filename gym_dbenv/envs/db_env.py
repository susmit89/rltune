import gym
from gym import error, spaces, utils
from client import DBENGINE
import numpy as np


class DBENV(gym.Env):
    metadata = {
        "render.modes": ["human"],
    }
    def __init__(self):
        db = DBENGINE()
        t_columns = db.column_schema()
        n_columns = db.get_column_count()
        t_tables=tuple([i[1]+1 for i in n_columns])
        low = np.zeros(len(t_tables), dtype=int)
        high =  np.array(t_tables, dtype=int) - np.ones(len(t_tables), dtype=int)
        self.action_space = spaces.Box(low, high,dtype=int)
        self.observation_space = spaces.Box(-1, 1, shape=(2*len(t_columns),), dtype='int')

    def __del__(self):
        pass

    def _configure(self, display=None):
        pass

    def _seed(self, seed=None):
        pass

    def step(self, action):
        done = 0
        reward = 0
        self.state = 0
        info = {}
        return self.state, reward, done, info

    def reset(self):
        #clear the status
        pass

    def is_game_over(self):
        pass

    def _render(self, mode="human", close=False):
        pass
