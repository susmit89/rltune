import gym


class DBENV(gym.Env):
    metadata = {
        "render.modes": ["human"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self):

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
