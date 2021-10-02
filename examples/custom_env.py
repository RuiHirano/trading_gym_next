from backtesting.test import GOOG
import random
import gym
from trading_gym_next import EnvParameter, TradingEnv

class CustomEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        env = TradingEnv(param)
        super().__init__(env)
        self.env = env
        self.position_side = 0 # 0: No Position, 1: Long Position, 2: Short Position

    def step(self, action):

        if self.position_side == 1 and action == 1:
            action = 0
        if self.position_side == 2 and action == 2:
            action = 0

        obs, reward, done, info = self.env.step(action, size=1)

        if self.position_side == 1 and action == 2 or self.position_side == 2 and action == 1:
            done = True

        if info["position"].size == 0:
            if self.position_side != 0:
                self.position_side = 0
        elif info["position"].size > 0:
            self.position_side = 1
        elif info["position"].size < 0:
            self.position_side = 2

        return obs, reward, done, info