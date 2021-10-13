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

def run_custom_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", eval=False, window_size=10)
    print(GOOG)
    env = CustomEnv(param)
    for i in range(200):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action)
            print("episode: {}, step: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(i, step, action, reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print(obs.tail())
            step += 1

    stats = env.stats()
    print(stats)

if __name__ == "__main__":
    run_custom_env()