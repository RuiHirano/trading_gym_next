from backtesting.test import GOOG
import random
import gym
from trading_gym_next import EnvParameter, TradingEnv

########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
#######################################
class CustomEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        env = TradingEnv(param)
        super().__init__(env)
        self.env = env
        self.side = None # None: No Position, LONG: Long Position, SHORT: Short Position

    def step(self, action):

        if self.side == "LONG" and action == 1:
            action = 0
        if self.side == "SHORT" and action == 2:
            action = 0

        obs, reward, done, info = self.env.step(action, size=1)

        if self.side == "LONG" and action == 2 or self.side == "SHORT" and action == 1:
            done = True

        if info["position"].size == 0:
            if self.side != 0:
                self.side = 0
        elif info["position"].size > 0:
            self.side = "LONG"
        elif info["position"].size < 0:
            self.side = "SHORT"

        return obs, reward, done, info

def run_custom_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", add_feature=True, window_size=10)
    print(GOOG)
    env = CustomEnv(param)
    for i in range(20):
        print("Episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            next_obs, reward, done, info = env.step(action)
            print("obs", obs.tail())
            print("action: ", action)
            print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
            print("next_obs", next_obs.tail())
            print("-"*10)
            obs = next_obs

    stats = env.stats()
    print(stats)

if __name__ == "__main__":
    run_custom_env()