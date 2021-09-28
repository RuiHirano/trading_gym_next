from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv

if __name__ == "__main__":
    param = EnvParameter(df=GOOG[:40], mode="sequential", window_size=10)
    print(GOOG[:40])
    env = TradingEnv(param)
    
    for i in range(2):
        print("episode: ", i)
        obs = env.reset()
        for k in range(5):
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action)
            print("episode: {}, step: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}".format(i, k, action, reward, done, info["timestamp"], info["episode_step"]))
            print(obs.tail())
    env.stats()
    env.plot()