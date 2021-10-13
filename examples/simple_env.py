from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv

def run_simple_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", window_size=10)
    print(GOOG)
    env = TradingEnv(param)
    
    obs = env.reset()
    for i in range(100):
        print("episode: ", i)
        obs = env.reset()
        for k in range(10):
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action, size=1)
            print("episode: {}, step: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(i, k, action, reward, done, info["timestamp"]-1, info["episode_step"]-1, info["position"]))
            print(obs.tail())
    print("finished")
    stats = env.stats()
    print(stats)

if __name__ == "__main__":
    run_simple_env()