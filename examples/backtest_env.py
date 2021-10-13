from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv

def run_backtest_env():
    param = EnvParameter(df=GOOG[:200], mode="backtest", window_size=10)
    print(GOOG)
    env = TradingEnv(param)
    
    obs = env.reset()
    done = False
    while not done:
        action = random.choice([0,1,2])
        obs, reward, done, info = env.step(action, size=1)
        print("step: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["timestamp"]-1, action, reward, done, info["timestamp"]-1, info["episode_step"]-1, info["position"]))
        print(obs.tail())
    print("finished")
    stats = env.stats()
    print(stats)
    env.plot()

if __name__ == "__main__":
    run_backtest_env()