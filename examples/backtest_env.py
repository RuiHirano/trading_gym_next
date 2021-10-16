from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv

def run_backtest_env():
    param = EnvParameter(df=GOOG[:200], mode="sequential", window_size=10)
    print(GOOG)
    env = TradingEnv(param)
    obs = env.reset()
    done = False
    while not done:
        action = random.choice([0,1,2])
        next_obs, reward, done, info = env.step(action, size=1)
        print("obs", obs.tail())
        print("action: ", action)
        print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
        print("next_obs", next_obs.tail())
        print("-"*10)
        obs = next_obs
    print("finished")
    stats = env.stats()
    print(stats)
    env.plot()

if __name__ == "__main__":
    run_backtest_env()