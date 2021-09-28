from backtesting.test import GOOG
import random
from trading_gym_next import EnvParameter, TradingEnv

class CustomEnv(TradingEnv):

    def _reward(self):
        # sum of profit
        return sum([trade.pl for trade in self.strategy.trades])

    def _done(self):
        return True if self.episode_step >= 5 else False

if __name__ == "__main__":
    param = EnvParameter(
        df=GOOG[:40], 
        mode="sequential", 
        window_size=10,
        cash=10000,
        commission=0.01,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    )
    env = CustomEnv(param)
    
    for i in range(2):
        print("episode: ", i)
        obs = env.reset()
        done = False
        while not done:
            action = random.choice([0,1,2])
            obs, reward, done, info = env.step(action, size=0.2)
            print("episode: {}, action: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}".format(i, action, reward, done, info["timestamp"], info["episode_step"]))
            print(obs.tail())
    stats = env.stats()
    print(stats)
    env.plot()