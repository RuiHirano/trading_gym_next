import unittest
import sys
from numpy.core.fromnumeric import size
sys.path.append("..")
from trading_gym_next.env import TradingEnv, EnvParameter, BacktestingThread, EpisodeFactory, Episode
from backtesting.test import GOOG, EURUSD
import time

class TestBacktestingThread(unittest.TestCase):
    def setUp(self):
        self.data_len = 30
        pass

    def create_thread(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        bt = BacktestingThread(
            EURUSD[:self.data_len], 
            param.window_size,
            cash=param.cash,
            commission=param.commission,
            margin=param.margin,
            trade_on_close=param.trade_on_close,
            hedging=param.hedging,
            exclusive_orders=param.exclusive_orders,
        )
        bt.daemon = True
        bt.start()
        return bt


    def test_get_timestamp_in_order(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        bt = self.create_thread()
        for i in range(self.data_len-param.window_size):
            strategy = bt.get_strategy()
            self.assertEqual(strategy.data.df.index[-1], EURUSD[:self.data_len].index[param.window_size+i])

    def test_get_data_in_order(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        bt = self.create_thread()
        for i in range(self.data_len-param.window_size):
            strategy = bt.get_strategy()
            self.assertEqual(strategy.data.df.tail(1)["Close"].values[0], EURUSD[:param.window_size+i+1].tail(1)["Close"].values[0])

    def test_buy(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        bt = self.create_thread()
        for i in range(self.data_len-param.window_size):
            strategy = bt.get_strategy()
            strategy.buy(size=1)
            self.assertEqual(strategy.position.size, i)
            self.assertEqual(len(strategy.trades), i)
            self.assertEqual(len(strategy.orders), 1)

    def test_sell(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        bt = self.create_thread()
        for i in range(self.data_len-param.window_size):
            strategy = bt.get_strategy()
            strategy.sell(size=1)
            self.assertEqual(strategy.position.size, -i)
            self.assertEqual(len(strategy.trades), i)
            self.assertEqual(len(strategy.orders), 1)

class TestEpisode(unittest.TestCase):
    def setUp(self):
        self.data_len = 30
        self.data = EURUSD[self.data_len:]

    def test_get_episode_step_in_order(self):
        param = EnvParameter(df=EURUSD, mode="sequential", window_size=10)
        episode = Episode(self.data, param)
        for i in range(self.data_len-param.window_size):
            obs, reward, done, info = episode.status()
            episode.forward()
            self.assertEqual(info["episode_step"], i)

    def test_get_timestamp(self):
        param = EnvParameter(df=EURUSD, mode="sequential", window_size=10)
        episode = Episode(self.data, param)
        for i in range(self.data_len-param.window_size):
            obs, reward, done, info = episode.status()
            episode.forward()
            data_index = info["timestamp"]+param.window_size
            self.assertEqual(EURUSD.iloc[data_index].name, obs.tail(1).index.values[0])

    def test_done_when_all_data_finished(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        data = EURUSD[170:200]
        episode = Episode(data, param)
        loop = 30 - param.window_size
        for i in range(loop):
            obs, reward, done, info = episode.status()
            episode.forward()
            if i == loop-1:
                self.assertEqual(done, True)
            else:
                self.assertEqual(done, False)

    def test_reward_delay_one_step(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        data = EURUSD[170:200]
        episode = Episode(data, param)
        loop = 30 - param.window_size
        for i in range(loop):
            episode.strategy.buy()
            obs, reward, done, info = episode.status()
            episode.forward()
            if i == 0:
                self.assertEqual(reward, 0)
            else:
                self.assertNotEqual(reward, 0)

    def test_obs_size(self):
        param = EnvParameter(df=EURUSD[:200], mode="sequential", window_size=10)
        data = EURUSD[170:200]
        episode = Episode(data, param)
        loop = 30 - param.window_size
        for i in range(loop):
            episode.strategy.buy()
            obs, reward, done, info = episode.status()
            episode.forward()
            self.assertEqual(len(obs), param.window_size)

class TestEpisodeFactory(unittest.TestCase):
    def setUp(self):
        self.data_len = 30
        self.data = EURUSD[self.data_len:]

    def test_get_next_episode_sequential_timestamp(self):
        param = EnvParameter(df=EURUSD, mode="sequential", window_size=10)
        factory = EpisodeFactory(param)
        loop = 10
        episode_step = 2
        for i in range(loop):
            episode = factory.create()
            self.assertEqual(episode.timestamp%(episode_step+1), 0) # timestamp: 3,6,9....
            for k in range(episode_step):
                episode.forward()

    def test_get_next_episode_random_timestamp(self):
        param = EnvParameter(df=EURUSD, mode="random", window_size=10)
        factory = EpisodeFactory(param)
        loop = 10
        episode_step = 2
        for i in range(loop):
            episode = factory.create()
            for k in range(episode_step):
                episode.forward()
        # cannot test because timestamp is random value
