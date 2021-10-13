import unittest
import sys
sys.path.append("..")
from trading_gym_next.env import TradingEnv, EnvParameter, TimeManager
from backtesting.test import GOOG

class TestEnvReset(unittest.TestCase):
    def setUp(self):
        param = EnvParameter(df=GOOG[:40], mode="sequential", window_size=10)
        self.env = TradingEnv(param)


class TestTimeManager(unittest.TestCase):
    def setUp(self):
        param = EnvParameter(df=GOOG[:40], mode="sequential", window_size=10)
        self.tm = TimeManager(param)

    def test_init_values(self):
        self.assertEqual(self.tm.timestamp, 0)
        self.assertEqual(self.tm.epoch, 0)
        self.assertEqual(self.tm.step_count, 0)
        self.assertEqual(self.tm.episode_step, 0)

    def test_forward(self):
        self.tm.forward()
        self.assertEqual(self.tm.timestamp, 1)
        self.assertEqual(self.tm.step_count, 1)
        self.assertEqual(self.tm.episode_step, 1)

    def test_get_next_episode_timestamp_sequential(self):
        self.tm.forward()
        self.assertEqual(self.tm.timestamp, 1)
        self.assertEqual(self.tm.step_count, 1)
        self.assertEqual(self.tm.episode_step, 1)

