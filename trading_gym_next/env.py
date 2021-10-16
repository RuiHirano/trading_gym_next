from backtesting import Backtest, Strategy
from abc import *
import gym
from gym import spaces
from threading import (Event, Thread)
import random
from typing import Union, Callable
from gym.core import Env
import pandas as pd
import sys
import threading
import time
import numpy as np
import copy

class __FULL_EQUITY(float):
    def __repr__(self): return '.9999'
_FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

class TradingStrategy(Strategy):
    window_size: int
    kill_event: Event
    callback: Callable
    def init(self):
        pass

    def next(self):
        self.callback(self)

class BacktestingThread(Thread):
    def __init__(self, 
            data, 
            window_size,
            cash: float = 10_000,
            commission: float = .0,
            margin: float = 1.,
            trade_on_close=False,
            hedging=False,
            exclusive_orders=False
        ):
        threading.Thread.__init__(self)
        self.step_event = Event()
        self.callback_event = Event()
        self.kill_event = Event()
        self.result_event = Event()
        TradingStrategy.callback = self._callback
        TradingStrategy.window_size = window_size
        TradingStrategy.kill_event = self.kill_event
        self.bt = Backtest(
            data, 
            TradingStrategy,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
        )
        self.strategy = None
        self.result = None
        self.window_size = window_size
        
    def run(self):
        time.sleep(0.01) # for sync first data
        self.result = self.bt.run()
        self.result_event.set()

    def get_strategy(self):
        if not self.kill_event.is_set():
            self.step_event.set()
            self.callback_event.wait()
            self.callback_event.clear()
        return self.strategy

    def kill(self):
        self.step_event.set()
        self.callback_event.set()
        self.kill_event.set()

    def _callback(self, strategy: Strategy):
        self.strategy = strategy
        
        if not self.kill_event.is_set() and len(self.strategy.data) >= self.window_size:
            self.callback_event.set()
            self.step_event.wait()
            self.step_event.clear()

    def stats(self):
        self.kill()
        if not self.result_event.is_set():
            self.result_event.wait()
        return self.result

    def plot(self,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool,str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True
    ):
        self.bt.plot(
            results=results,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )

class EnvParameter:
    def __init__(self, 
        df: pd.DataFrame, 
        window_size: int,
        mode: str = "sequential", 
        add_feature: bool = False,
        step_length: int = 1,
        cash: float = 10_000,
        commission: float = .0,
        margin: float = 1.,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False
    ):
        self.df = df
        self.add_feature = add_feature
        self.window_size = window_size
        self.step_length = step_length
        self.mode = mode  # "sequential": sequential episode, "random": episode start is random, "backtest": for eval
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.hedging = hedging
        self.exclusive_orders = exclusive_orders
        self._check_param()

    def _check_param(self):
        self._check_column()
        self._check_length()
        self._check_mode()

    def _check_column(self):
        # column check
        if not all([item in self.df.columns for item in ['Open', 'High', 'Low', 'Close']]):
            raise RuntimeError(("Required column is not exist."))

    def _check_length(self):
        if self.window_size > len(self.df):
            raise RuntimeError("df length is not enough.")

    def _check_mode(self):
        if self.mode not in ["sequential", "random"]:
            raise RuntimeError(("Parameter mode is invalid. You should be 'random' or 'sequential'"))

class TradingEnv(gym.Wrapper):
    def __init__(self, param: EnvParameter):
        self.param = param
        self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL
        self.factory = EpisodeFactory(self.param)
        self.episode = None

    def step(self, action, size: float = _FULL_EQUITY,
            limit: float = None,
            stop: float = None,
            sl: float = None,
            tp: float = None):
        if self.episode == None:
            print("NotEpisodeInitError: Please run env.reset() at first.")
            sys.exit(1)

        if action == 1:
            self.episode.strategy.buy(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        elif action == 2:
            self.episode.strategy.sell(
                size=size,
                limit=limit,
                stop=stop,
                sl=sl,
                tp=tp,
            )
        
        self.episode.forward()
        obs, reward, done, info = self.episode.status()

        return obs, reward, done, info

    def reset(self):
        self.episode = self.factory.create()
        obs, _, _, _ = self.episode.status()
        return obs

    def stats(self):
        result = self.episode.bt.stats()
        return result

    def plot(self,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool,str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True
    ):
        self.episode.bt.plot(
            results=results,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )

class Episode:
    def __init__(self, episode_data, param: EnvParameter):
        if param.add_feature:
            self.features = pd.DataFrame({"Reward": np.zeros(len(episode_data)), "Position": np.zeros(len(episode_data))}, index=episode_data.index)
        self.episode_data = episode_data
        self.episode_step = 0
        self.timestamp = len(param.df) - len(episode_data)  # index of all data
        self.param = param
        self.finished = False
        self.bt = BacktestingThread(
            self.episode_data, 
            self.param.window_size,
            cash=self.param.cash,
            commission=self.param.commission,
            margin=self.param.margin,
            trade_on_close=self.param.trade_on_close,
            hedging=self.param.hedging,
            exclusive_orders=self.param.exclusive_orders,
        )
        self.bt.daemon = True
        self.bt.start()
        self.strategy = self.bt.get_strategy()

    def forward(self):
        if not self.finished:
            self.episode_step += 1
            self.timestamp += 1
            self.strategy = self.bt.get_strategy()
            self.finished = True if len(self.strategy.data.df) >= len(self.episode_data) else False
        

    def status(self):
        # calc info
        obs = self.observation()
        info = self.info()
        reward = self.reward()
        done = self.done()

        # add feature to obs
        if self.param.add_feature:
            self.features.at[self.strategy.data.df.index[-1], "Reward"] = reward
            self.features.at[self.strategy.data.df.index[-1], "Position"] = info["position"].size
            obs = pd.merge(obs, self.features[:len(self.strategy.data.df)], left_index=True, right_index=True)

        return obs, reward, done, info


    def observation(self):
        return self.strategy.data.df[-self.param.window_size:]

    def reward(self):
        # sum of profit percentage
        return sum([trade.pl_pct for trade in self.strategy.trades])

    def done(self):
        return self.finished

    def info(self):
        return {
            "date": self.strategy.data.df.index[-1],
            "episode_step": self.episode_step,
            "timestamp": self.timestamp,
            "orders": self.strategy.orders, 
            "trades": self.strategy.trades, 
            "position": self.strategy.position, 
            "closed_trades": self.strategy.closed_trades, 
        }

class EpisodeFactory:
    def __init__(self, param: EnvParameter):
        self.param = param
        self.episode = None

    def create(self):
        next_timestamp = self.get_next_episode_timestamp()
        data = self.get_next_episode_data(next_timestamp)
        self.episode = Episode(data, self.param)
        return self.episode

    def get_next_episode_data(self, timestamp):
        return self.param.df[timestamp:]

    def get_next_episode_timestamp(self):
        if self.param.mode == "random":
            return random.choice(range(len(self.param.df)-self.param.window_size))
        else: # "sequential" or "backtest"
            if self.episode == None or self.episode.timestamp + self.param.window_size > len(self.episode.episode_data):
                return 0
            else:
                return self.episode.timestamp + 1
