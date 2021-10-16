# Trading Gym Next
Trading environment for deep reinforcement learning in finance using backtesting.py

<img width="1352" alt="スクリーンショット 2021-09-28 13 25 52" src="https://user-images.githubusercontent.com/43264434/135023130-72e6ef12-8e0e-4698-8057-6c7ddcd25a67.png">

#### Reference
[backtesting.py document](https://kernc.github.io/backtesting.py/)  
[backtesting.py github](https://github.com/kernc/backtesting.py)

## Install

```
pip install trading_gym_next
```

## Example

```
cd examples
python simple_env.py
python custom_env.py
python backtest.py
```

## Quick Start

```
param = EnvParameter(df=GOOG[:200], mode="sequential", window_size=10)
print(GOOG)
env = TradingEnv(param)

for i in range(20):
    print("episode: ", i)
    obs = env.reset()
    for k in range(10):
        action = random.choice([0,1,2])
        next_obs, reward, done, info = env.step(action)
        print("obs", obs.tail())
        print("action: ", action)
        print("date: {}, reward: {}, done: {}, timestamp: {}, episode_step: {}, position: {}".format(info["date"], reward, done, info["timestamp"], info["episode_step"], info["position"]))
        print("next_obs", next_obs.tail())
        print("-"*10)
        obs = next_obs
print("finished")
stats = env.stats()
print(stats)
```

Output: 

```

episode: 0, step: 0, action: 1, reward: 0, done: False, timestamp: 0, episode_step: 0
              Open    High     Low   Close   Volume
2004-08-30  105.28  105.49  102.01  102.01  2601000
2004-08-31  102.30  103.71  102.16  102.37  2461400
2004-09-01  102.70  102.97   99.67  100.25  4573700
2004-09-02   99.19  102.37   98.94  101.51  7566900
2004-09-03  100.95  101.74   99.32  100.01  2578800
episode: 0, step: 1, action: 0, reward: 0.005643005643005683, done: False, timestamp: 1, episode_step: 1
              Open    High     Low   Close   Volume
2004-08-31  102.30  103.71  102.16  102.37  2461400
2004-09-01  102.70  102.97   99.67  100.25  4573700
2004-09-02   99.19  102.37   98.94  101.51  7566900
2004-09-03  100.95  101.74   99.32  100.01  2578800
2004-09-07  101.01  102.00   99.61  101.58  2926700

...

Start                     2004-08-26 00:00:00
End                       2004-10-14 00:00:00
Duration                     49 days 00:00:00
Exposure Time [%]                   65.714286
Equity Final [$]                     13121.08
Equity Peak [$]                      13121.08
Return [%]                            31.2108
Buy & Hold Return [%]               31.591141
Return (Ann.) [%]                  606.937469
Volatility (Ann.) [%]              208.035025
Sharpe Ratio                         2.917477
Sortino Ratio                       70.468812
Calmar Ratio                       234.874457
Max. Drawdown [%]                   -2.584093
Avg. Drawdown [%]                   -1.606885
Max. Drawdown Duration        6 days 00:00:00
Avg. Drawdown Duration        4 days 00:00:00
# Trades                                    1
Win Rate [%]                            100.0
Best Trade [%]                      31.233132
Worst Trade [%]                     31.233132
Avg. Trade [%]                      31.233132
Max. Trade Duration          30 days 00:00:00
Avg. Trade Duration          30 days 00:00:00
Profit Factor                             NaN
Expectancy [%]                      31.233132
SQN                                       NaN
_strategy                     TradingStrategy
_equity_curve                           Eq...
_trades                      Size  EntryBa...
dtype: object
```

## Custom Environment
```
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
```

## Properties

### TradingEnv Methods
| Name              | Description           |
| ---               | ---                   |
| env.step(action, (*option))              | Step environment timestamp. action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL|
| env.reset()             | Reset environment status. |
| env.stats()             | Compute and get statistics data |
| env.plot((*option))              | Plot trade results |

env.render method is not implemented.

### Env Parameters
| Name              | Default       | Description           |
| ---               | ---           | ---                   |
| df                | Required*     | pandas.DataFrame with columns 'Open', 'High', 'Low', 'Close' and (optionally) 'Volume'. |
| window_size       | Required*     | Window size of observation for each step|
| add_feature       | False     | add reward and position size features on observation|
| mode              | "sequential"  | If "sequential", the next episode begins with a continuation of the time stamp of the previous episode. If "random", The next episode begins with a random timestamp. |
| step_length       | 1             | How much to shift the timestamp of the next step |
| cash              | 10,000        | cash                  |
| commission        | .0            | commission            |
| margin            | 1.            | margin                |
| trade_on_close    | False         | trade_on_close        |
| hedging           | False         | hedging               |
| exclusive_orders  | False         | exclusive_orders      |

Parameters after cash are backtesting.py parameters. Please see documentation https://kernc.github.io/backtesting.py/

### Step Parameters

| Name              | Default       | Description           |
| ---               | ---           | ---                   |
| action     | Required*               | Action is [0, 1, 2] 0: HOLD, 1: BUY, 2: SELL|
| size       | _FULL_EQUITY(0.999)     | Size               |
| limit      | None                    | Limit Rate         |
| stop       | None                    | Stop Rate          |
| sl         | None                    | Stop loss rate     |
| tp         | None                    | Take profit rate   |

Parameters excluding action are backtesting.py parameters. Please see documentation https://kernc.github.io/backtesting.py/


Step Example:
```
obs, reward, done, info = env.step(action, size=0.1, limit=120.13, stop=119.13, sl=118.13, tp=121.13)
```

### Plot Parameters

| Name              | Default       | Description           |
| ---               | ---           | ---                   |
| results           | None       | Results |
| filename          | None       | Filename |
| plot_width        | None       | Plot width |
| plot_equity       | True       | Plot equity |
| plot_return       | False      | Plot return |
| plot_pl           | True       | Plot pl |
| plot_volume       | True       | Plot volume |
| plot_drawdown     | False      | Plot drawdown |
| smooth_equity     | False      | Smooth equity |
| relative_equity   | True       | Relative equity |
| superimpose       | True       | Superimpose |
| resample          | True       | Resample |
| reverse_indicators| False      | Reverse indicators |
| show_legend       | True       | Show legend |
| open_browser      | True       | Open browser |

You don't need input results parameter.  
Parameters are backtesting.py parameters. Please see documentation https://kernc.github.io/backtesting.py/

Plot Example:
```
env.plot(
    filename="test.html",
    plot_equity=True,
    plot_return=True,
    plot_pl=True,
    plot_volume=True,
    plot_drawdown=True,
    smooth_equity=True,
    relative_equity=True,
    superimpose=True,
    resample=True,
    reverse_indicators=True,
    show_legend=True,
    open_browser=True,
)
```

You can view on browser.

<img width="1352" alt="スクリーンショット 2021-09-28 13 25 52" src="https://user-images.githubusercontent.com/43264434/135023130-72e6ef12-8e0e-4698-8057-6c7ddcd25a67.png">
