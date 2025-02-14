import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

RATES_CSV_PATH = "../data/rates.csv"

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

df = pd.read_csv(RATES_CSV_PATH)
df.set_index('time', inplace=True)

start_datetime = df.index[0]
end_datetime = df.index[-1]
print(f"Data collected from {start_datetime} to {end_datetime}")

print(df.tail())

# Split train/test (80/20)
split_idx = int(len(df) * 0.8)
train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

import gymnasium as gym
import numpy as np
from gymnasium import spaces

def calculate_lot_size(balance, risk, entry_price, stop_loss, contract_size=1.0):
    """
    Calculate lot size as a function of a fixed percentage of balance (risk)
    """
    return min(max(round((balance * risk) / (abs(entry_price - stop_loss) * contract_size), 2), 0.01), 100.0)


def calculate_balance_change(lot_size, entry_price, exit_price, position, contract_size=1.0):
    """
    Calculate the profit or loss based on trade parameters.
    
    For a long trade (position == 1), profit = (exit_price - entry_price) * lot_size * contract_size.
    For a short trade (position == -1), profit = (entry_price - exit_price) * lot_size * contract_size.
    """
    if position == 1:
        return (exit_price - entry_price) * lot_size * contract_size
    elif position == -1:
        return (entry_price - exit_price) * lot_size * contract_size
    else:
        return 0.0

class BitcoinTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, lot_percentage=0.01):
        super(BitcoinTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.lot_percentage = lot_percentage

        # Define discrete price differences for SL/TP
        self.sl_tp_levels = np.array([100, 200, 300, 400,500,600,700,800,900,1000])

        # Action space: (trade action, stop-loss index, take-profit index)
        self.action_space = spaces.MultiDiscrete([
            3,  # 0: Hold, 1: Buy, 2: Sell
            len(self.sl_tp_levels),  # Stop-Loss selection
            len(self.sl_tp_levels)   # Take-Profit selection
        ])

        # Observation space (OHLCV + indicators)
        obs_dim = data.shape[1]
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        trade_action, sl_index, tp_index = action
        current_price = self.data.iloc[self.current_step]["close"]
        self.current_step += 1
        sl_value = self.sl_tp_levels[sl_index]
        tp_value = self.sl_tp_levels[tp_index]
        reward = 0
        done = self.current_step >= len(self.data) - 1

        # Simulate trade execution
        if trade_action == 1:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.sl_price = self.entry_price - sl_value
            self.tp_price = self.entry_price + tp_value
            # Calculate lot size based on risk percentage and distance to SL
            self.lot_size = calculate_lot_size(self.balance, self.lot_percentage, self.entry_price, self.sl_price)
            # print(f"BUY | self.lot_size: {self.lot_size}")
            self.trade_open_steps = 0  
        elif trade_action == 2:  # Sell
            self.position = -1
            self.entry_price = current_price
            self.sl_price = self.entry_price + sl_value
            self.tp_price = self.entry_price - tp_value
            self.lot_size = calculate_lot_size(self.balance, self.lot_percentage, self.entry_price, self.sl_price)
            # print(f"SELL | self.lot_size: {self.lot_size}")
            self.trade_open_steps = 0  
        elif trade_action == 0 and self.position == 0:
            reward -= 0.1  # Small penalty for inaction

        # Holding penalty
        if self.position != 0:
            self.trade_open_steps += 1
            reward -= 0.001 * self.trade_open_steps  # Encourage taking profits

        # Reward based on hitting SL/TP
        if self.position == 1:
            if current_price >= self.tp_price:
                profit = calculate_balance_change(self.lot_size, self.entry_price, self.tp_price, self.position)
                reward += profit / sl_value  # Reward per unit risk
                self.balance += profit
                trade = {
                    "entry": self.entry_price,
                    "exit": current_price,
                    "position": self.position,
                    "reward": reward,
                    "pnl": profit
                }
                self.trades.append(trade)
                self.position = 0
            elif current_price <= self.sl_price:
                loss = calculate_balance_change(self.lot_size, self.entry_price, self.sl_price, self.position)
                reward -= loss / sl_value
                self.balance -= loss
                trade = {
                    "entry": self.entry_price,
                    "exit": current_price,
                    "position": self.position,
                    "reward": reward,
                    "pnl": profit
                }
                self.trades.append(trade)
                self.position = 0
        elif self.position == -1:
            if current_price <= self.tp_price:
                profit = calculate_balance_change(self.lot_size, self.entry_price, self.tp_price, self.position)
                reward += profit / sl_value
                self.balance += profit
                trade = {
                    "entry": self.entry_price,
                    "exit": current_price,
                    "position": self.position,
                    "reward": reward,
                    "pnl": profit
                }
                self.trades.append(trade)
                self.position = 0
            elif current_price >= self.sl_price:
                loss = calculate_balance_change(self.lot_size, self.entry_price, self.sl_price, self.position)
                reward -= loss / sl_value
                self.balance -= loss
                trade = {
                    "entry": self.entry_price,
                    "exit": current_price,
                    "position": self.position,
                    "reward": reward,
                    "pnl": profit
                }
                self.trades.append(trade)
                self.position = 0

        # Check for account bankruptcy
        if self.balance <= 0:
            self.balance = 0
            done = True

        obs = self.data.iloc[self.current_step].values
        return obs, reward, done, {}, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.open_position = None
        self.trades = []
        return self.data.iloc[self.current_step].values, {}

    def _get_obs(self):
        return self.data.iloc[self.current_step].values

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = BitcoinTradingEnv(df)
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=0, n_epochs=25)
model.learn(total_timesteps=1000000, progress_bar=True)
model.save("ppo_bitcoin_trader")

# Test
test_env = BitcoinTradingEnv(test_data, initial_balance=1000.0)

obs, info = test_env.reset()

balance_over_time = []
actions_log = []
done = False

while not done:
    action, _ = model.predict(obs)
    actions_log.append(action)
    obs, reward, done, _, _ = test_env.step(action)
    balance_over_time.append(test_env.balance)

print("Final balance:", test_env.balance)

# Create a DataFrame of opened positions and print the tail
trades_df = pd.DataFrame(test_env.trades)
print(f"Number of trades: {len(trades_df)}")
print(trades_df)

# Plot the balance over time
plt.plot(balance_over_time)
plt.xlabel("Time Steps")
plt.ylabel("Balance")
plt.title("Balance Over Time")
plt.show()