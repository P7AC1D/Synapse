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
        self.lot_percentage = lot_percentage
        self.trades = []
        self.open_positions = []  # Allows multiple concurrently open positions

        # Define discrete price differences for SL/TP
        self.sl_tp_levels = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

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
        current_price = self.data.iloc[self.current_step]["unscaled_close"]

        # Increment step and check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Initialize reward for this step
        reward = 0

        # Loop through open positions and calculate base reward
        for pos in self.open_positions:
            reward += calculate_balance_change(pos["lot_size"], pos["entry_price"], current_price, pos["position"])

        # New trade execution based on action
        sl_value = self.sl_tp_levels[sl_index]
        tp_value = self.sl_tp_levels[tp_index]

        if trade_action == 1:  # Buy
            new_position = {
                "position": 1,
                "entry_price": current_price,
                "sl_price": current_price - sl_value,
                "tp_price": current_price + tp_value,
                "lot_size": calculate_lot_size(self.balance, self.lot_percentage, current_price, current_price - sl_value)
            }
            self.open_positions.append(new_position)
            reward += 2

        elif trade_action == 2:  # Sell
            new_position = {
                "position": -1,
                "entry_price": current_price,
                "sl_price": current_price + sl_value,
                "tp_price": current_price - tp_value,
                "lot_size": calculate_lot_size(self.balance, self.lot_percentage, current_price, current_price + sl_value)
            }
            self.open_positions.append(new_position)
            reward += 2

        elif trade_action == 0:  # Inaction
            reward -= 5.0

        # Check all open positions for SL or TP triggers
        closed_positions = []
        for pos in self.open_positions:
            if pos["position"] == 1:
                # For long trades: TP if price has risen; SL if dropped
                if current_price >= pos["tp_price"]:
                    profit = calculate_balance_change(pos["lot_size"], pos["entry_price"], pos["tp_price"], pos["position"])
                    reward += (profit / self.balance) * 100
                    self.balance += profit
                    pos["exit"] = current_price
                    pos["reward"] = (profit / self.balance) * 100
                    pos["pnl"] = profit
                    closed_positions.append(pos)
                elif current_price <= pos["sl_price"]:
                    loss = -calculate_balance_change(pos["lot_size"], pos["entry_price"], pos["sl_price"], pos["position"])
                    reward += -(loss / self.balance) * 100
                    self.balance -= loss
                    pos["exit"] = current_price
                    pos["reward"] = -(loss / self.balance) * 100
                    pos["pnl"] = -loss
                    closed_positions.append(pos)
            elif pos["position"] == -1:
                # For short trades: TP if price has fallen; SL if risen
                if current_price <= pos["tp_price"]:
                    profit = calculate_balance_change(pos["lot_size"], pos["entry_price"], pos["tp_price"], pos["position"])
                    reward += (profit / self.balance) * 100
                    self.balance += profit
                    pos["exit"] = current_price
                    pos["reward"] = (profit / self.balance) * 100
                    pos["pnl"] = profit
                    closed_positions.append(pos)
                elif current_price >= pos["sl_price"]:
                    loss = -calculate_balance_change(pos["lot_size"], pos["entry_price"], pos["sl_price"], pos["position"])
                    reward += -(loss / self.balance) * 100
                    self.balance -= loss
                    pos["exit"] = current_price
                    pos["reward"] = -(loss / self.balance) * 100
                    pos["pnl"] = -loss
                    closed_positions.append(pos)

        # Remove closed positions from open_positions and record them
        for pos in closed_positions:
            self.trades.append(pos)
            self.open_positions.remove(pos)

        # Optional bonus for consecutive winning trades
        if len(self.trades) > 1 and self.trades[-1]["pnl"] > 0 and self.trades[-2]["pnl"] > 0:
            reward += 1

        # Bankruptcy and end-of-episode adjustments
        if self.balance <= 0:
            self.balance = 0
            done = True

        if done and (self.balance > self.initial_balance):
            reward += (self.balance - self.initial_balance) * 0.1

        if done and len(self.trades) == 0:
            reward -= 5

        obs = self.data.iloc[self.current_step].values
        # Gymnasium expects (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.open_positions = []
        self.trades = []
        return self.data.iloc[self.current_step].values, {}