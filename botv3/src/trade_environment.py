import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

def calculate_lot_size(balance, risk, entry_price, stop_loss, contract_size=1.0):
    """Calculates lot size based on balance risk percentage."""
    return min(max(round((balance * risk) / (abs(entry_price - stop_loss) * contract_size), 2), 0.01), 100.0)

def calculate_balance_change(lot_size, entry_price, exit_price, position, contract_size=1.0):
    """Calculates profit/loss based on trade parameters."""
    if position == 1:  # Buy
        return (exit_price - entry_price) * lot_size * contract_size
    elif position == -1:  # Sell
        return (entry_price - exit_price) * lot_size * contract_size
    return 0.0

class BitcoinTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, lot_percentage=0.01, bar_count=50):
        super(BitcoinTradingEnv, self).__init__()
        self.max_positions = 5
        self.position_features = 5
        self.steps_since_trade = 0
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_percentage = lot_percentage
        self.bar_count = bar_count
        self.trades = []
        self.open_positions = []
        self.max_balance = initial_balance  # Track highest balance for drawdown calc

        # SL/TP Levels
        self.sl_tp_levels = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

        # Action Space: (trade action, SL index, TP index, close position index)
        self.action_space = spaces.MultiDiscrete([
            4,  # 0: Hold, 1: Buy, 2: Sell, 3: Close
            len(self.sl_tp_levels),  # Stop-Loss selection
            len(self.sl_tp_levels),  # Take-Profit selection
            self.max_positions       # Select position to close
        ])

        # Observation space: historical bars + open positions
        obs_dim = (bar_count * data.shape[1]) + (self.max_positions * self.position_features)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

    def get_history(self):
        """Returns last 'bar_count' bars."""
        start = max(0, self.current_step - self.bar_count + 1)
        history = self.data.iloc[start:self.current_step+1].values
        if history.shape[0] < self.bar_count:
            padding = np.zeros((self.bar_count - history.shape[0], self.data.shape[1]))
            history = np.vstack((padding, history))
        return history.flatten()

    def get_open_positions(self):
        """Returns active positions in a fixed-size array."""
        positions_array = np.zeros((self.max_positions, self.position_features), dtype=np.float32)
        for i, pos in enumerate(self.open_positions[:self.max_positions]):
            positions_array[i] = [
                pos["position"],
                pos["entry_price"],
                pos["sl_price"],
                pos["tp_price"],
                pos["lot_size"]
            ]
        return positions_array.flatten()
        
    def step(self, action):
        trade_action, sl_index, tp_index = action[:3]
        current_price = self.data.iloc[self.current_step]["close"]

        # Track balance before update
        previous_balance = self.balance
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Initialize reward
        reward = 0

        # Evaluate existing positions
        for pos in self.open_positions:
            unrealized_pnl = calculate_balance_change(pos["lot_size"], pos["entry_price"], current_price, pos["position"])
            # Smaller focus on unrealized PnL
            reward += (unrealized_pnl / previous_balance)

        # Execute new trade
        sl_value = self.sl_tp_levels[sl_index]
        tp_value = self.sl_tp_levels[tp_index]

        if trade_action in [1, 2]:  # Buy or Sell
            if len(self.open_positions) >= self.max_positions:
                reward -= 5  # Penalize invalid trade attempt
            else:
                position = 1 if trade_action == 1 else -1
                sl_value = self.sl_tp_levels[sl_index]
                tp_value = self.sl_tp_levels[tp_index]
                entry_price = current_price
                sl_price = entry_price - sl_value if position == 1 else entry_price + sl_value
                tp_price = entry_price + tp_value if position == 1 else entry_price - tp_value
                lot_size = calculate_lot_size(self.balance, self.lot_percentage, entry_price, sl_price)

                self.open_positions.append({
                    "position": position,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot_size": lot_size,
                    "entry_step": self.current_step
                })

                reward += 1
                self.steps_since_trade = 0

        elif trade_action == 3 and self.open_positions:  # Close position
            close_index = action[3]
            if close_index < len(self.open_positions):
                trade_to_close = self.open_positions.pop(close_index)
                exit_price = current_price
                pnl = calculate_balance_change(trade_to_close["lot_size"], trade_to_close["entry_price"], exit_price, trade_to_close["position"])
                reward += (pnl / previous_balance)
                self.balance += pnl
                trade_to_close["pnl"] = pnl
                trade_to_close["exit_price"] = exit_price
                self.trades.append(trade_to_close)
            else:
                # Penalize incorrect index
                reward -= 5
        else:
            # Penalize inactivity
            self.steps_since_trade += 1
            reward -= self.steps_since_trade

        # Handle stop-loss/take-profit exits
        closed_positions = []
        for i, pos in enumerate(self.open_positions):
            hit_tp = current_price >= pos["tp_price"] if pos["position"] == 1 else current_price <= pos["tp_price"]
            hit_sl = current_price <= pos["sl_price"] if pos["position"] == 1 else current_price >= pos["sl_price"]

            if hit_tp or hit_sl:
                exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
                pnl = calculate_balance_change(pos["lot_size"], pos["entry_price"], exit_price, pos["position"])
                reward += (pnl / previous_balance) * 2
                self.balance += pnl
                closed_positions.append(i)
                pos["pnl"] = pnl
                pos["exit_price"] = exit_price
                self.trades.append(pos)

        for i in sorted(closed_positions, reverse=True):
            self.open_positions.pop(i)

        # Encourage higher R:R
        for trade in self.trades:
            risk = abs(trade["entry_price"] - trade["sl_price"])
            rr_ratio = abs(trade["exit_price"] - trade["entry_price"]) / risk
            reward += 5 if rr_ratio >= 1 else -2

        # Build observation
        obs = np.concatenate([self.get_history(), self.get_open_positions()])
        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.open_positions.clear()
        self.trades.clear()
        return np.concatenate([self.get_history(), self.get_open_positions()]), {}

    def render(self, mode="human", close=False):
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Max Balance: {self.max_balance:.2f}")
            print(f"Open Positions: {len(self.open_positions)}")
            for pos in self.open_positions:
                print(f"  Position: {pos['position']} Entry: {pos['entry_price']:.2f} SL: {pos['sl_price']:.2f}, TP: {pos['tp_price']:.2f}")
            print(f"Total Trades: {len(self.trades)}\n")