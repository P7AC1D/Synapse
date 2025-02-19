import gymnasium as gym
import numpy as np
import random
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
    def __init__(self, data, initial_balance=10000, lot_percentage=0.01, bar_count=50):
        super(BitcoinTradingEnv, self).__init__()
        self.max_positions = 5
        self.position_features = 6
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_percentage = lot_percentage
        self.bar_count = bar_count
        self.trades = []
        self.open_positions = {}

        # Define discrete price differences for SL/TP
        self.sl_tp_levels = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

        # Action space: (trade action, stop-loss index, take-profit index)
        self.action_space = spaces.MultiDiscrete([
            4,  # 0: Hold, 1: Buy, 2: Sell, 3: Close position
            len(self.sl_tp_levels),  # Stop-Loss selection
            len(self.sl_tp_levels),  # Take-Profit selection
            self.max_positions       # Position selection for closure
        ])

        # Observation space: history of bar_count bars + open positions features
        obs_dim = (bar_count * data.shape[1]) + (self.max_positions * self.position_features)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

    def _generate_trade_id(self):
        return random.randint(100000, 999999)
    
    def _get_history(self):
        # Get the last bar_count bars; pad with zeros if not enough bars exist yet.
        start = max(0, self.current_step - self.bar_count + 1)
        history = self.data.iloc[start:self.current_step+1].values
        if history.shape[0] < self.bar_count:
            padding = np.zeros((self.bar_count - history.shape[0], self.data.shape[1]))
            history = np.vstack((padding, history))
        return history.flatten()

    def _get_open_positions(self):
        positions_array = np.zeros((self.max_positions, self.position_features), dtype=np.float32)
        open_keys = list(self.open_positions.keys())
        for i, key in enumerate(open_keys[:self.max_positions]):
            pos = self.open_positions[key]
            positions_array[i] = [
                pos.get("trade_id", 0),
                pos["position"],
                pos["entry_price"],
                pos["sl_price"],
                pos["tp_price"],
                pos["lot_size"]
            ]
        return positions_array.flatten()
        
    def step(self, action):
        trade_action, sl_index, tp_index = action[:3]
        current_price = self.data.iloc[self.current_step]["unscaled_close"]
        
        # Store balance before updating
        previous_balance = self.balance

        # Increment step and check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Initialize reward for this step
        reward = 0

        # Loop through open positions and calculate base reward
        for pos in self.open_positions.values():
            unrealised_pnl = calculate_balance_change(pos["lot_size"], pos["entry_price"], current_price, pos["position"])
            reward += (unrealised_pnl / previous_balance) * 25

        # New trade execution based on action
        sl_value = self.sl_tp_levels[sl_index]
        tp_value = self.sl_tp_levels[tp_index]

        if trade_action == 1:  # Buy
            trade_id = self._generate_trade_id()
            self.open_positions[trade_id] = {
                "trade_id": trade_id,
                "position": 1,
                "entry_price": current_price,
                "sl_price": current_price - sl_value,
                "tp_price": current_price + tp_value,
                "lot_size": calculate_lot_size(self.balance, self.lot_percentage, current_price, current_price - sl_value)
            }

        elif trade_action == 2:  # Sell
            trade_id = self._generate_trade_id()
            self.open_positions[trade_id] = {
                "trade_id": trade_id,
                "position": -1,
                "entry_price": current_price,
                "sl_price": current_price + sl_value,
                "tp_price": current_price - tp_value,
                "lot_size": calculate_lot_size(self.balance, self.lot_percentage, current_price, current_price + sl_value)
            }

        elif trade_action == 3 and self.open_positions:  # Close position
            close_index = action[3]
            open_keys = list(self.open_positions.keys())
            if close_index < len(open_keys):
                trade_key = open_keys[close_index]
                trade_to_close = self.open_positions.pop(trade_key)
                exit_price = current_price  # or use trade-specific SL/TP logic if needed
                pnl = calculate_balance_change(trade_to_close["lot_size"], trade_to_close["entry_price"], exit_price, trade_to_close["position"])
                reward += (pnl / previous_balance) * 100
                self.balance += pnl
                trade_to_close.update({"exit": current_price, "reward": (pnl / previous_balance) * 100, "pnl": pnl})
                self.trades.append(trade_to_close)

        # Track closed positions from SL/TP hit
        closed_positions = []
        for key, pos in list(self.open_positions.items()):
            hit_tp = current_price >= pos["tp_price"] if pos["position"] == 1 else current_price <= pos["tp_price"]
            hit_sl = current_price <= pos["sl_price"] if pos["position"] == 1 else current_price >= pos["sl_price"]

            if hit_tp or hit_sl:
                exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
                pnl = calculate_balance_change(pos["lot_size"], pos["entry_price"], exit_price, pos["position"])
                reward += (pnl / previous_balance) * 100
                self.balance += pnl

                pos.update({"exit": current_price, "reward": (pnl / previous_balance) * 100, "pnl": pnl})
                closed_positions.append(key)
                self.trades.append(pos)

        # Remove closed positions
        for key in closed_positions:
            self.open_positions.pop(key, None)

        if len(self.trades) >= 5:
            last_trades = self.trades[-5:]
            if all(trade["pnl"] > 0 for trade in last_trades):
                reward += 1
            elif all(trade["pnl"] < 0 for trade in last_trades):
                reward -= 1

        # Bankruptcy check
        if self.balance <= 0:
            self.balance = 0
            self.open_positions.clear()
            done = True

        # End-of-episode adjustments
        if done and (self.balance > self.initial_balance):
            reward += (self.balance - self.initial_balance) * 0.1
        elif done and len(self.trades) == 0:
            reward -= 5

        # Build observation using a history of bars and current open positions
        obs = np.concatenate([self._get_history(), self._get_open_positions()])
        terminated = done
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.open_positions = {}
        self.trades = []

        # Build the initial observation using history (will be padded if not enough bars) and open positions
        obs = np.concatenate([self._get_history(), self._get_open_positions()])
        return obs, {}