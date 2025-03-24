import gymnasium as gym
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lot_percentage: float = 0.01, bar_count: int = 50, 
                 normalization_window: int = 100, random_start: bool = False):
        """Trading environment focused on balance growth."""
        super(TradingEnv, self).__init__()
        
        self.raw_data = data.copy()
        self.current_step = 0
        self.bar_count = bar_count
        self.normalization_window = normalization_window
        self.random_start = random_start
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.lot_percentage = lot_percentage
        
        self.trades: List[Dict[str, Any]] = []
        self.open_positions: List[Dict[str, Any]] = []
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        
        self._setup_action_space()
        self._setup_observation_space(data.shape[1])
    
    def _setup_action_space(self) -> None:
        """Configure combined continuous action space."""
        self.action_space = spaces.Box(
            low=np.array([-1, 1000, 1000]),
            high=np.array([1, 10000, 10000]),
            dtype=np.float32
        )
    
    def _setup_observation_space(self, feature_count: int) -> None:
        """Configure the observation space dimensions."""
        obs_dim = self.bar_count * feature_count
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )
        
    def normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize data using only past information."""
        lookback_start = max(0, self.current_step - self.normalization_window)
        lookback_end = self.current_step + 1
        history_window = self.raw_data.iloc[lookback_start:lookback_end].values
        scaler = StandardScaler()
        scaler.fit(history_window)
        return scaler.transform(window)

    def get_history(self) -> np.ndarray:
        """Return normalized data window for observation."""
        start = max(0, self.current_step - self.bar_count + 1)
        end = self.current_step + 1
        window = self.raw_data.iloc[start:end].values
        normalized_window = self.normalize_window(window)
        if normalized_window.shape[0] < self.bar_count:
            padding = np.zeros((self.bar_count - normalized_window.shape[0], normalized_window.shape[1]))
            normalized_window = np.vstack((padding, normalized_window))
        return normalized_window.flatten()

    def _process_action(self, action: np.ndarray) -> Tuple[int, float, float]:
        """Process the continuous action into position and SL/TP points."""
        # Convert continuous position (-1 to 1) into discrete decision
        if action[0] > 0.33:  # Upper third for buy
            position = 1
        elif action[0] < -0.33:  # Lower third for sell
            position = -1
        else:  # Middle third for hold
            position = 0
            
        # Clip SL/TP within bounds but don't enforce RRR
        sl_points = np.clip(action[1], 1000, 10000)
        tp_points = np.clip(action[2], 1000, 10000)
        
        return position, sl_points, tp_points
    
    def _execute_trade(self, position: int, sl_points: float, tp_points: float,
                      current_price: float, spread: float) -> None:
        """Execute a trade with no minimum RRR requirement."""
        if position == 0:
            self.steps_since_trade += 1
            return
            
        if len(self.open_positions) >= 2:
            return
            
        long_positions = sum(1 for p in self.open_positions if p["position"] == 1)
        short_positions = sum(1 for p in self.open_positions if p["position"] == -1)
        
        if position == 1 and long_positions >= 1:
            return
        if position == -1 and short_positions >= 1:
            return
            
        entry_price = current_price
        sl_price = entry_price - sl_points if position == 1 else entry_price + sl_points
        tp_price = entry_price + tp_points if position == 1 else entry_price - tp_points
        
        # Calculate RRR
        rrr = tp_points / sl_points if sl_points > 0 else 0
        
        # Full risk for first position
        risk_amount = self.lot_percentage * self.balance
        
        # Reduce risk by 60% for second position
        if len(self.open_positions) > 0:
            risk_amount *= 0.4
        
        lot_size = risk_amount / abs(entry_price - sl_price)
        lot_size = min(max(round(lot_size, 2), 0.01), 100.0)
        
        self.open_positions.append({
            "position": position,
            "entry_price": entry_price,
            "spread": spread,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lot_size": lot_size,
            "entry_step": self.current_step,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "rrr": rrr
        })
        self.steps_since_trade = 0
            
    def _evaluate_positions(self, high_price: float, low_price: float, 
                          current_price: float, spread: float, 
                          prev_balance: float) -> Tuple[float, List[int]]:
        """Evaluate positions with RRR-based rewards."""
        reward = 0
        closed_positions = []
        
        # Unrealized positions get small RRR-scaled rewards for price movement
        for pos in self.open_positions:
            unrealized_pnl = 0
            if pos["position"] == 1:
                unrealized_pnl = ((current_price - pos["entry_price"]) - spread) * pos["lot_size"]
            else:
                unrealized_pnl = ((pos["entry_price"] - current_price) - spread) * pos["lot_size"]
            
            # Small RRR bonus for positive unrealized PnL
            if unrealized_pnl > 0:
                rrr_scale = min(pos["rrr"] / 1.5, 2.0)  # Cap at 2x reward
                reward += (unrealized_pnl / prev_balance) * 0.1 * rrr_scale
            else:
                reward += (unrealized_pnl / prev_balance) * 0.1
        
        for i, pos in enumerate(self.open_positions):
            hit_tp = (high_price >= pos["tp_price"] if pos["position"] == 1 
                     else low_price <= pos["tp_price"])
            hit_sl = (low_price <= pos["sl_price"] if pos["position"] == 1 
                     else high_price >= pos["sl_price"])

            if hit_tp or hit_sl:
                exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
                pnl = 0
                if pos["position"] == 1:
                    pnl = ((exit_price - pos["entry_price"]) - spread) * pos["lot_size"]
                else:
                    pnl = ((pos["entry_price"] - exit_price) - spread) * pos["lot_size"]
                
                # Base reward from PnL
                reward += (pnl / prev_balance)
                self.balance += pnl
                
                # Additional RRR-based rewards/penalties
                if pnl > 0:
                    # Winning trade gets bonus based on RRR
                    rrr_bonus = min(pos["rrr"] / 1.5, 2.0)  # Cap at 2x bonus
                    reward += rrr_bonus
                    self.win_count += 1
                else:
                    # Losing trade gets penalty based on bad RRR
                    if pos["rrr"] < 1.0:
                        reward -= (1.0 - pos["rrr"]) * 0.5  # Penalty for poor RRR
                    self.loss_count += 1
                    
                closed_positions.append(i)
                pos["pnl"] = pnl
                pos["exit_price"] = exit_price
                pos["exit_step"] = self.current_step
                pos["hit_tp"] = hit_tp
                self.trades.append(pos)

        return reward, closed_positions
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one environment step focusing on balance growth."""
        position, sl_points, tp_points = self._process_action(action)

        current_price = self.raw_data.iloc[self.current_step]["close"]
        high_price = self.raw_data.iloc[self.current_step]["high"]
        low_price = self.raw_data.iloc[self.current_step]["low"]
        spread = self.raw_data.iloc[self.current_step]["spread"] / 100.0

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        trade_reward = 0
        growth_reward = 0
        drawdown_penalty = 0

        position_reward, closed_positions = self._evaluate_positions(
            high_price, low_price, current_price, spread, previous_balance
        )
        trade_reward += position_reward
        
        for i in sorted(closed_positions, reverse=True):
            self.open_positions.pop(i)
            
        if position != 0:
            self._execute_trade(position, sl_points, tp_points, current_price, spread)

        if previous_balance > 0:
            growth_ratio = self.balance / self.initial_balance
            if growth_ratio > 1:
                growth_reward += np.log(growth_ratio) * 10.0
            
            balance_change_ratio = self.balance / previous_balance
            if hasattr(self, 'last_balance_ratio'):
                growth_acceleration = balance_change_ratio - self.last_balance_ratio
                if growth_acceleration > 0:
                    growth_reward += growth_acceleration * 5.0
            self.last_balance_ratio = balance_change_ratio
            
            if self.balance < self.max_balance:
                drawdown = (self.max_balance - self.balance) / self.max_balance
                drawdown_penalty = drawdown * 5.0

        reward = (trade_reward * 0.3) + (growth_reward * 0.6) - (drawdown_penalty * 0.1)
        
        if self.balance <= 0:
            self.open_positions.clear()
            reward = -20
            done = True

        if done:
            final_return = (self.balance / self.initial_balance) - 1
            reward += final_return * 20.0
            
            if self.balance > self.initial_balance:
                reward += (np.log(self.balance / self.initial_balance) * 10.0)
        
        obs = self.get_history()
        self.reward = reward

        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        if self.random_start:
            self.current_step = np.random.randint(0, len(self.raw_data) - self.bar_count)
        else:
            self.current_step = 0
            
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.open_positions.clear()
        self.trades.clear()
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        
        return self.get_history(), {}

    def render(self) -> None:
        """Print current environment state."""
        print(f"\n===== Environment State at Step {self.current_step} =====")
        print(f"Open Positions: {len(self.open_positions)}")
        for pos in self.open_positions:
            print(f"  Position: {pos['position']} Entry: {pos['entry_price']:.2f} "
                  f"SL: {pos['sl_price']:.2f}, TP: {pos['tp_price']:.2f} "
                  f"RRR: {pos['rrr']:.2f}")
        
        if len(self.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        num_tp = sum(1 for trade in self.trades if trade["pnl"] > 0.0)
        num_sl = sum(1 for trade in self.trades if trade["pnl"] < 0.0)
        total_trades = len(self.trades)
        
        avg_pnl_tp = trades_df[trades_df["pnl"] > 0.0]["pnl"].mean() if num_tp > 0 else 0.0
        avg_pnl_sl = abs(trades_df[trades_df["pnl"] < 0.0]["pnl"].mean()) if num_sl > 0 else 0.0
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        expected_value = trades_df["pnl"].mean() if total_trades > 0 else 0.0
        
        avg_rrr = trades_df["rrr"].mean() if total_trades > 0 else 0.0
        
        num_buy = trades_df[trades_df["position"] == 1].shape[0]
        num_sell = trades_df[trades_df["position"] == -1].shape[0]
        
        buy_win_rate = (trades_df[(trades_df["position"] == 1) & (trades_df["pnl"] > 0.0)].shape[0] / num_buy * 100) if num_buy > 0 else 0.0
        sell_win_rate = (trades_df[(trades_df["position"] == -1) & (trades_df["pnl"] > 0.0)].shape[0] / num_sell * 100) if num_sell > 0 else 0.0
        total_win_rate = (num_tp / total_trades * 100) if total_trades > 0 else 0.0

        def kelly_criterion(win_rate, win_loss_ratio):
            if win_loss_ratio == 0:
                return 0.0
            return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
            
        def sharpe_ratio(returns, risk_free_rate=0.00, trading_periods=252):
            if len(returns) < 2:
                return 0.0
            excess_returns = np.array(returns) - (risk_free_rate / trading_periods)
            return (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(trading_periods)

        kelly_criteria = kelly_criterion(total_win_rate / 100.0, avg_rrr)
        sharpe = sharpe_ratio(trades_df["pnl"] / self.initial_balance) if total_trades > 0 else 0.0

        metrics_text = (
            f"\n===== Trading Performance Metrics =====\n"
            f"Current Balance: {self.balance:.2f}\n"
            f"Total Return: {total_return:.2f}%\n"
            f"Total Trades: {total_trades}\n"
            f"Total Win Rate: {total_win_rate:.2f}%\n"
            f"Long Trades: {num_buy}\n"
            f"Long Win Rate: {buy_win_rate:.2f}%\n"
            f"Short Trades: {num_sell}\n"
            f"Short Win Rate: {sell_win_rate:.2f}%\n"
            f"Average Win: {avg_pnl_tp:.2f}\n"
            f"Average Loss: {avg_pnl_sl:.2f}\n"
            f"Average RRR: {avg_rrr:.2f}\n"
            f"Expected Value: {expected_value:.2f}\n"
            f"Kelly Criterion: {kelly_criteria:.2f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )
        print(metrics_text)

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        np.random.seed(seed)
