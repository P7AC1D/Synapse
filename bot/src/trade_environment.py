import gymnasium
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces

class TradingEnv(gymnasium.Env):
    metadata = {"render_modes": None, "render_fps": None}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lot_percentage: float = 0.02, bar_count: int = 10, 
                 random_start: bool = False):
        super().__init__()
        
        # Trading constants
        self.POINT_VALUE = 0.01
        self.PIP_VALUE = 0.0001
        self.MIN_LOTS = 0.01
        self.MAX_LOTS = 100.0
        self.CONTRACT_SIZE = 1.0
        self.MIN_SL_POINTS = 100.0  # Minimum stop loss in price points
        self.MIN_RR_RATIO = 2.0     # Minimum risk/reward ratio
        
        self.prices = {
            'close': data['close'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'spread': data['spread'].values,
            'atr': data['ATR'].values
        }
        
        self.raw_data = self._preprocess_data(data)
        self.current_step = 0
        self.bar_count = bar_count
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
        self.completed_episodes = 0
        self.episode_steps = 0
        
        self._setup_action_space()
        self._setup_observation_space(5)  # 5 features

    def _setup_action_space(self) -> None:
        """Configure simple action space for position direction only."""
        self.action_space = spaces.Box(
            low=np.array([-1]),  # Sell
            high=np.array([1]),  # Buy
            dtype=np.float32
        )

    def _setup_observation_space(self, feature_count: int) -> None:
        """Setup observation space for features only."""
        obs_dim = self.bar_count * feature_count
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_dim,), dtype=np.float32
        )

    def _process_action(self, action: np.ndarray) -> Tuple[int, float, float]:
        """Process action with fixed RRR and ATR-based stop loss."""
        position = np.sign(action[0]) if abs(action[0]) > 0.1 else 0
        
        # Calculate SL based on ATR
        current_atr = self.prices['atr'][self.current_step]
        sl_points = max(current_atr, self.MIN_SL_POINTS)  # At least 100 points or 1 ATR
        
        # Fixed RRR of 1.0
        tp_points = sl_points
        
        return position, sl_points, tp_points

    def _execute_trade(self, position: int, sl_points: float, tp_points: float, raw_spread: float) -> None:
        """Execute a trade with enforced minimum stop loss."""
        if position == 0:
            self.steps_since_trade += 1
            return
            
        current_price = self.prices['close'][self.current_step]

        # Enforce minimum SL and TP distances
        sl_points = max(sl_points, self.MIN_SL_POINTS)
        tp_points = max(tp_points, sl_points * self.MIN_RR_RATIO)

        if position == 1:  # BUY
            entry_price = current_price + raw_spread
            sl_price = entry_price - sl_points
            tp_price = entry_price + tp_points
        else:  # SELL
            entry_price = current_price - raw_spread
            sl_price = entry_price + sl_points
            tp_price = entry_price - tp_points
        
        rrr = tp_points / sl_points if sl_points > 0 else 0
        
        # Position sizing based on risk amount
        stop_loss_distance = abs(entry_price - sl_price)
        value_per_lot = stop_loss_distance * self.CONTRACT_SIZE
        risk_amount = self.lot_percentage * self.balance
        lot_size = risk_amount / value_per_lot
        
        lot_size = min(max(round(lot_size, 2), self.MIN_LOTS), self.MAX_LOTS)
        
        self.open_positions.append({
            "position": position,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lot_size": lot_size,
            "entry_step": self.current_step,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "rrr": rrr,
            "entry_spread": raw_spread,
            "entry_atr": self.prices['atr'][self.current_step]
        })
        self.steps_since_trade = 0

    def _evaluate_positions(self, high_price: float, low_price: float, raw_spread: float,
                          prev_balance: float) -> Tuple[float, List[int]]:
        """Evaluate open positions and calculate rewards."""
        reward = 0.0
        closed_positions = []
        current_price = self.prices['close'][self.current_step]
        
        if not self.open_positions:
            return reward, closed_positions
        
        positions = np.array([pos["position"] for pos in self.open_positions])
        entry_prices = np.array([pos["entry_price"] for pos in self.open_positions])
        lot_sizes = np.array([pos["lot_size"] for pos in self.open_positions])
        
        points = np.where(positions == 1,
                         current_price - entry_prices,
                         entry_prices - current_price) / self.POINT_VALUE
        
        unrealized_pnls = points * lot_sizes * self.POINT_VALUE
        
        rrr_scales = np.minimum([pos["rrr"] for pos in self.open_positions], 3.0)
        positive_mask = unrealized_pnls > 0
        reward_multipliers = np.where(positive_mask, 0.1 * rrr_scales, 0.05)
        reward += np.sum((unrealized_pnls / prev_balance) * reward_multipliers)
        
        buy_mask = positions == 1
        tp_prices = np.array([pos["tp_price"] for pos in self.open_positions])
        sl_prices = np.array([pos["sl_price"] for pos in self.open_positions])
        
        hit_tp = np.where(buy_mask,
                         high_price + raw_spread >= tp_prices,
                         low_price - raw_spread <= tp_prices)
        hit_sl = np.where(buy_mask,
                         low_price - raw_spread <= sl_prices,
                         high_price + raw_spread >= sl_prices)
        
        for i, (pos, tp_hit, sl_hit) in enumerate(zip(self.open_positions, hit_tp, hit_sl)):
            if tp_hit or sl_hit:
                exit_price = pos["tp_price"] if tp_hit else pos["sl_price"]
                points = (exit_price - pos["entry_price"]) if pos["position"] == 1 else (pos["entry_price"] - exit_price)
                points /= self.POINT_VALUE
                
                pnl = points * pos["lot_size"] * self.POINT_VALUE
                pos["actual_rrr"] = pos["tp_points"] / pos["sl_points"] if pos["sl_points"] > 0 else 0
                
                reward += (pnl / prev_balance)
                self.balance += pnl
                
                if pnl > 0:
                    reward += min(pos["rrr"] / 1.0, 3.0) * 2.0
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                closed_positions.append(i)
                pos.update({
                    "pnl": pnl,
                    "exit_price": exit_price,
                    "exit_step": self.current_step,
                    "hit_tp": tp_hit
                })
                self.trades.append(pos)
        
        return reward, closed_positions

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data for the model with simplified features."""
        close = data['close'].values
        
        # Core price features
        returns = np.diff(close) / close[:-1]
        returns = np.insert(returns, 0, 0)
        returns = np.clip(returns, -0.1, 0.1)
        
        # Simple volatility (10-period)
        vol = pd.Series(returns).rolling(10, min_periods=1).std().fillna(0).values
        vol = np.clip(vol, 1e-8, 0.1)
        
        # Basic trend
        trend = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)
        
        # Normalized features
        rsi_norm = data['RSI'].values / 100  # Scale to 0-1
        atr_norm = data['ATR'].values / close  # Normalize by price
        
        features = np.column_stack([
            returns,    # Price movement
            vol,       # Market volatility
            trend,     # Trend direction
            rsi_norm,  # Momentum
            atr_norm   # Volatility for position sizing
        ])
        
        features = np.nan_to_num(features, 0)
        return pd.DataFrame(features)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an environment step."""
        position, sl_points, tp_points = self._process_action(action)

        high_price = self.prices['high'][self.current_step]
        low_price = self.prices['low'][self.current_step]
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.episode_steps += 1
        self.current_step += 1
        
        done = (self.episode_steps >= self.max_steps) or (self.current_step >= len(self.raw_data) - 1)

        trade_reward = 0
        growth_reward = 0
        drawdown_penalty = 0

        position_reward, closed_positions = self._evaluate_positions(
            high_price, low_price, current_spread, previous_balance
        )
        trade_reward += position_reward
        
        for i in sorted(closed_positions, reverse=True):
            self.open_positions.pop(i)
            
        if position != 0:
            self._execute_trade(position, sl_points, tp_points, current_spread)
            trade_reward += 0.1  # Small reward for taking trades
        
        if previous_balance > 0:
            growth_ratio = self.balance / self.initial_balance
            if growth_ratio > 1:
                growth_reward += np.log(growth_ratio) * 5.0
            
            if hasattr(self, 'last_balance_ratio'):
                growth_acceleration = self.balance / previous_balance - self.last_balance_ratio
                if growth_acceleration > 0:
                    growth_reward += growth_acceleration * 2.0
            self.last_balance_ratio = self.balance / previous_balance
            
            if self.balance < self.max_balance:
                drawdown = (self.max_balance - self.balance) / self.max_balance
                drawdown_penalty = drawdown * 5.0

        reward = (trade_reward * 0.9) + (growth_reward * 0.15) - (drawdown_penalty * 0.05)
        
        # Penalize waiting too long between trades
        if self.steps_since_trade > 5:
            reward -= (self.steps_since_trade - 5) * 0.05
        
        # Add trading frequency bonus
        if len(self.trades) > 0:
            trade_frequency = len(self.trades) / self.episode_steps
            reward += trade_frequency * 2.0
        
        # Check for bankruptcy or excessive drawdown
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        if self.balance <= 0:
            self.open_positions.clear()
            reward = -100
            done = True
        elif max_drawdown >= 0.5:
            self.open_positions.clear()
            reward = -50
            done = True

        if done:
            final_return = (self.balance / self.initial_balance) - 1
            win_rate = (self.win_count / (self.win_count + self.loss_count)) if (self.win_count + self.loss_count) > 0 else 0
            
            if self.balance > self.initial_balance:
                reward += final_return * 20.0
                reward += win_rate * 20.0
                reward += (len(self.trades) / 50.0) * 10.0
            else:
                reward += final_return * 15.0
        
        obs = self.get_history()
        self.reward = reward
        
        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        min_episode_length = 500
        max_episode_length = min(5000, len(self.raw_data) - self.bar_count)
        
        scale_factor = min(self.completed_episodes / 25, 1.0)
        self.max_steps = int(min_episode_length + (max_episode_length - min_episode_length) * scale_factor)
        
        if self.random_start:
            latest_start = len(self.raw_data) - self.max_steps - self.bar_count
            self.current_step = np.random.randint(0, max(1, latest_start))
        else:
            self.current_step = 0
            
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.open_positions.clear()
        self.trades.clear()
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.episode_steps = 0
        
        self.completed_episodes += 1
        
        return self.get_history(), {}
        
    def get_history(self) -> np.ndarray:
        """Get the observation window."""
        start = max(0, self.current_step - self.bar_count + 1)
        end = self.current_step + 1
        window = self.raw_data.values[start:end]
        
        if window.shape[0] < self.bar_count:
            full_window = np.zeros((self.bar_count, window.shape[1]), dtype=np.float32)
            full_window[-window.shape[0]:] = window
            window = full_window
        
        return window.ravel()

    def render(self) -> None:
        """Print environment state and trade statistics."""
        print(f"\n===== Episode {self.completed_episodes}, Step {self.episode_steps}/{self.max_steps} =====")
        print(f"Open Positions: {len(self.open_positions)}")
        
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
        
        avg_rrr = trades_df["actual_rrr"].mean() if total_trades > 0 else 0.0
        avg_holding_length = (trades_df["exit_step"] - trades_df["entry_step"]).mean() if total_trades > 0 else 0.0
        
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
            f"Win Rate: {total_win_rate:.2f}%\n"
            f"Buy Win Rate: {buy_win_rate:.2f}%\n"
            f"Sell Win Rate: {sell_win_rate:.2f}%\n"
            f"Average TP Profit: {avg_pnl_tp:.2f}\n"
            f"Average SL Loss: {avg_pnl_sl:.2f}\n"
            f"Average RRR: {avg_rrr:.2f}\n"
            f"Average Hold Length: {avg_holding_length:.1f} bars\n"
            f"Expected Value: {expected_value:.2f}\n"
            f"Kelly Criterion: {kelly_criteria:.4f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )
        
        print(metrics_text)
