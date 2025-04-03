"""
Trading environment for grid-based trading with PPO-LSTM.

This module implements a custom OpenAI Gym environment for training
a PPO-LSTM model to trade using a dynamic grid strategy.
"""

import gymnasium
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces
import gymnasium as gym
from gymnasium.utils import EzPickle

class Grid:
    """Dynamic grid for position management."""
    
    def __init__(self, direction: int, initial_price: float, atr: float):
        self.direction = direction        # 1 for long, -1 for short
        self.base_price = initial_price   # Initial entry
        self.avg_entry = initial_price    # Weighted average entry
        self.positions = []               # List of active positions
        self.total_pnl = 0.0             # Current unrealized PnL
        self.total_lots = 0.0            # Total position size
        
        # Grid sizing parameters
        self.initial_atr = atr
        self.grid_size = self.calculate_grid_size(atr)
        
        # Dynamic bounds
        self.min_grid_size = atr * 0.5
        self.max_grid_size = atr * 3.0
        
    def calculate_grid_size(self, current_atr: float) -> float:
        """Calculate grid size based on volatility."""
        volatility_ratio = current_atr / self.initial_atr
        base_multiplier = 1.5  # Base ATR multiplier
        
        # Adjust multiplier based on volatility
        if volatility_ratio > 1.5:
            base_multiplier *= 0.8  # Tighter grid in high volatility
        elif volatility_ratio < 0.5:
            base_multiplier *= 1.2  # Wider grid in low volatility
            
        return current_atr * base_multiplier
        
    def update_metrics(self, current_price: float, current_atr: float) -> None:
        """Update grid metrics with adaptive sizing."""
        # Update grid size dynamically
        target_size = self.calculate_grid_size(current_atr)
        self.grid_size = self.grid_size * 0.8 + target_size * 0.2  # Smooth transition
        
        if self.positions:
            # Calculate weighted average entry
            total_value = sum(p["entry_price"] * p["lot_size"] for p in self.positions)
            self.total_lots = sum(p["lot_size"] for p in self.positions)
            self.avg_entry = total_value / self.total_lots
            
    def should_add_position(self, current_price: float) -> bool:
        """Determine if we should add a new position based on price movement."""
        # First position is always allowed
        if not self.positions:
            return True
        
        # Calculate price movement thresholds
        if self.direction == 1:  # Long grid
            # For longs, we want to add when price moves down enough
            price_movement = (current_price - self.avg_entry) / self.avg_entry
            grid_threshold = -self.grid_size / self.avg_entry
            
            # More aggressive entry for first few positions
            position_factor = max(0.7, 1.0 - len(self.positions) * 0.1)
            adjusted_threshold = grid_threshold * position_factor
            
            return price_movement < adjusted_threshold
            
        else:  # Short grid
            # For shorts, we want to add when price moves up enough
            price_movement = (current_price - self.avg_entry) / self.avg_entry
            grid_threshold = self.grid_size / self.avg_entry
            
            # More aggressive entry for first few positions
            position_factor = max(0.7, 1.0 - len(self.positions) * 0.1)
            adjusted_threshold = grid_threshold * position_factor
            
            return price_movement > adjusted_threshold

class TradingEnv(gym.Env, EzPickle):
    """Trading environment for grid-based trading with PPO-LSTM."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 balance_per_lot: float = 1000.0, random_start: bool = False,
                 bar_count: int = 10):
        super().__init__()
        EzPickle.__init__(self)
        
        # Save original datetime index
        self.original_index = data.index.copy() if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index)
        
        # Trading constants
        self.POINT_VALUE = 0.01
        self.PIP_VALUE = 0.0001
        self.MIN_LOTS = 0.01
        self.MAX_LOTS = 100.0
        self.CONTRACT_SIZE = 1.0
        self.BALANCE_PER_LOT = balance_per_lot
        self.MAX_DRAWDOWN = 0.5
        
        self.prices = {
            'close': data['close'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'spread': data['spread'].values,
            'atr': data['ATR'].values
        }
        
        self.raw_data = self._preprocess_data(data)
        self.current_step = 0
        self.random_start = random_start
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.previous_balance = initial_balance  # Add tracking for reward calculation
        
        # Trading state
        self.trades: List[Dict[str, Any]] = []
        self.positions = []
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        self.completed_episodes = 0
        self.episode_steps = 0
        # Dynamic trade cooldown based on volatility
        self.base_cooldown = 5
        self.trade_cooldown = self.base_cooldown
        self.max_positions = 5
        
        # Grid tracking
        self.active_grid = None
        self.grid_metrics = {
            'position_count': 0,
            'avg_profit_per_close': 0.0,
            'grid_efficiency': 0.0,
            'current_direction': 0
        }
        
        self._setup_action_space()
        self._setup_observation_space(10)  # Updated for new feature count

        # Verify required columns
        required_columns = ['close', 'high', 'low', 'spread', 'ATR', 'RSI', 'EMA_fast', 'EMA_slow']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data for the model with advanced features."""
        features_df = pd.DataFrame(index=self.original_index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Price data
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            opens = data['open'].values  # Using 'opens' since 'open' is a built-in function
            
            # Returns and volatility
            returns = np.diff(close) / close[:-1]
            returns = np.insert(returns, 0, 0)
            returns = np.clip(returns, -0.1, 0.1)
            
            vol = pd.Series(returns).rolling(20, min_periods=1).std()
            vol_normalized = (vol - vol.rolling(100, min_periods=1).min()) / \
                           (vol.rolling(100, min_periods=1).max() - vol.rolling(100, min_periods=1).min() + 1e-8)
            
            # Market Regime Detection (ADX-based)
            pdm = high[1:] - high[:-1]
            ndm = low[:-1] - low[1:]
            pdm = np.insert(np.where(pdm > 0, pdm, 0), 0, 0)
            ndm = np.insert(np.where(ndm > 0, ndm, 0), 0, 0)
            
            tr = np.maximum(high - low,
                          np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
            
            atr_period = 14
            atr = pd.Series(tr).rolling(atr_period).mean().values
            pdi = pd.Series(pdm).rolling(atr_period).mean() / atr * 100
            ndi = pd.Series(ndm).rolling(atr_period).mean() / atr * 100
            dx = np.abs(pdi - ndi) / (pdi + ndi) * 100
            adx = pd.Series(dx).rolling(atr_period).mean()
            
            # Volatility Breakout Signals
            boll_std = pd.Series(close).rolling(20).std()
            upper_band = pd.Series(close).rolling(20).mean() + (boll_std * 2)
            lower_band = pd.Series(close).rolling(20).mean() - (boll_std * 2)
            
            # Price Action Features
            body = close - opens
            upper_wick = high - np.maximum(close, opens)
            lower_wick = np.minimum(close, opens) - low
            
            # Moving Average Features
            fast_ma = data['EMA_fast'].values
            slow_ma = data['EMA_slow'].values
            ma_diff = (fast_ma - slow_ma) / slow_ma
            ma_diff_norm = np.clip(ma_diff * 10, -1, 1)
            
            # Store normalized features
            features_df['returns'] = returns
            features_df['volatility'] = vol_normalized
            features_df['trend'] = ma_diff_norm
            features_df['rsi'] = data['RSI'].values / 50 - 1  # Normalize to [-1, 1]
            features_df['atr'] = 2 * (atr / close - np.min(atr / close)) / \
                                (np.max(atr / close) - np.min(atr / close) + 1e-8) - 1
            
            # New Features
            features_df['adx_trend'] = adx / 100  # Normalize to [0,1]
            features_df['volatility_breakout'] = (close - lower_band) / (upper_band - lower_band + 1e-8)
            features_df['body_to_range'] = body / (high - low + 1e-8)
            features_df['wick_ratio'] = (upper_wick - lower_wick) / (upper_wick + lower_wick + 1e-8)
            features_df['trend_strength'] = np.clip(adx/25 - 1, -1, 1)  # Normalized trend strength
        
        return features_df.fillna(0)

    def _setup_action_space(self) -> None:
        """Configure discrete action space."""
        self.action_space = spaces.Discrete(3)

    def _setup_observation_space(self, feature_count: int) -> None:
        """Setup observation space with proper feature bounds."""
        # All features are normalized between -1 and 1:
        # Base features:
        # - returns: [-0.1, 0.1]
        # - volatility: [-1, 1]
        # - trend: [-1, 1]
        # - rsi: [-1, 1]
        # - atr: [-1, 1]
        # New features:
        # - adx_trend: [0, 1]
        # - volatility_breakout: [0, 1]
        # - body_to_range: [-1, 1]
        # - wick_ratio: [-1, 1]
        # - trend_strength: [-1, 1]
        feature_count = 10  # Update to match total number of features
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(feature_count,), dtype=np.float32
        )

    def _process_action(self, action: Union[int, np.ndarray]) -> int:
        """Convert discrete action to direction.
        
        Args:
            action: Integer action from policy (0: hold, 1: long, 2: short)
            
        Returns:
            int: Converted direction (-1: short, 0: hold, 1: long)
        """
        # Handle array input from policy
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Ensure action is within valid range
        action = int(action) % 3
            
        # Map action to direction
        direction_map = {
            0: 0,   # Hold
            1: 1,   # Long
            2: -1   # Short
        }
        
        return direction_map[action]

    def _execute_grid_trade(self, direction: int, raw_spread: float) -> float:
        """Execute a trade with dynamic grid sizing."""
        # Update trade cooldown based on volatility
        current_atr = self.prices['atr'][self.current_step]
        vol_scale = current_atr / self.prices['close'][self.current_step]
        self.trade_cooldown = max(2, int(self.base_cooldown * (1.0 - vol_scale * 5)))
        
        # Check cooldown and return penalty if needed
        if direction == 0:
            self.steps_since_trade += 1
            return 0.0
        elif self.steps_since_trade < self.trade_cooldown:
            self.steps_since_trade += 1
            return -0.1 * (1.0 + len(self.positions) / self.max_positions)  # Higher penalty with more positions
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        
        if self.active_grid and direction != self.active_grid.direction:
            self._close_grid()
            return -0.2
        
        if not self.active_grid:
            self.active_grid = Grid(direction, current_price, current_atr)
            self.grid_metrics['current_direction'] = direction
        
        self.active_grid.update_metrics(current_price, current_atr)
            
        if not self.active_grid.should_add_position(current_price):
            return -0.1
            
        # Calculate dynamic lot sizing based on multiple factors
        equity_ratio = self.balance / self.initial_balance
        risk_factor = max(0.5, min(1.5, equity_ratio))  # Scale risk based on equity growth
        
        # Base lot size calculation
        base_lot = max(self.MIN_LOTS, round(self.balance / self.BALANCE_PER_LOT / 100, 2))
        
        # Volatility scaling (reduce size in high volatility)
        volatility_ratio = current_atr / self.active_grid.initial_atr
        volatility_scale = 1.0 / max(1.0, volatility_ratio)  # Inverse relationship
        
        # Position scaling (reduce subsequent position sizes)
        position_discount = max(0.5, 1.0 - (len(self.positions) * 0.15))
        
        # Trend alignment bonus
        trend = self.raw_data.values[self.current_step][2]  # Normalized trend feature
        trend_alignment = 1.0
        if (self.active_grid.direction == 1 and trend > 0.5) or \
           (self.active_grid.direction == -1 and trend < -0.5):
            trend_alignment = 1.2  # Increase size when trading with trend
        
        # Combine all scaling factors
        lot_size = min(
            self.MAX_LOTS,
            base_lot * risk_factor * volatility_scale * position_discount * trend_alignment
        )
        
        position = {
            "direction": direction,
            "entry_price": current_price + (raw_spread if direction == 1 else -raw_spread),
            "lot_size": lot_size,
            "entry_time": str(self.original_index[self.current_step]),
            "entry_step": self.current_step,
            "entry_atr": current_atr,
            "grid_size": self.active_grid.grid_size,
            "current_profit_pips": 0.0
        }
        
        self.positions.append(position)
        self.active_grid.positions.append(position)
        self.grid_metrics['position_count'] = len(self.positions)
        
        self.active_grid.update_metrics(current_price, current_atr)
        self.steps_since_trade = 0
        
        position_bonus = min(1.0, len(self.positions) / 5.0) * 0.3
        volatility_bonus = min(1.0, current_atr / self.active_grid.initial_atr) * 0.2
        direction_bonus = 0.2 if len(self.positions) == 1 else 0.1
        
        return position_bonus + volatility_bonus + direction_bonus
    
    def _close_grid(self) -> float:
        """Close all positions in current grid."""
        if not self.active_grid:
            return 0.0
        
        current_price = self.prices['close'][self.current_step]
        total_pnl = 0.0
        
        for pos in self.positions:
            if self.active_grid.direction == 1:
                profit_points = current_price - pos["entry_price"]
            else:
                profit_points = pos["entry_price"] - current_price
            
            pos["exit_price"] = current_price
            pos["exit_step"] = self.current_step
            pos["exit_time"] = str(self.original_index[self.current_step])
            pos["profit_pips"] = profit_points / 0.0001
            pos["pnl"] = profit_points * pos["lot_size"]
            pos["hold_time"] = pos["exit_step"] - pos["entry_step"]
            total_pnl += pos["pnl"]
            
            if pos["pnl"] > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            self.trades.append(pos)
            
        # Update balance and clear positions
        self.balance += total_pnl
        self.positions.clear()
        self.active_grid = None
        self.grid_metrics['position_count'] = 0
        self.grid_metrics['current_direction'] = 0
        
        return total_pnl

    def _manage_grid_positions(self) -> Tuple[float, List[Dict[str, Any]]]:
        """Manage grid positions."""
        if not self.active_grid:
            return 0.0, []
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        total_pnl = 0.0
        
        for pos in self.positions:
            if self.active_grid.direction == 1:
                profit_points = current_price - pos["entry_price"]
            else:
                profit_points = pos["entry_price"] - current_price
                
            pos["current_profit_pips"] = profit_points / 0.0001
            pos_pnl = profit_points * pos["lot_size"]
            total_pnl += pos_pnl
            
        self.active_grid.total_pnl = total_pnl
        self.active_grid.update_metrics(current_price, current_atr)
        
        # Calculate dynamic take profit and stop loss based on volatility and positions
        current_atr = self.prices['atr'][self.current_step]
        current_price = self.prices['close'][self.current_step]
        volatility_factor = current_atr / current_price
        position_factor = 1.0 + (len(self.positions) * 0.2)
        
        # Grid value represents total risk
        grid_value = self.active_grid.grid_size * self.active_grid.total_lots
        
        # Scale take profit and stop loss based on market conditions
        base_tp_factor = 1.0 + (0.1 * len(self.positions))  # Increased TP with more positions
        base_sl_factor = 1.5 - (0.1 * len(self.positions))  # Tighter SL with more positions
        
        # Adjust factors based on volatility
        volatility_scale = np.clip(volatility_factor * 20, 0.5, 2.0)
        take_profit = grid_value * base_tp_factor * volatility_scale
        stop_loss = -grid_value * base_sl_factor / volatility_scale  # Tighter SL in high volatility
        
        # Consider trend for exit thresholds
        trend = self.raw_data.values[self.current_step][2]  # Trend feature from normalized data
        if (self.active_grid.direction == 1 and trend < -0.5) or \
           (self.active_grid.direction == -1 and trend > 0.5):
            # Tighten exits when trading against trend
            take_profit *= 0.8
            stop_loss *= 1.2
        
        should_close = (
            total_pnl > take_profit or
            (total_pnl < stop_loss and len(self.positions) >= 2) or
            (abs(total_pnl) < grid_value * 0.1 and len(self.positions) >= 4)  # Close small PnL with many positions
        )
        
        if should_close:
            reward = self._close_grid()
            return reward, []
            
        if len(self.positions) < self.max_positions:
            current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
            if self.active_grid.should_add_position(current_price):
                self._execute_grid_trade(self.active_grid.direction, current_spread)
                
        return total_pnl, []  # Return raw PnL, let calculate_reward handle reward calculation

    def _check_grid_spacing(self) -> float:
        """Verify grid positions are properly spaced.
        
        Returns:
            float: 1.0 for good spacing, less for poor spacing
        """
        if not self.active_grid or len(self.active_grid.positions) < 2:
            return 1.0
            
        # Sort positions by entry price
        positions = sorted(
            self.active_grid.positions,
            key=lambda x: x["entry_price"]
        )
        
        # Calculate average and minimum grid spacing
        spacings = []
        for i in range(1, len(positions)):
            spacing = abs(positions[i]["entry_price"] - positions[i-1]["entry_price"])
            spacings.append(spacing)
            
        avg_spacing = sum(spacings) / len(spacings)
        target_spacing = self.active_grid.grid_size
        
        # Score based on how close to target spacing
        spacing_score = min(1.0, target_spacing / max(avg_spacing, target_spacing * 0.1))
        return spacing_score
        
    def calculate_reward(self) -> float:
        """Calculate the core reward based on ROE and grid quality."""
        # Primary reward based on return on equity
        roe = (self.balance - self.previous_balance) / self.initial_balance
        base_reward = roe * 100  # Scale up for learning
        
        # Risk adjustment based on drawdown
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        risk_multiplier = max(0.1, 1.0 - (current_drawdown * 2))  # Floor at 0.1
        
        # Grid quality metrics
        if self.active_grid:
            grid_positions = len(self.active_grid.positions)
            grid_spacing = self._check_grid_spacing()
            grid_multiplier = min(1.0 + (grid_positions * 0.2), 2.0) * grid_spacing
        else:
            grid_multiplier = 1.0
            
        return base_reward * risk_multiplier * grid_multiplier
        
    def get_action_penalty(self) -> float:
        """Calculate penalties for undesirable actions."""
        if self.steps_since_trade < self.trade_cooldown:
            return -0.1  # Penalty for trading too frequently
            
        if self.active_grid and len(self.active_grid.positions) >= self.max_positions:
            return -0.2  # Penalty for overextending grid
            
        return 0.0
        
    def get_terminal_reward(self) -> float:
        """Calculate terminal state rewards/penalties."""
        if self.balance <= 0:
            return -10.0  # Strong penalty for bankruptcy
            
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        if max_drawdown >= self.MAX_DRAWDOWN:
            return -5.0  # Penalty for excessive drawdown
            
        # Positive terminal reward only if profitable
        if self.balance > self.initial_balance:
            return (self.balance / self.initial_balance - 1) * 5.0
            
        return 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an environment step."""
        direction = self._process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.episode_steps += 1
        self.current_step += 1
        
        # Calculate terminal conditions
        end_of_data = (self.current_step >= len(self.raw_data) - 1)
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        done = end_of_data

        # Store previous balance for reward calculation
        self.previous_balance = previous_balance
        
        # Execute grid management and trade actions
        self._manage_grid_positions()
        if direction != 0:
            self._execute_grid_trade(direction, current_spread)
            
        # Calculate core rewards
        reward = self.calculate_reward()      # Core reward based on ROE and grid quality
        reward += self.get_action_penalty()   # Add any action penalties
        
        # Check terminal conditions
        if self.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN:
            self._close_grid()
            reward = self.get_terminal_reward()
            done = True
        elif end_of_data:  # End of episode but not bankruptcy/max drawdown
            reward += self.get_terminal_reward()
        
        self.grid_metrics.update({
            'position_count': len(self.positions),
            'avg_profit_per_close': (sum(t["pnl"] for t in self.trades) / len(self.trades)) if self.trades else 0.0,
            'grid_efficiency': (self.balance - self.initial_balance) / self.initial_balance * 100
        })
        
        obs = self.get_history()
        self.reward = reward
        
        truncated = self.current_step >= len(self.raw_data) - 1
        return obs, reward, done, truncated, {
            "balance": self.balance,
            "total_pnl": self.balance - self.initial_balance,
            "drawdown": max_drawdown * 100,
            "grid_metrics": self.grid_metrics
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        if self.random_start:
            self.current_step = np.random.randint(0, len(self.raw_data) - 1)
        else:
            self.current_step = 0
            
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.previous_balance = self.initial_balance  # Reset previous balance tracking
        
        self.trades.clear()
        self.positions.clear()
        
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.episode_steps = 0
        
        self.active_grid = None
        self.grid_metrics.update({
            'position_count': 0,
            'avg_profit_per_close': 0.0,
            'grid_efficiency': 0.0,
            'current_direction': 0
        })
        
        self.completed_episodes += 1
        
        return self.get_history(), {
            "balance": self.balance,
            "positions": len(self.positions)
        }
        
    def get_history(self) -> np.ndarray:
        """Get current bar features."""
        return self.raw_data.values[self.current_step]

    def render(self) -> None:
        """Print environment state and trade statistics."""
        print(f"\n===== Episode {self.completed_episodes}, Step {self.episode_steps} =====")
        print(f"Current Balance: {self.balance:.2f}")
        print(f"Grid Positions: {len(self.positions)}")
        
        if len(self.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        long_trades = trades_df[trades_df["direction"] == 1]
        short_trades = trades_df[trades_df["direction"] == -1]
        long_wins = long_trades[long_trades["pnl"] > 0]
        short_wins = short_trades[short_trades["pnl"] > 0]
        
        avg_hold_time = trades_df["hold_time"].mean()
        avg_win_hold = winning_trades["hold_time"].mean()
        avg_loss_hold = losing_trades["hold_time"].mean()
        
        print("\n===== Performance Metrics =====")
        print(f"Total Return: {((self.balance - self.initial_balance) / self.initial_balance * 100):.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Overall Win Rate: {(len(winning_trades) / len(self.trades) * 100):.2f}%")
        print(f"Average Win: {winning_trades['pnl'].mean():.2f}")
        print(f"Average Loss: {losing_trades['pnl'].mean():.2f}")
        print(f"Sharpe Ratio: {(winning_trades['pnl'].mean() / winning_trades['pnl'].std() * np.sqrt(252)):.2f}")
        print(f"Current Drawdown: {((self.max_balance - self.balance) / self.max_balance * 100):.2f}%")
        
        print("\n===== Hold Time Analysis =====")
        print(f"Average Hold Time: {avg_hold_time:.1f} bars")
        print(f"Winners Hold Time: {avg_win_hold:.1f} bars")
        print(f"Losers Hold Time: {avg_loss_hold:.1f} bars")
        
        print("\n===== Directional Performance =====")
        total_trades = len(trades_df)
        long_pct = (len(long_trades) / total_trades * 100) if total_trades > 0 else 0.0
        short_pct = (len(short_trades) / total_trades * 100) if total_trades > 0 else 0.0
        
        print(f"Long Trades: {len(long_trades)} ({long_pct:.1f}%)")
        print(f"Long Win Rate: {(len(long_wins) / len(long_trades) * 100):.1f}% (Avg PnL: {long_trades['pnl'].mean():.2f})" if len(long_trades) > 0 else "Long Win Rate: N/A")
        print(f"Short Trades: {len(short_trades)} ({short_pct:.1f}%)")
        print(f"Short Win Rate: {(len(short_wins) / len(short_trades) * 100):.1f}% (Avg PnL: {short_trades['pnl'].mean():.2f})" if len(short_trades) > 0 else "Short Win Rate: N/A")
        
        print("\n===== Grid Stats =====")
        if self.active_grid:
            direction = "Long" if self.active_grid.direction == 1 else "Short"
            print(f"Active {direction} Grid:")
            print(f"  Positions: {len(self.active_grid.positions)}")
            print(f"  Average Entry: {self.active_grid.avg_entry:.2f}")
            print(f"  Grid Size: {self.active_grid.grid_size:.2f}")
            print(f"  Total PnL: {self.active_grid.total_pnl:.2f}")
