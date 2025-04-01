"""
Trading environment for grid-based trading with PPO-LSTM.

This module implements a custom OpenAI Gym environment for training
a PPO-LSTM model to trade using a dynamic grid strategy. The environment uses:
- Discrete action space (Hold/Buy/Sell)
- Single timestep observations with 5 features:
  * Returns
  * Volatility
  * Trend
  * RSI
  * ATR
- Temporal dependencies handled by LSTM in model
- Dynamic grid sizing based on ATR
"""

import gymnasium
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces

import warnings
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
        """Determine if we should add a new position."""
        if not self.positions:
            return True
            
        if self.direction == 1:  # Long grid
            return current_price < self.avg_entry - self.grid_size
        else:  # Short grid
            return current_price > self.avg_entry + self.grid_size
            
    def calculate_next_lot_size(self, base_lot: float, max_lots: float) -> float:
        """Calculate the lot size for the next position."""
        position_multiplier = 1 + len(self.positions) * 0.5
        return min(max_lots, base_lot * position_multiplier)

class TradingEnv(gym.Env, EzPickle):
    """Trading environment for grid-based trading with PPO-LSTM."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 balance_per_lot: float = 1000.0, random_start: bool = False):
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
        self.BALANCE_PER_LOT = balance_per_lot  # Amount in balance required per 0.01 lot
        self.MAX_DRAWDOWN = 0.5     # Maximum allowed drawdown (50%)
        
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
        
        # Trading state
        self.trades: List[Dict[str, Any]] = []
        self.positions = []      # List of positions in current grid
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        self.completed_episodes = 0
        self.episode_steps = 0
        self.trade_cooldown = 5  # Minimum bars between trades
        
        # Grid tracking
        self.active_grid = None  # Single grid that can be long or short
        self.grid_metrics = {
            'position_count': 0,
            'avg_profit_per_close': 0.0,
            'grid_efficiency': 0.0,
            'current_direction': 0  # 0: None, 1: Long, -1: Short
        }
        
        # Episode settings
        self.max_steps = 500  # Default episode length
        
        self._setup_action_space()
        self._setup_observation_space(5)  # 5 features for state space

        # Verify required columns are present
        required_columns = ['close', 'high', 'low', 'spread', 'ATR', 'RSI', 'EMA_fast', 'EMA_slow']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data for the model with simplified features."""
        features_df = pd.DataFrame(index=self.original_index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            close = data['close'].values
            
            # Calculate returns and clip to remove outliers
            returns = np.diff(close) / close[:-1]
            returns = np.insert(returns, 0, 0)
            returns = np.clip(returns, -0.1, 0.1)
            
            # Calculate volatility (10-period rolling standard deviation)
            vol = pd.Series(returns).rolling(10, min_periods=1).std().fillna(0).values
            vol = np.clip(vol, 1e-8, 0.1)
            
            # Calculate trend based on EMAs
            trend = np.where(data['EMA_fast'].values > data['EMA_slow'].values, 1, -1)
            
            # Normalize RSI and ATR
            rsi_norm = data['RSI'].values / 100
            atr_norm = data['ATR'].values / close
            atr_norm = np.clip(atr_norm, 0, 0.1)
            
            # Combine features
            features_df['returns'] = returns
            features_df['volatility'] = vol
            features_df['trend'] = trend
            features_df['rsi'] = rsi_norm
            features_df['atr'] = atr_norm
            
        return features_df.fillna(0)

    def _setup_action_space(self) -> None:
        """Configure discrete action space for direction only."""
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def _setup_observation_space(self, feature_count: int) -> None:
        """Setup observation space for single bar features."""
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(feature_count,), dtype=np.float32
        )

    def _process_action(self, action: Union[int, np.ndarray]) -> int:
        """Convert discrete action to direction."""
        # Convert numpy array to integer if needed
        if isinstance(action, np.ndarray):
            action = action.item()
            
        direction_map = {
            0: 0,     # Hold
            1: 1,     # Buy
            2: -1     # Sell
        }
        return direction_map[int(action)]

    def _execute_grid_trade(self, direction: int, raw_spread: float) -> float:
        """Execute a trade with dynamic grid sizing."""
        if direction == 0 or self.steps_since_trade < self.trade_cooldown:
            self.steps_since_trade += 1
            return -0.1 if direction != 0 else 0.0
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        
        # Check if we need to close existing grid in opposite direction
        if self.active_grid and direction != self.active_grid.direction:
            # Close existing grid and get reward
            self._close_grid()
            return -0.2  # Penalty for changing direction
        
        # Create new grid if needed
        if not self.active_grid:
            self.active_grid = Grid(direction, current_price, current_atr)
            self.grid_metrics['current_direction'] = direction
        
        # Update grid metrics
        self.active_grid.update_metrics(current_price, current_atr)
            
        # Check if we should add position
        if not self.active_grid.should_add_position(current_price):
            return -0.1  # Penalty for trying to add position too soon
            
        # Calculate position size based on volatility and grid utilization
        base_lot = max(self.MIN_LOTS, round(self.balance / self.BALANCE_PER_LOT / 100, 2))
        volatility_scale = min(current_atr / self.active_grid.initial_atr, 2.0)
        position_scale = 1.0 + (len(self.positions) * 0.2)  # Scale up with grid size
        lot_size = min(self.MAX_LOTS, base_lot * volatility_scale * position_scale)
        
        # Create new position
        position = {
            "direction": direction,
            "entry_price": current_price + (raw_spread if direction == 1 else -raw_spread),
            "lot_size": lot_size,
            "entry_step": self.current_step,
            "entry_atr": current_atr,
            "grid_size": self.active_grid.grid_size,
            "current_profit_pips": 0.0
        }
        
        # Add position to tracking
        self.positions.append(position)
        self.active_grid.positions.append(position)
        self.grid_metrics['position_count'] = len(self.positions)
        
        self.active_grid.update_metrics(current_price, current_atr)
        self.steps_since_trade = 0
        
        # Calculate reward based on grid utilization and volatility
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
            if self.active_grid.direction == 1:  # Long positions
                profit_points = current_price - pos["entry_price"]
            else:  # Short positions
                profit_points = pos["entry_price"] - current_price
            
            pos["exit_price"] = current_price
            pos["exit_step"] = self.current_step
            pos["exit_time"] = str(self.original_index[self.current_step])
            pos["profit_pips"] = profit_points / 0.0001
            pos["pnl"] = profit_points * pos["lot_size"]
            total_pnl += pos["pnl"]
            
            # Update trading stats
            if pos["pnl"] > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            self.trades.append(pos)
            
        self.balance += total_pnl
        self.positions.clear()
        
        # Calculate final grid reward
        reward = self._calculate_grid_reward(self.active_grid.positions, total_pnl)
        
        # Reset grid
        self.active_grid = None
        self.grid_metrics['position_count'] = 0
        self.grid_metrics['current_direction'] = 0
        
        return reward

    def _manage_grid_positions(self) -> Tuple[float, List[Dict[str, Any]]]:
        """Manage grid positions including entry, exit, and pyramiding decisions."""
        if not self.active_grid:
            return 0.0, []
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        total_pnl = 0.0
        
        # Update positions and calculate PnL
        for pos in self.positions:
            if self.active_grid.direction == 1:  # Long positions
                profit_points = current_price - pos["entry_price"]
            else:  # Short positions
                profit_points = pos["entry_price"] - current_price
                
            pos["current_profit_pips"] = profit_points / 0.0001
            pos_pnl = profit_points * pos["lot_size"]
            total_pnl += pos_pnl
            
        self.active_grid.total_pnl = total_pnl
        self.active_grid.update_metrics(current_price, current_atr)
        
        # Dynamic closure conditions based on current grid size
        grid_value = self.active_grid.grid_size * self.active_grid.total_lots
        take_profit = grid_value * (1.0 + 0.1 * len(self.positions))
        stop_loss = -grid_value * (1.5 - 0.1 * len(self.positions))
        
        should_close = (
            total_pnl > take_profit or       # Take profit
            (total_pnl < stop_loss and       # Stop loss
             len(self.positions) >= 3)       # After pyramiding
        )
        
        if should_close:
            reward = self._close_grid()
            return reward, []
            
        # Check for grid expansion
        if len(self.positions) < 5:  # Max 5 positions per grid
            current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
            if self.active_grid.should_add_position(current_price):
                self._execute_grid_trade(self.active_grid.direction, current_spread)
                
        return self._calculate_grid_reward(self.positions, total_pnl), []

    def _calculate_grid_reward(self, positions: List[Dict[str, Any]], total_pnl: float) -> float:
        """Calculate reward for grid trading performance."""
        if not positions:
            return 0.0
            
        # Calculate grid efficiency
        grid_value = positions[0]["grid_size"] * sum(p["lot_size"] for p in positions)
        efficiency = min(2.0, abs(total_pnl / grid_value)) if grid_value > 0 else 0.0
        
        # Risk-adjusted PnL
        risk_adjusted_pnl = (total_pnl / self.initial_balance) * len(positions)
        
        # Grid utilization bonus
        utilization = len(positions) / 5.0  # Maximum 5 positions per grid
        
        # Calculate reward components
        pnl_reward = risk_adjusted_pnl * 0.6
        efficiency_reward = efficiency * 0.2
        utilization_bonus = utilization * 0.2
        
        return pnl_reward + efficiency_reward + utilization_bonus

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an environment step."""
        direction = self._process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.episode_steps += 1
        self.current_step += 1
        
        done = (self.episode_steps >= self.max_steps) or (self.current_step >= len(self.raw_data) - 1)

        # Process existing positions and get grid trading reward
        grid_reward, _ = self._manage_grid_positions()
        
        # Execute new trade if requested
        execution_reward = 0.0
        if direction != 0:
            execution_reward = self._execute_grid_trade(direction, current_spread)
        
        # Calculate growth and drawdown components
        growth_reward = np.log(self.balance / previous_balance) * 5.0 if self.balance > previous_balance else 0.0
        drawdown = (self.max_balance - self.balance) / self.max_balance
        drawdown_penalty = drawdown * 3.0 if drawdown > 0.1 else 0.0
        
        # Combine rewards with grid-focused weights
        reward = (
            (grid_reward * 0.6) +           # Grid management
            (execution_reward * 0.2) +      # Trade execution
            (growth_reward * 0.2) -         # Account growth
            drawdown_penalty                # Risk management
        )
            
        # Check for bankruptcy or excessive drawdown
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        if self.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN:
            self._close_grid()  # Close any active positions
            reward = -100 if self.balance <= 0 else -50
            done = True
            
        # Episode completion rewards
        if done and self.balance > self.initial_balance:
            final_return = (self.balance / self.initial_balance) - 1
            win_rate = (self.win_count / (self.win_count + self.loss_count)) if (self.win_count + self.loss_count) > 0 else 0
            reward += final_return * 10.0 + win_rate * 10.0
        
        # Update grid metrics
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

        min_episode_length = 500
        max_episode_length = min(5000, len(self.raw_data))
        
        scale_factor = min(self.completed_episodes / 25, 1.0)
        self.max_steps = int(min_episode_length + (max_episode_length - min_episode_length) * scale_factor)
        
        if self.random_start:
            latest_start = len(self.raw_data) - self.max_steps - 1
            self.current_step = np.random.randint(0, max(1, latest_start))
        else:
            self.current_step = 0
            
        # Reset account state
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        
        # Reset all trading state
        self.trades.clear()
        self.positions.clear()
        
        # Reset trading counters
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.episode_steps = 0
        
        # Reset grid tracking
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
        print(f"\n===== Episode {self.completed_episodes}, Step {self.episode_steps}/{self.max_steps} =====")
        print(f"Current Balance: {self.balance:.2f}")
        print(f"Grid Positions: {len(self.positions)}")
        
        if len(self.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate basic metrics
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        # Calculate directional metrics
        long_trades = trades_df[trades_df["direction"] == 1]
        short_trades = trades_df[trades_df["direction"] == -1]
        long_wins = long_trades[long_trades["pnl"] > 0]
        short_wins = short_trades[short_trades["pnl"] > 0]
        
        # Calculate hold times
        trades_df["hold_time"] = trades_df["exit_step"] - trades_df["entry_step"]
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
        print(f"Long Trades: {len(long_trades)} ({(len(long_trades) / len(trades_df) * 100):.1f}%)")
        print(f"Long Win Rate: {(len(long_wins) / len(long_trades) * 100):.1f}% (Avg PnL: {long_trades['pnl'].mean():.2f})")
        print(f"Short Trades: {len(short_trades)} ({(len(short_trades) / len(trades_df) * 100):.1f}%)")
        print(f"Short Win Rate: {(len(short_wins) / len(short_trades) * 100):.1f}% (Avg PnL: {short_trades['pnl'].mean():.2f})")
        
        print("\n===== Grid Stats =====")
        if self.active_grid:
            direction = "Long" if self.active_grid.direction == 1 else "Short"
            print(f"Active {direction} Grid:")
            print(f"  Positions: {len(self.active_grid.positions)}")
            print(f"  Average Entry: {self.active_grid.avg_entry:.2f}")
            print(f"  Grid Size: {self.active_grid.grid_size:.2f}")
            print(f"  Total PnL: {self.active_grid.total_pnl:.2f}")
