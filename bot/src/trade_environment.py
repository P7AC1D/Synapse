"""
Trading environment for single-position trading with PPO-LSTM.

This module implements a custom OpenAI Gym environment for training
a PPO-LSTM model to trade using a single position strategy.
"""

import gymnasium
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces
import gymnasium as gym
from gymnasium.utils import EzPickle
from enum import IntEnum

class Action(IntEnum):
    """Trading actions enumeration."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

class TradingEnv(gym.Env, EzPickle):
    """Trading environment for single-position trading with PPO-LSTM."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 balance_per_lot: float = 1000.0, random_start: bool = False,
                 bar_count: int = 10, max_hold_bars: int = 64):  # bar_count is deprecated
        super().__init__()
        EzPickle.__init__(self)
        
        self.MAX_HOLD_BARS = max_hold_bars  # Maximum bars to hold a position
        
        # Save original datetime index
        self.original_index = data.index.copy() if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index)
        
        # Trading constants adjusted for XAUUSD - UPDATED VALUES
        self.POINT_VALUE = 0.01      # Gold moves in 0.01 increments
        self.PIP_VALUE = 0.01        # Gold pip and point values are the same
        self.MIN_LOTS = 0.01         # Minimum 0.01 lots (standard for gold)
        self.MAX_LOTS = 50.0         # Reduced max lots due to gold's higher pip value
        self.CONTRACT_SIZE = 100.0   # Standard gold contract = 100 oz
        self.BALANCE_PER_LOT = balance_per_lot  # Will be higher for gold
        self.MAX_DRAWDOWN = 0.4      # More conservative drawdown for gold
        
        # Verify required columns
        required_columns = ['open', 'close', 'high', 'low', 'spread']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add volume if not present
        if 'volume' not in data.columns:
            print("Warning: 'volume' column not found, using synthetic volume data")
            data['volume'] = np.ones(len(data))
        
        # Preprocess data and calculate all technical indicators
        self.raw_data, atr_values = self._preprocess_data(data)
        
        # Store data length after preprocessing for consistent indexing
        self.data_length = len(self.raw_data)
        
        # Store price data matching preprocessed data length
        self.prices = {
            'close': data.loc[self.original_index, 'close'].values,
            'high': data.loc[self.original_index, 'high'].values,
            'low': data.loc[self.original_index, 'low'].values,
            'spread': data.loc[self.original_index, 'spread'].values,
            'atr': atr_values
        }
        
        self.current_step = 0
        self.random_start = random_start
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.previous_balance = initial_balance  # Add tracking for reward calculation
        
        # Trading state
        self.trades: List[Dict[str, Any]] = []
        self.current_position = None
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        self.completed_episodes = 0
        self.episode_steps = 0
        
        # Trade metrics
        self.trade_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'current_direction': 0
        }
        
        self._setup_action_space()
        self._setup_observation_space(10)  # Keep the same observation space


    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data for the model with advanced features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features_df, atr_values)
        """
        # Create DataFrame with same index as input data
        features_df = pd.DataFrame(index=data.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Price data
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            opens = data['open'].values
            
            # ===== CALCULATE TECHNICAL INDICATORS =====
            
            # Calculate ATR
            tr = np.maximum(high - low,
                          np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]  # Fix first value
            
            atr_period = 14
            atr = pd.Series(tr).rolling(atr_period).mean().values
            
            # Calculate RSI
            delta = pd.Series(close).diff().fillna(0).values
            gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(window=14).mean().values
            loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(window=14).mean().values
            
            # Avoid division by zero
            rs = np.zeros_like(gain)
            mask = loss != 0
            rs[mask] = gain[mask] / loss[mask]
            rsi = 100 - (100 / (1 + rs))            
            
            # Time encoding
            minutes_in_day = 24 * 60
            bar_times = pd.to_datetime(data.index)
            time_index = bar_times.hour * 60 + bar_times.minute
            sin_time = np.sin(2 * np.pi * time_index / minutes_in_day)
            cos_time = np.cos(2 * np.pi * time_index / minutes_in_day)
            
            # Returns Calculation
            returns = np.diff(close) / close[:-1]
            returns = np.insert(returns, 0, 0)
            returns = np.clip(returns, -0.1, 0.1)
            
            # Calculate Trend Strength (ADX-based)
            pdm = np.maximum(high[1:] - high[:-1], 0)  # Positive directional movement
            ndm = np.maximum(low[:-1] - low[1:], 0)    # Negative directional movement
            pdm = np.insert(pdm, 0, 0)  # Add 0 at start
            ndm = np.insert(ndm, 0, 0)  # Add 0 at start
            
            # Smooth DM values with fillna to handle NaN values
            pdm_smooth = pd.Series(pdm).rolling(atr_period, min_periods=1).mean().fillna(0)
            ndm_smooth = pd.Series(ndm).rolling(atr_period, min_periods=1).mean().fillna(0)
            
            # Calculate directional indicators with safe values
            atr_safe = np.where(atr < 1e-8, 1e-8, atr)  # Prevent division by zero
            pdi = (pdm_smooth.values / atr_safe) * 100
            ndi = (ndm_smooth.values / atr_safe) * 100
            
            # Calculate DX and ADX with proper NaN handling
            sum_di = pdi + ndi
            sum_di = np.where(sum_di < 1e-8, 1e-8, sum_di)  # Prevent division by zero
            dx = np.abs(pdi - ndi) / sum_di * 100
            adx = pd.Series(dx).rolling(atr_period, min_periods=1).mean().fillna(0).values
            trend_strength = np.clip(adx/25 - 1, -1, 1)
            
            # Volatility Breakout using Bollinger Bands with improved NaN handling
            boll_std = pd.Series(close).rolling(20, min_periods=1).std().fillna(0).values
            ma20 = pd.Series(close).rolling(20, min_periods=1).mean().fillna(close[0]).values
            upper_band = ma20 + (boll_std * 2)
            lower_band = ma20 - (boll_std * 2)
            
            # Safer division with explicit NaN handling
            band_range = (upper_band - lower_band)
            band_range = np.where(band_range < 1e-8, 1e-8, band_range)  # Prevent division by zero
            
            position = close - lower_band
            volatility_breakout = np.divide(position, band_range, out=np.zeros_like(position), where=band_range!=0)
            volatility_breakout = np.clip(volatility_breakout, 0, 1)
            
            # Combined Price Action Signal
            body = close - opens
            upper_wick = high - np.maximum(close, opens)
            lower_wick = np.minimum(close, opens) - low
            range_ = high - low + 1e-8
            
            # Combine body_to_range and wick_ratio into one signal
            candle_pattern = (body/range_ + 
                           (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2
            candle_pattern = np.clip(candle_pattern, -1, 1)
            
            # Store optimized feature set
            features_df['returns'] = returns
            features_df['rsi'] = rsi / 50 - 1  # Normalize to [-1, 1]
            features_df['atr'] = 2 * (atr / close - np.nanmin(atr / close)) / \
                              (np.nanmax(atr / close) - np.nanmin(atr / close) + 1e-8) - 1
            features_df['volatility_breakout'] = volatility_breakout
            features_df['trend_strength'] = trend_strength
            features_df['candle_pattern'] = candle_pattern
            features_df['sin_time'] = sin_time  # Already in [-1, 1] range
            features_df['cos_time'] = cos_time  # Already in [-1, 1] range
            
            # Calculate lookback period based on the longest indicator window
            lookback = max(20, atr_period)  # Use max of Bollinger (20) and ATR period
            
            # Forward fill any NaN values in features
            features_df = features_df.dropna()
            
            # Only keep data after the lookback period to ensure all indicators are properly calculated
            features_df = features_df.iloc[lookback:]
            
            # Ensure we have enough data after preprocessing
            if len(features_df) < 100:
                raise ValueError(f"Insufficient data after preprocessing: {len(features_df)} bars. Need at least 100 bars.")
        
        # Update price data to match cleaned features
        valid_indices = features_df.index
        # Get integer positions of valid indices in original data
        valid_positions = data.index.get_indexer(valid_indices)
        
        # Use integer positions for numpy array indexing
        atr = atr[valid_positions]
        self.prices = {
            'close': data.loc[valid_indices, 'close'].values,
            'high': data.loc[valid_indices, 'high'].values,
            'low': data.loc[valid_indices, 'low'].values,
            'spread': data.loc[valid_indices, 'spread'].values,
            'atr': atr
        }
        self.original_index = valid_indices
        
        # Return both the features dataframe and the ATR values for position management
        return features_df, atr

    def _setup_action_space(self) -> None:
        """Configure discrete action space: 0=hold, 1=buy, 2=sell, 3=close."""
        self.action_space = spaces.Discrete(4)

    def _setup_observation_space(self, _: int = 11) -> None:
        """Setup observation space with proper feature bounds."""
        # Optimized feature set (11 features):
        # 1. returns [-0.1, 0.1] - Price momentum
        # 2. rsi [-1, 1] - Momentum oscillator
        # 3. atr [-1, 1] - Volatility indicator
        # 4. volatility_breakout [0, 1] - Trend with volatility context
        # 5. trend_strength [-1, 1] - ADX-based trend quality
        # 6. candle_pattern [-1, 1] - Combined price action signal
        # 7. sin_time [-1, 1] - Sine encoding of time of day
        # 8. cos_time [-1, 1] - Cosine encoding of time of day
        # 9. position_type [-1, 0, 1] - Current position (short/none/long)
        # 10. hold_time [0, 1] - Normalized position hold time
        # 11. unrealized_pnl [-1, 1] - Current position P&L
        feature_count = 11  # Added hold time and unrealized P&L
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(feature_count,), dtype=np.float32
        )

    def _process_action(self, action: Union[int, np.ndarray]) -> int:
        """Convert action to trading decision.
        
        Args:
            action: Integer action from policy (0: hold, 1: buy, 2: sell, 3: close)
            
        Returns:
            int: Processed action (0: hold, 1: buy, 2: sell, 3: close)
        """
        # Handle array input from policy
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Ensure action is within valid range
        action = int(action) % 4
        
        return action

    def _execute_trade(self, direction: int, raw_spread: float) -> float:
        """Execute a trade with the given direction.
        
        Args:
            direction: Direction of the trade (1: buy, 2: sell)
            raw_spread: Current spread to adjust entry price
            
        Returns:
            float: Always 0 (reward comes from P&L)
        """
        # Check if we already have a position
        if self.current_position is not None:
            return 0
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        
        # Calculate lot size based on account balance
        lot_size = max(
            self.MIN_LOTS, 
            min(
                self.MAX_LOTS,
                round(self.balance * self.MIN_LOTS / self.BALANCE_PER_LOT, 2)
            )
        )
        
        # Calculate adjusted entry price based on direction and spread
        if direction == 1:  # Long position
            # For long positions, we buy at the ask price (close + spread)
            adjusted_entry_price = current_price + raw_spread
        else:  # Short position
            # For short positions, we sell at the bid price (close)
            adjusted_entry_price = current_price

        self.current_position = {
            "direction": 1 if direction == 1 else -1,
            "entry_price": adjusted_entry_price,  # Use spread-adjusted price
            "entry_spread": raw_spread,
            "lot_size": lot_size,
            "entry_time": str(self.original_index[self.current_step]),
            "entry_step": self.current_step,
            "entry_atr": current_atr,
            "current_profit_pips": 0.0
        }
        
        self.trade_metrics['current_direction'] = self.current_position["direction"]
        
        return 0.05 # Small incentive to explore
    
    def _close_position(self) -> float:
        """Close current position and calculate P/L.
        
        Returns:
            float: Reward for closing the position
        """
        if not self.current_position:
            return -0.1  # Penalty for trying to close when no position exists
        
        current_price = self.prices['close'][self.current_step]
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        lot_size = self.current_position["lot_size"]
        entry_step = self.current_position["entry_step"]  # Store before clearing position
        
        # Get current spread for exit price adjustment
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
        
        # Calculate profit or loss with spread at exit
        if direction == 1:  # Long position
            # For long exits, we sell at bid price
            exit_price = current_price
            profit_points = exit_price - entry_price
        else:  # Short position
            # For short exits, we buy back at ask price
            exit_price = current_price + current_spread
            profit_points = entry_price - exit_price
            
        pnl = profit_points * lot_size * self.CONTRACT_SIZE  # Account for 100oz contract size
        profit_pips = profit_points / self.PIP_VALUE
        
        # Record trade details
        self.current_position.update({
            "exit_price": exit_price,
            "exit_spread": current_spread,
            "exit_step": self.current_step,
            "exit_time": str(self.original_index[self.current_step]),
            "profit_pips": profit_pips,
            "pnl": pnl,
            "hold_time": self.current_step - entry_step
        })
        
        # Update trade statistics
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        self.trades.append(self.current_position)
        
        # Update balance and check for bankruptcy
        self.balance += pnl
        if self.balance <= 0:
            self.balance = 0  # Prevent negative balance
            # Calculate reward before clearing position
            normalized_pnl = pnl / self.initial_balance * 100
            # Clear position before returning
            self.current_position = None
            self.trade_metrics['current_direction'] = 0
            return normalized_pnl * -0.5  # Reduced negative reward for bankruptcy
        
        # Calculate hold time before clearing position
        hold_time = self.current_step - entry_step
        
        # Clear position
        self.current_position = None
        self.trade_metrics['current_direction'] = 0
        
        # Update trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t["pnl"] > 0]
            losing_trades = [t for t in self.trades if t["pnl"] <= 0]
            
            self.trade_metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0.0
            self.trade_metrics['avg_profit'] = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
            self.trade_metrics['avg_loss'] = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Base reward on P/L relative to account size only
        normalized_pnl = pnl / self.initial_balance * 100
        return normalized_pnl

    def _manage_position(self) -> float:
        """Calculate current position's unrealized P/L.
        
        Returns:
            float: Current unrealized P/L
        """
        if not self.current_position:
            return 0.0
            
        current_price = self.prices['close'][self.current_step]
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        lot_size = self.current_position["lot_size"]
        
        # Get current spread for unrealized P&L calculation
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
        
        # Calculate unrealized P/L including spread impact
        if direction == 1:  # Long position
            current_exit_price = current_price - current_spread  # Worse exit price for longs
            profit_points = current_exit_price - entry_price
        else:  # Short position
            current_exit_price = current_price + current_spread  # Worse exit price for shorts
            profit_points = entry_price - current_exit_price
            
        unrealized_pnl = profit_points * lot_size * self.CONTRACT_SIZE  # Account for 100oz contract size
        profit_pips = profit_points / self.PIP_VALUE
        
        self.current_position["current_profit_pips"] = profit_pips
        
        return unrealized_pnl        
        
    def get_action_penalty(self) -> float:
        """Calculate penalties for undesirable actions."""
        return 0.0  # No penalties in simplified reward structure
        
    def get_terminal_reward(self) -> float:
        """Calculate terminal state rewards/penalties."""
        return 0.0  # No terminal rewards in simplified structure

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an environment step."""
        action = self._process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        # Store previous balance for reward calculation
        previous_balance = self.balance
        self.previous_balance = previous_balance
        
        # Update balance tracking
        self.max_balance = max(self.balance, self.max_balance)
        
        # Update timesteps
        self.episode_steps += 1
        self.current_step += 1
        
        reward = 0.0

        # Execute trade actions
        if action == 1:  # Buy
            reward += self._execute_trade(1, current_spread)
        elif action == 2:  # Sell
            reward += self._execute_trade(2, current_spread)
        elif action == 3:  # Close
            reward += self._close_position()
        elif action == 0:  # Hold
            # Add small penalty for holding too long to encourage decision-making
            if self.current_position:  # Only apply hold penalty if a position exists
                hold_time = self.current_step - self.current_position["entry_step"]
                if hold_time > self.MAX_HOLD_BARS / 2:
                    reward -= 0.01  # Small time decay penalty
            
        unrealized_pnl = self._manage_position()

        # Manage current position and check bankruptcy
        if self.current_position:
            potential_balance = self.balance + unrealized_pnl
            if potential_balance <= 0:
                # Force close position and get final observation
                self._close_position()
                final_obs = self.get_history()
                done = True
                truncated = False
                return final_obs, -1.0, done, truncated, {
                    "balance": self.balance,
                    "total_pnl": self.balance - self.initial_balance,
                    "drawdown": 100.0,  # Maximum drawdown
                    "position": {},
                    "trade_metrics": self.trade_metrics,
                    "total_trades": len(self.trades)
                }
        
        # Calculate terminal conditions
        end_of_data = (self.current_step >= self.data_length - 1)
        max_drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 1.0
        done = end_of_data or self.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN
        
        # Auto-close position at end of episode
        if done and self.current_position:
            self._close_position()
        
        # Get observation
        obs = self.get_history()
        
        truncated = self.current_step >= self.data_length - 1
        
        # Calculate ATR-adjusted reward
        current_atr = self.prices['atr'][self.current_step]
        hold_time = self.current_step - self.current_position["entry_step"] if self.current_position else 0
        position_type = self.current_position["direction"] if self.current_position else 0        
        
        self.reward = reward
        
        # Calculate position info for info dict
        position_info = {}
        if self.current_position:
            position_info = {
                "direction": "long" if self.current_position["direction"] == 1 else "short",
                "entry_price": self.current_position["entry_price"],
                "lot_size": self.current_position["lot_size"],
                "unrealized_pnl": unrealized_pnl,
                "profit_pips": self.current_position["current_profit_pips"],
                "hold_time": self.current_step - self.current_position["entry_step"]
            }
        
        return obs, float(reward), done, truncated, {
            "balance": self.balance,
            "total_pnl": self.balance - self.initial_balance,
            "drawdown": max_drawdown * 100,
            "position": position_info,
            "trade_metrics": self.trade_metrics,
            "total_trades": len(self.trades)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        if self.random_start:
            # Ensure we leave enough room for at least one full episode
            max_start = max(0, self.data_length - 100)  # Leave 100 steps minimum
            self.current_step = np.random.randint(0, max_start)
        else:
            self.current_step = 0
            
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.previous_balance = self.initial_balance  # Reset previous balance tracking
        
        self.trades.clear()
        self.current_position = None
        
        self.win_count = 0
        self.loss_count = 0
        self.episode_steps = 0
        
        self.trade_metrics.update({
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'current_direction': 0
        })
        
        self.completed_episodes += 1
        
        return self.get_history(), {
            "balance": self.balance,
            "position": None
        }
        
    def get_history(self) -> np.ndarray:
        """Get current bar features including position type, hold time and unrealized P&L."""
        features = self.raw_data.values[self.current_step]
        
        # Add position type (-1: short, 0: no position, 1: long)
        position_type = 0
        normalized_hold_time = 0.0
        
        if self.current_position:
            position_type = self.current_position["direction"]  # Will be -1 for short or 1 for long
            hold_time = self.current_step - self.current_position["entry_step"]
            normalized_hold_time = min(hold_time / self.MAX_HOLD_BARS, 1.0)
        
        # Calculate normalized unrealized P&L
        if self.current_position:
            unrealized_pnl = self._manage_position()
            # Normalize P&L relative to initial balance
            normalized_pnl = np.clip(unrealized_pnl / self.initial_balance, -1, 1)
        else:
            normalized_pnl = 0.0  # No position
        
        # Add position type, normalized hold time, and normalized P&L to features
        return np.append(features, [position_type, normalized_hold_time, normalized_pnl])

    def render(self) -> None:
        """Print environment state and trade statistics."""
        print(f"\n===== Episode {self.completed_episodes}, Step {self.episode_steps} =====")
        print(f"Current Balance: {self.balance:.2f}")
        print(f"Current Position: {'None' if not self.current_position else ('Long' if self.current_position['direction'] == 1 else 'Short')}")
        
        if self.current_position:
            current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
            current_price = self.prices['close'][self.current_step]
            
            if self.current_position["direction"] == 1:  # Long
                current_exit_price = current_price - current_spread  # Worse exit price for longs
                unrealized_pnl = (current_exit_price - self.current_position["entry_price"]) * self.current_position["lot_size"] * self.CONTRACT_SIZE
            else:  # Short
                current_exit_price = current_price + current_spread  # Worse exit price for shorts
                unrealized_pnl = (self.current_position["entry_price"] - current_exit_price) * self.current_position["lot_size"] * self.CONTRACT_SIZE
                
            print(f"Position Details:")
            print(f"  Entry Price: {self.current_position['entry_price']:.5f}")
            print(f"  Current Price: {current_price:.5f}")
            print(f"  Current Spread: {current_spread:.5f}")
            print(f"  Potential Exit Price: {current_exit_price:.5f}")
            print(f"  Lot Size: {self.current_position['lot_size']:.2f}")
            print(f"  Unrealized P/L: {unrealized_pnl:.2f}")
            print(f"  Hold Time: {self.current_step - self.current_position['entry_step']} bars")
        
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
        
        avg_hold_time = trades_df["hold_time"].mean() if "hold_time" in trades_df.columns else 0
        avg_win_hold = winning_trades["hold_time"].mean() if "hold_time" in winning_trades.columns else 0
        avg_loss_hold = losing_trades["hold_time"].mean() if "hold_time" in losing_trades.columns else 0
        
        print("\n===== Performance Metrics =====")
        print(f"Total Return: {((self.balance - self.initial_balance) / self.initial_balance * 100):.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Overall Win Rate: {(len(winning_trades) / len(self.trades) * 100):.2f}%")
        print(f"Average Win: {winning_trades['pnl'].mean():.2f}")
        print(f"Average Loss: {losing_trades['pnl'].mean():.2f}")
        print(f"Profit Factor: {abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()):.2f}" if losing_trades['pnl'].sum() != 0 else "Profit Factor: ∞")
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
