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

class TradingEnv(gym.Env, EzPickle):
    """Trading environment for single-position trading with PPO-LSTM."""
    
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
        
        # Store price data for easy access
        self.prices = {
            'close': data['close'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'spread': data['spread'].values,
            'atr': atr_values  # This now comes from _preprocess_data
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
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        self.completed_episodes = 0
        self.episode_steps = 0
        # Dynamic trade cooldown based on volatility
        self.base_cooldown = 5
        self.trade_cooldown = self.base_cooldown
        
        # Trade metrics
        self.trade_metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'current_direction': 0
        }
        
        self._setup_action_space()
        self._setup_observation_space(10)  # Keep the same observation space

        # Verify required columns
        required_columns = ['close', 'high', 'low', 'spread', 'ATR', 'RSI', 'EMA_fast', 'EMA_slow']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data for the model with advanced features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features_df, atr_values)
        """
        features_df = pd.DataFrame(index=self.original_index)
        
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
            
            # Calculate EMAs
            ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
            ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
            
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
            
            pdi = pd.Series(pdm).rolling(atr_period).mean() / (atr + 1e-8) * 100
            ndi = pd.Series(ndm).rolling(atr_period).mean() / (atr + 1e-8) * 100
            
            # Avoid division by zero
            dx = np.zeros_like(pdi)
            mask = (pdi + ndi) != 0
            dx[mask] = np.abs(pdi[mask] - ndi[mask]) / (pdi[mask] + ndi[mask]) * 100
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
            ma_diff = (ema_fast - ema_slow) / ema_slow
            ma_diff_norm = np.clip(ma_diff * 10, -1, 1)
            
            # Store normalized features
            features_df['returns'] = returns
            features_df['volatility'] = vol_normalized
            features_df['trend'] = ma_diff_norm
            features_df['rsi'] = rsi / 50 - 1  # Normalize to [-1, 1]
            features_df['atr'] = 2 * (atr / close - np.nanmin(atr / close)) / \
                                (np.nanmax(atr / close) - np.nanmin(atr / close) + 1e-8) - 1
            
            # New Features
            features_df['adx_trend'] = adx / 100  # Normalize to [0,1]
            features_df['volatility_breakout'] = (close - lower_band) / (upper_band - lower_band + 1e-8)
            features_df['body_to_range'] = body / (high - low + 1e-8)
            features_df['wick_ratio'] = (upper_wick - lower_wick) / (upper_wick + lower_wick + 1e-8)
            features_df['trend_strength'] = np.clip(adx/25 - 1, -1, 1)  # Normalized trend strength
        
        # Replace NaN values with zeros
        features_df = features_df.fillna(0)
        
        # Return both the features dataframe and the ATR values for position management
        return features_df, atr

    def _setup_action_space(self) -> None:
        """Configure discrete action space: 0=hold, 1=buy, 2=sell, 3=close."""
        self.action_space = spaces.Discrete(4)

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
            float: Reward for the action
        """
        # Update trade cooldown based on volatility
        current_atr = self.prices['atr'][self.current_step]
        vol_scale = current_atr / self.prices['close'][self.current_step]
        self.trade_cooldown = max(2, int(self.base_cooldown * (1.0 - vol_scale * 5)))
        
        # Check if we already have a position
        if self.current_position is not None:
            return -0.1  # Penalty for trying to open when already have position
            
        # Check cooldown
        if self.steps_since_trade < self.trade_cooldown:
            self.steps_since_trade += 1
            return -0.1  # Penalty for trading too frequently
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        
        # Calculate lot size based on account balance
        lot_size = max(
            self.MIN_LOTS, 
            min(
                self.MAX_LOTS,
                round(self.balance / self.BALANCE_PER_LOT, 2)
            )
        )
        
        # Create position
        self.current_position = {
            "direction": 1 if direction == 1 else -1,  # 1 for buy, -1 for sell
            "entry_price": current_price + (raw_spread if direction == 1 else -raw_spread),
            "lot_size": lot_size,
            "entry_time": str(self.original_index[self.current_step]),
            "entry_step": self.current_step,
            "entry_atr": current_atr,
            "current_profit_pips": 0.0
        }
        
        self.trade_metrics['current_direction'] = self.current_position["direction"]
        self.steps_since_trade = 0
        
        # Calculate entry reward based on volatility and trend alignment
        trend = self.raw_data.values[self.current_step][2]  # Normalized trend feature
        trend_alignment_bonus = 0.0
        
        if (direction == 1 and trend > 0.3) or (direction == 2 and trend < -0.3):
            trend_alignment_bonus = 0.2  # Bonus for trading with trend
            
        volatility_bonus = min(1.0, current_atr / (self.prices['close'][self.current_step] * 0.01)) * 0.1
        
        return 0.1 + trend_alignment_bonus + volatility_bonus
    
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
        
        # Calculate profit or loss
        if direction == 1:  # Long position
            profit_points = current_price - entry_price
        else:  # Short position
            profit_points = entry_price - current_price
            
        pnl = profit_points * lot_size
        profit_pips = profit_points / self.PIP_VALUE
        
        # Record trade details
        self.current_position.update({
            "exit_price": current_price,
            "exit_step": self.current_step,
            "exit_time": str(self.original_index[self.current_step]),
            "profit_pips": profit_pips,
            "pnl": pnl,
            "hold_time": self.current_step - self.current_position["entry_step"]
        })
        
        # Update trade statistics
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        self.trades.append(self.current_position)
        
        # Update balance and clear position
        self.balance += pnl
        self.current_position = None
        self.trade_metrics['current_direction'] = 0
        
        # Update trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t["pnl"] > 0]
            losing_trades = [t for t in self.trades if t["pnl"] <= 0]
            
            self.trade_metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0.0
            self.trade_metrics['avg_profit'] = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
            self.trade_metrics['avg_loss'] = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Reward based on P/L and hold time
        hold_time = self.current_step - self.current_position["entry_step"]
        hold_factor = min(1.0, hold_time / 20)  # Scale factor based on hold time
        
        # Base reward on P/L relative to account size
        normalized_pnl = pnl / self.initial_balance * 100
        
        return normalized_pnl * hold_factor

    def _manage_position(self) -> float:
        """Manage current position - update unrealized P/L and check for exits.
        
        Returns:
            float: Current unrealized P/L
        """
        if not self.current_position:
            return 0.0
            
        current_price = self.prices['close'][self.current_step]
        direction = self.current_position["direction"]
        entry_price = self.current_position["entry_price"]
        lot_size = self.current_position["lot_size"]
        
        # Calculate unrealized P/L
        if direction == 1:  # Long position
            profit_points = current_price - entry_price
        else:  # Short position
            profit_points = entry_price - current_price
            
        unrealized_pnl = profit_points * lot_size
        profit_pips = profit_points / self.PIP_VALUE
        
        self.current_position["current_profit_pips"] = profit_pips
        
        # Check for stop loss or take profit (automatic exit conditions)
        current_atr = self.prices['atr'][self.current_step]
        entry_atr = self.current_position["entry_atr"]
        
        # Dynamic stop loss and take profit based on ATR
        stop_loss_pips = -2.0 * entry_atr / self.PIP_VALUE
        take_profit_pips = 3.0 * entry_atr / self.PIP_VALUE
        
        # Adjust stop/target based on time in trade
        hold_time = self.current_step - self.current_position["entry_step"]
        if hold_time > 10:
            # Tighten stop loss and take profit as time passes
            stop_loss_pips = stop_loss_pips * (1.0 + min(1.0, hold_time / 20))
            take_profit_pips = take_profit_pips * (1.0 - min(0.3, hold_time / 40))
        
        # Check exit conditions
        if (profit_pips <= stop_loss_pips) or (profit_pips >= take_profit_pips):
            self._close_position()
            
        return unrealized_pnl
        
    def calculate_reward(self, unrealized_pnl: float) -> float:
        """Calculate reward based on unrealized P/L and risk metrics.
        
        Args:
            unrealized_pnl: Current unrealized P/L for open position
            
        Returns:
            float: Calculated reward
        """
        # Primary reward based on return on equity
        roe = (self.balance - self.previous_balance) / self.initial_balance
        base_reward = roe * 100  # Scale up for learning
        
        # Add unrealized P/L component if position is open
        if self.current_position:
            # Scale unrealized P/L by position hold time
            hold_time = self.current_step - self.current_position["entry_step"]
            unrealized_factor = min(1.0, hold_time / 20)  # Increase weight as position is held longer
            unrealized_roe = unrealized_pnl / self.initial_balance
            unrealized_component = unrealized_roe * 50 * unrealized_factor  # Lower weight than realized P/L
            base_reward += unrealized_component
        
        # Risk adjustment based on drawdown
        current_drawdown = (self.max_balance - self.balance) / self.max_balance
        risk_multiplier = max(0.1, 1.0 - (current_drawdown * 2))  # Floor at 0.1
        
        return base_reward * risk_multiplier
        
    def get_action_penalty(self) -> float:
        """Calculate penalties for undesirable actions."""
        if self.steps_since_trade < self.trade_cooldown:
            return -0.1  # Penalty for trading too frequently
        
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
        action = self._process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.episode_steps += 1
        self.current_step += 1
        
        # Calculate terminal conditions
        end_of_data = (self.current_step >= len(self.raw_data) - 1)
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        done = end_of_data or self.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN

        # Store previous balance for reward calculation
        self.previous_balance = previous_balance
        
        # Execute trade actions based on action
        reward_modifier = 0.0
        
        if action == 0:  # Hold
            self.steps_since_trade += 1
        elif action == 1:  # Buy
            reward_modifier = self._execute_trade(1, current_spread)
        elif action == 2:  # Sell
            reward_modifier = self._execute_trade(2, current_spread)
        elif action == 3:  # Close
            reward_modifier = self._close_position()
            
        # Manage current position
        unrealized_pnl = self._manage_position()
            
        # Calculate core rewards
        reward = self.calculate_reward(unrealized_pnl)  # Core reward based on ROE
        reward += reward_modifier                        # Add action rewards/penalties
        reward += self.get_action_penalty()              # Add any action penalties
        
        # Check terminal conditions
        if done:
            # Close any open position at end of episode
            if self.current_position:
                self._close_position()
            
            # Add terminal reward
            reward += self.get_terminal_reward()
        
        obs = self.get_history()
        self.reward = reward
        
        truncated = self.current_step >= len(self.raw_data) - 1
        
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
        
        return obs, reward, done, truncated, {
            "balance": self.balance,
            "total_pnl": self.balance - self.initial_balance,
            "drawdown": max_drawdown * 100,
            "position": position_info,
            "trade_metrics": self.trade_metrics
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
        self.current_position = None
        
        self.steps_since_trade = 0
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
        """Get current bar features."""
        return self.raw_data.values[self.current_step]

    def render(self) -> None:
        """Print environment state and trade statistics."""
        print(f"\n===== Episode {self.completed_episodes}, Step {self.episode_steps} =====")
        print(f"Current Balance: {self.balance:.2f}")
        print(f"Current Position: {'None' if not self.current_position else ('Long' if self.current_position['direction'] == 1 else 'Short')}")
        
        if self.current_position:
            unrealized_pnl = 0
            if self.current_position["direction"] == 1:  # Long
                unrealized_pnl = (self.prices['close'][self.current_step] - self.current_position["entry_price"]) * self.current_position["lot_size"]
            else:  # Short
                unrealized_pnl = (self.current_position["entry_price"] - self.prices['close'][self.current_step]) * self.current_position["lot_size"]
                
            print(f"Position Details:")
            print(f"  Entry Price: {self.current_position['entry_price']:.5f}")
            print(f"  Current Price: {self.prices['close'][self.current_step]:.5f}")
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
