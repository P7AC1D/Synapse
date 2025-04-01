import gymnasium
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces

# Suppress gymnasium warnings about step API
import warnings
import gymnasium as gym
warnings.filterwarnings('ignore', category=DeprecationWarning)

class TradingEnv(gym.Env):
    metadata = {"render_modes": None}
    
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
        self.BALANCE_PER_LOT = 1000.0  # Amount in balance required for 0.01 lot
        self.MAX_DRAWDOWN = 0.5     # Maximum allowed drawdown (50%)
        self.MIN_GRID_MULTIPLIER = 0.1  # Minimum grid size as ATR multiplier
        self.MAX_GRID_MULTIPLIER = 3.0  # Maximum grid size as ATR multiplier
        
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
        self.long_positions: List[Dict[str, Any]] = []  # Long (buy) positions
        self.short_positions: List[Dict[str, Any]] = [] # Short (sell) positions
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        # Trading metrics
        self.completed_episodes = 0
        self.episode_steps = 0
        self.trade_cooldown = 5  # Minimum bars between trades
        self.long_count = 0      # Track long trades
        self.short_count = 0     # Track short trades
        
        # Grid tracking
        self.grid_id_counter = 0
        self.active_grids = {
            'long': {},  # Dict to track long grid trades
            'short': {}  # Dict to track short grid trades
        }
        self.grid_metrics = {
            'total_grids': 0,
            'avg_positions_per_grid': 0.0,
            'grid_efficiency': 0.0
        }
        
        # Register current state
        self.current_grid_profits = {
            'long': {},  # Track profits for each long grid
            'short': {}  # Track profits for each short grid
        }
        
        self._setup_action_space()
        self._setup_observation_space(5)  # 5 features

    def _setup_action_space(self) -> None:
        """Configure action space for position direction and grid size."""
        self.action_space = spaces.Box(
            low=np.array([-1, self.MIN_GRID_MULTIPLIER]),  # Direction and min grid size
            high=np.array([1, self.MAX_GRID_MULTIPLIER]),  # Direction and max grid size
            dtype=np.float32
        )

    def _setup_observation_space(self, feature_count: int) -> None:
        """Setup observation space for features only."""
        obs_dim = self.bar_count * feature_count
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_dim,), dtype=np.float32
        )

    def _process_action(self, action: np.ndarray) -> Tuple[int, float]:
        """Process action to determine direction and grid size."""
        # Extract direction (-1 sell, 0 hold, 1 buy)
        direction = np.sign(action[0]) if abs(action[0]) > 0.1 else 0
        
        # Extract normalized grid size and convert to ATR multiplier
        grid_multiplier = np.clip(action[1], self.MIN_GRID_MULTIPLIER, self.MAX_GRID_MULTIPLIER)
        
        return direction, grid_multiplier

    def _execute_grid_trade(self, direction: int, grid_multiplier: float, raw_spread: float) -> float:
        """Execute a trade as part of a grid strategy."""
        if direction == 0 or self.steps_since_trade < self.trade_cooldown:
            self.steps_since_trade += 1
            return -0.1 if direction != 0 else 0.0
        
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        grid_size_pips = current_atr * grid_multiplier
        
        # Calculate lot size based on account balance (0.01 lot per $1000)
        lot_size = max(self.MIN_LOTS, round(self.balance / self.BALANCE_PER_LOT / 100, 2))
        lot_size = min(lot_size, self.MAX_LOTS)
        
        # Initialize grid
        self.grid_id_counter += 1
        grid_id = self.grid_id_counter
        grid_direction = 'long' if direction == 1 else 'short'
        
        # Create new position
        position = {
            "grid_id": grid_id,
            "direction": direction,
            "entry_price": current_price + (raw_spread if direction == 1 else -raw_spread),
            "lot_size": lot_size,
            "grid_size_pips": grid_size_pips,
            "grid_multiplier": grid_multiplier,
            "entry_step": self.current_step,
            "entry_atr": current_atr,
            "entry_spread": raw_spread,
            "current_profit_pips": 0.0
        }
        
        # Initialize or update grid tracking
        if grid_id not in self.active_grids[grid_direction]:
            self.active_grids[grid_direction][grid_id] = {
                'positions': [position],
                'grid_size': grid_size_pips,
                'entry_step': self.current_step,
                'total_profit': 0.0
            }
        else:
            self.active_grids[grid_direction][grid_id]['positions'].append(position)
        
        # Add position to appropriate list
        if direction == 1:
            self.long_positions.append(position)
            self.long_count += 1
        else:
            self.short_positions.append(position)
            self.short_count += 1
        self.steps_since_trade = 0
        
        # Return small reward for opening a position in less-traded direction
        total_trades = max(1, self.long_count + self.short_count)
        long_ratio = self.long_count / total_trades
        direction_bonus = 0.2 * (1 - long_ratio) if direction == 1 else 0.2 * long_ratio
        
        return direction_bonus

    def _manage_grid_positions(self) -> Tuple[float, List[Dict[str, Any]]]:
        """Manage grid positions including entry, exit, and pyramiding decisions."""
        total_reward = 0.0
        closed_positions = []
        
        # Process long positions
        long_reward, long_closed = self._manage_direction_grid(self.long_positions, 1)
        total_reward += long_reward
        closed_positions.extend(("long", idx) for idx in long_closed)
        
        # Process short positions
        short_reward, short_closed = self._manage_direction_grid(self.short_positions, -1)
        total_reward += short_reward
        closed_positions.extend(("short", idx) for idx in short_closed)
        
        # Close positions and update metrics
        self._process_closed_positions(closed_positions)
        
        return total_reward, closed_positions

    def _manage_direction_grid(self, positions: List[Dict[str, Any]], direction: int) -> Tuple[float, List[int]]:
        """Manage grid positions for a specific direction."""
        if not positions:
            return 0.0, []
            
        current_price = self.prices['close'][self.current_step]
        current_atr = self.prices['atr'][self.current_step]
        positions_to_close = []
        total_pnl = 0.0
        
        # Update profit/loss for all positions
        for pos in positions:
            if direction == 1:  # Long positions
                profit_pips = (current_price - pos["entry_price"]) / self.PIP_VALUE
            else:  # Short positions
                profit_pips = (pos["entry_price"] - current_price) / self.PIP_VALUE
                
            pos["current_profit_pips"] = profit_pips
            total_pnl += profit_pips * pos["lot_size"]
            
        # Get grid parameters from most recent position
        current_grid = positions[-1]
        grid_size_pips = current_grid["grid_size_pips"]
        
        # Close all positions if total profit exceeds grid size
        if total_pnl > grid_size_pips:
            positions_to_close = list(range(len(positions)))
            return self._calculate_grid_reward(positions, total_pnl), positions_to_close
            
        # Check worst position for grid expansion
        worst_position = min(positions, key=lambda x: x["current_profit_pips"])
        if worst_position["current_profit_pips"] < -grid_size_pips:
            # Add new position with current grid parameters
            self._execute_grid_trade(direction, current_grid["grid_multiplier"], 
                                   self.prices['spread'][self.current_step] * self.POINT_VALUE)
        
        return self._calculate_grid_reward(positions, total_pnl), positions_to_close

    def _calculate_grid_reward(self, positions: List[Dict[str, Any]], total_pnl: float) -> float:
        """Calculate reward for grid trading performance."""
        if not positions:
            return 0.0
            
        # Base reward on PnL relative to account size
        reward = total_pnl * self.PIP_VALUE * 100.0  # Scale pip value
        
        # Reward for proper grid utilization
        grid_utilization = len(positions) / max(1, abs(total_pnl / positions[0]["grid_size_pips"]))
        grid_reward = min(3.0, grid_utilization) * 0.5
        
        # Penalize excessive drawdown
        drawdown = (self.max_balance - self.balance) / self.max_balance
        drawdown_penalty = max(0, (drawdown - 0.1) * 10.0)  # Start penalizing after 10% drawdown
        
        return (reward + grid_reward - drawdown_penalty) / self.initial_balance

    def _update_grid_metrics(self, closed_direction: str = None, closed_grid_id: int = None) -> None:
        """Update grid trading performance metrics."""
        long_grids = len(self.active_grids['long'])
        short_grids = len(self.active_grids['short'])
        total_grids = long_grids + short_grids
        
        # Count active and completed positions
        total_positions = 0
        total_pnl = 0.0
        
        for direction in ['long', 'short']:
            for grid_id, grid in self.active_grids[direction].items():
                # Count positions in this grid
                total_positions += len(grid['positions'])
                total_pnl += grid['total_profit']
        
        # Include closed trades in metrics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            total_grids = max(total_grids, trades_df['grid_id'].nunique())
            total_positions += len(trades_df)
            total_pnl += trades_df['pnl'].sum()

        # Calculate metrics
        if total_grids > 0:
            self.grid_metrics.update({
                'total_grids': total_grids,
                'avg_positions_per_grid': total_positions / total_grids,
                'grid_efficiency': (total_pnl / self.initial_balance) * 100
            })
        else:
            self.grid_metrics.update({
                'total_grids': 0,
                'avg_positions_per_grid': 0.0,
                'grid_efficiency': 0.0
            })

    def _process_closed_positions(self, closed_positions: List[Tuple[str, int]]) -> None:
        """Process and record closed positions."""
        # Sort in reverse order to safely remove positions
        closed_positions.sort(key=lambda x: x[1], reverse=True)
        
        for direction, idx in closed_positions:
            positions = self.long_positions if direction == "long" else self.short_positions
            if 0 <= idx < len(positions):
                position = positions[idx]
                
                # Calculate final PnL
                exit_price = self.prices['close'][self.current_step]
                if direction == "long":
                    profit_pips = (exit_price - position["entry_price"]) / self.PIP_VALUE
                else:
                    profit_pips = (position["entry_price"] - exit_price) / self.PIP_VALUE
                
                pnl = profit_pips * position["lot_size"] * self.PIP_VALUE
                self.balance += pnl
                
                # Update grid tracking
                grid_id = position.get('grid_id', None)
                if grid_id is not None:
                    if direction == "long":
                        if grid_id in self.active_grids['long']:
                            self.active_grids['long'][grid_id]['total_profit'] += pnl
                    else:
                        if grid_id in self.active_grids['short']:
                            self.active_grids['short'][grid_id]['total_profit'] += pnl

                # Record completed trade
                position.update({
                    "pnl": pnl,
                    "exit_price": exit_price,
                    "exit_step": self.current_step,
                    "profit_pips": profit_pips,
                    "grid_id": grid_id
                })
                self.trades.append(position)
                
                # Update grid metrics
                self._update_grid_metrics()
                
                # Update win/loss counts
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                # Remove closed position
                positions.pop(idx)

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
        direction, grid_multiplier = self._process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.episode_steps += 1
        self.current_step += 1
        
        done = (self.episode_steps >= self.max_steps) or (self.current_step >= len(self.raw_data) - 1)

        # Process existing positions and get grid trading reward
        grid_reward, _ = self._manage_grid_positions()
        
        # Execute new trade if requested and no positions exist in that direction
        execution_reward = 0.0
        if direction != 0:
            positions = self.long_positions if direction == 1 else self.short_positions
            if not positions:  # Only open new position if none exist in this direction
                execution_reward = self._execute_grid_trade(direction, grid_multiplier, current_spread)
        
        # Calculate growth and drawdown components
        growth_reward = 0.0
        drawdown_penalty = 0.0
        
        if previous_balance > 0:
            # Growth reward
            growth_ratio = self.balance / self.initial_balance
            if growth_ratio > 1:
                growth_reward = np.log(growth_ratio) * 5.0
            
            # Additional reward for accelerating growth
            if hasattr(self, 'last_balance_ratio'):
                growth_acceleration = self.balance / previous_balance - self.last_balance_ratio
                if growth_acceleration > 0:
                    growth_reward += growth_acceleration * 2.0
            self.last_balance_ratio = self.balance / previous_balance
            
            # Drawdown penalty
            drawdown = (self.max_balance - self.balance) / self.max_balance
            drawdown_penalty = drawdown * 5.0

        # Combine rewards
        reward = (grid_reward * 0.7) + (execution_reward * 0.2) + (growth_reward * 0.1) - (drawdown_penalty * 0.1)
        
        # Additional reward components
        if self.steps_since_trade > 5:
            reward -= (self.steps_since_trade - 5) * 0.05  # Penalize waiting too long
        
        # Add trading frequency bonus
        total_positions = len(self.long_positions) + len(self.short_positions)
        if total_positions > 0:
            position_frequency = total_positions / self.episode_steps
            reward += position_frequency * 2.0
            
        # Check for bankruptcy or excessive drawdown
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        if self.balance <= 0:
            self.long_positions.clear()
            self.short_positions.clear()
            reward = -100
            done = True
        elif max_drawdown >= self.MAX_DRAWDOWN:  # Using configured max drawdown
            self.long_positions.clear()
            self.short_positions.clear()
            reward = -50
            done = True
            
        # Episode completion rewards
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
        
        return obs, reward, done, False, {
            "total_pnl": self.balance - self.initial_balance,
            "drawdown": max_drawdown * 100,
            "grid_positions": len(self.long_positions) + len(self.short_positions),
            "grid_metrics": self.grid_metrics,
            "long_positions": len(self.long_positions),
            "short_positions": len(self.short_positions)
        }

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
            
        # Reset account state
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        
        # Reset all trading state
        self.long_positions.clear()
        self.short_positions.clear()
        self.trades.clear()
        
        # Reset trading counters
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.episode_steps = 0
        
        # Reset grid tracking
        self.grid_id_counter = 0
        self.active_grids = {'long': {}, 'short': {}}
        self.current_grid_profits = {'long': {}, 'short': {}}
        self.grid_metrics = {
            'total_grids': 0,
            'avg_positions_per_grid': 0.0,
            'grid_efficiency': 0.0
        }
        
        self.completed_episodes += 1
        
        return self.get_history(), {
            "balance": self.balance,
            "total_grids": self.grid_metrics["total_grids"],
            "positions": 0
        }
        
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
        print(f"Current Balance: {self.balance:.2f}")
        print(f"Long Positions: {len(self.long_positions)}")
        print(f"Short Positions: {len(self.short_positions)}")
        
        if len(self.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        num_tp = sum(1 for trade in self.trades if trade["pnl"] > 0.0)
        num_sl = sum(1 for trade in self.trades if trade["pnl"] < 0.0)
        total_trades = len(self.trades)
        
        # Grid-specific metrics
        unique_grids = len(trades_df["grid_id"].unique()) if "grid_id" in trades_df else 0
        avg_grid_size = trades_df["grid_size_pips"].mean() if "grid_size_pips" in trades_df else 0
        avg_profit_per_grid = trades_df.groupby("grid_id")["pnl"].sum().mean() if "grid_id" in trades_df else 0
        max_positions_per_grid = trades_df.groupby("grid_id").size().max() if "grid_id" in trades_df else 0
        
        # Standard metrics
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        win_rate = (num_tp / total_trades * 100) if total_trades > 0 else 0.0
        
        # Direction-specific metrics
        long_trades = trades_df[trades_df["direction"] == 1] if "direction" in trades_df else pd.DataFrame()
        short_trades = trades_df[trades_df["direction"] == -1] if "direction" in trades_df else pd.DataFrame()
        
        long_win_rate = (long_trades[long_trades["pnl"] > 0].shape[0] / len(long_trades) * 100) if not long_trades.empty else 0.0
        short_win_rate = (short_trades[short_trades["pnl"] > 0].shape[0] / len(short_trades) * 100) if not short_trades.empty else 0.0
        
        # Calculate Sharpe ratio
        if len(trades_df) > 1:
            returns = trades_df["pnl"] / self.initial_balance
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Grid performance metrics
        grid_metrics_text = (
            f"\n===== Grid Trading Metrics =====\n"
            f"Total Unique Grids: {unique_grids}\n"
            f"Average Grid Size: {avg_grid_size:.1f} pips\n"
            f"Average Profit per Grid: {avg_profit_per_grid:.2f}\n"
            f"Max Positions per Grid: {max_positions_per_grid}\n"
            f"Current Grid Positions: {len(self.long_positions) + len(self.short_positions)}\n"
        )
        
        # Overall performance metrics
        performance_metrics_text = (
            f"\n===== Performance Metrics =====\n"
            f"Total Return: {total_return:.2f}%\n"
            f"Total Trades: {total_trades}\n"
            f"Overall Win Rate: {win_rate:.2f}%\n"
            f"Long Win Rate: {long_win_rate:.2f}%\n"
            f"Short Win Rate: {short_win_rate:.2f}%\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
            f"Current Drawdown: {((self.max_balance - self.balance) / self.max_balance * 100):.2f}%\n"
        )
        
        print(grid_metrics_text)
        print(performance_metrics_text)
