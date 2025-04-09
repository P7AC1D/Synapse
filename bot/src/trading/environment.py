"""Trading environment for single-position trading with PPO-LSTM."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
from gymnasium.utils import EzPickle
import gymnasium as gym

from .actions import Action, ActionHandler
from .features import FeatureProcessor
from .metrics import MetricsTracker
from .rendering import Renderer
from .rewards import RewardCalculator

class TradingEnv(gym.Env, EzPickle):
    """Trading environment for single-position trading with PPO-LSTM."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    @property
    def win_count(self) -> int:
        """Get win count from metrics."""
        return self.metrics.win_count if hasattr(self, 'metrics') else 0
        
    @property
    def loss_count(self) -> int:
        """Get loss count from metrics."""
        return self.metrics.loss_count if hasattr(self, 'metrics') else 0
    
    @property
    def balance(self) -> float:
        """Get current account balance."""
        if hasattr(self, 'metrics'):
            return self.metrics.balance
        return self.initial_balance  # Fallback before metrics initialization
        
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000,
                 balance_per_lot: float = 1000.0, random_start: bool = False,
                 max_hold_bars: int = 64, min_hold_bars: int = 3, model = None):
        """Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot
            random_start: Whether to start from random positions
            max_hold_bars: Maximum bars to hold a position
            model: PPO model reference for state persistence
        """
        super().__init__()
        EzPickle.__init__(self)
        
        # Trading constants
        self.POINT_VALUE = 0.01      # Gold moves in 0.01 increments
        self.PIP_VALUE = 0.01        # Gold pip and point values are the same
        self.MIN_LOTS = 0.01         # Minimum 0.01 lots
        self.MAX_LOTS = 50.0         # Maximum lots
        self.CONTRACT_SIZE = 100.0   # Standard gold contract size
        self.BALANCE_PER_LOT = balance_per_lot
        self.MAX_DRAWDOWN = 0.4      # Maximum drawdown
        self.MAX_HOLD_BARS = max_hold_bars
        self.initial_balance = initial_balance
        
        # Initialize components
        self.feature_processor = FeatureProcessor()
        self.metrics = MetricsTracker(initial_balance)
        self.action_handler = ActionHandler(self)
        self.renderer = Renderer()
        self.reward_calculator = RewardCalculator(self, max_hold_bars)
        
        # Verify required columns
        required_columns = ['open', 'close', 'high', 'low', 'spread']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Add volume if not present
        if 'volume' not in data.columns:
            print("Warning: 'volume' column not found, using synthetic volume data")
            data['volume'] = np.ones(len(data))
            
        # Process data and setup spaces
        self.raw_data, self.atr_values = self.feature_processor.preprocess_data(data)
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.feature_processor.setup_observation_space()
        
        # Store model reference for state persistence
        self.model = model
        self._last_lstm_state = None
        
        # Save datetime index and data length
        self.original_index = data.index
        self.data_length = len(self.raw_data)
        
        # Store price data
        self.prices = {
            'close': data.loc[self.original_index, 'close'].values,
            'high': data.loc[self.original_index, 'high'].values,
            'low': data.loc[self.original_index, 'low'].values,
            'spread': data.loc[self.original_index, 'spread'].values,
            'atr': self.atr_values
        }
        
        # State variables
        self.initial_balance = initial_balance
        self.random_start = random_start
        self.current_step = 0
        self.episode_steps = 0
        self.completed_episodes = 0
        self.current_position = None
        self.trades = []
        self.trade_metrics = {'current_direction': 0}
        
        # Trading constraints
        self.min_hold_bars = min_hold_bars
        self.current_hold_time = 0
        self._optimal_hold = min_hold_bars  # Initialize optimal hold time
        
    def calculate_optimal_hold_time(self) -> int:
        """Calculate optimal hold time based on current market conditions."""
        current_atr = self.prices['atr'][self.current_step]
        base_hold = self.min_hold_bars
        
        # Scale hold time by ATR volatility
        volatility_factor = current_atr / np.mean(self.prices['atr'])
        optimal_hold = int(base_hold * (1/volatility_factor))
        
        # Ensure within bounds
        return max(self.min_hold_bars, min(optimal_hold, self.MAX_HOLD_BARS))
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Process action and get current state
        action = self.action_handler.process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
        
        # Get position info before action
        position_type = self.current_position["direction"] if self.current_position else 0
        bars_held = self.current_step - self.current_position["entry_step"] if self.current_position else 0
        current_atr = self.prices['atr'][self.current_step]
        
        # Update counters
        self.episode_steps += 1
        self.current_step += 1
        
        # Update optimal hold time and initialize reward
        self._optimal_hold = self.calculate_optimal_hold_time()
        reward = 0.0
        
        # Update hold time for existing positions
        if self.current_position:
            self.current_hold_time += 1

        # Handle different actions
        if action in [Action.BUY, Action.SELL]:
            if self.current_position is not None:
                reward = -0.5  # Penalty for invalid action
            else:
                direction = 1 if action == Action.BUY else 2
                self.action_handler.execute_trade(direction, current_spread)
                self.current_hold_time = 0
                reward = self.reward_calculator.calculate_reward(
                    action, position_type, 0, current_atr,
                    self.current_hold_time, self._optimal_hold)
                
        elif action == Action.CLOSE:
            if not self.current_position:
                reward = -0.1  # Penalty for closing without position
            elif self.current_hold_time < self.min_hold_bars:
                reward = -0.1  # Penalty for early close
                action = Action.HOLD
            elif self.current_hold_time > self._optimal_hold * 1.5:
                # Close position with holding penalty
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
                reward = self.reward_calculator.calculate_reward(
                    action, position_type, pnl, current_atr,
                    self.current_hold_time, self._optimal_hold) - 0.05
                self.current_position = None
                self.current_hold_time = 0
            else:
                # Normal close
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
                reward = self.reward_calculator.calculate_reward(
                    action, position_type, pnl, current_atr,
                    self.current_hold_time, self._optimal_hold)
                self.current_position = None
                self.current_hold_time = 0
                
        elif action == Action.HOLD:
            unrealized_pnl, profit_pips = self.action_handler.manage_position()
            if self.current_position:
                self.current_position["current_profit_pips"] = profit_pips
                reward = self.reward_calculator.calculate_reward(
                    action, position_type, unrealized_pnl, current_atr,
                    self.current_hold_time, self._optimal_hold)
        
        # Check for bankruptcy
        if self.metrics.balance <= 0:
            final_obs = self.get_observation()
            terminal_reward = self.reward_calculator.calculate_terminal_reward(self.metrics.balance, self.initial_balance)
            return final_obs, terminal_reward, True, False, self._get_info()
        
        # Calculate terminal conditions
        end_of_data = self.current_step >= self.data_length - 1
        max_drawdown = self.metrics.get_drawdown()
        done = end_of_data or self.metrics.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN
        
        # Auto-close position at end of episode
        if done:
            if self.current_position and self.current_hold_time >= self.min_hold_bars:
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
            self.current_hold_time = 0  # Reset hold time at end of episode
        
        # Get observation and check truncation
        obs = self.get_observation()
        truncated = self.current_step >= self.data_length - 1
        
        return obs, float(reward), done, truncated, self._get_info()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
        # Store LSTM state before reset if available
        if self.model is not None and hasattr(self.model.policy, 'lstm_states'):
            self._last_lstm_state = self.model.policy.lstm_states
            
        if self.random_start:
            max_start = max(0, self.data_length - 100)
            self.current_step = np.random.randint(0, max_start)
        else:
            self.current_step = 0
            
        self.episode_steps = 0
        self.completed_episodes += 1
        self.current_position = None
        self.trades = []
        self.trade_metrics = {'current_direction': 0}
        
        # Reset metrics and reward state
        self.metrics.reset()
        self.reward_calculator.previous_balance_high = self.initial_balance
        self.reward_calculator.last_direction = None
        self.reward_calculator.bars_since_consolidation = 0
        
        # Restore LSTM state if available
        if self._last_lstm_state is not None and self.model is not None:
            self.model.policy.lstm_states = self._last_lstm_state
        
        return self.get_observation(), self._get_info()
        
    def render(self) -> None:
        """Render the environment."""
        self.renderer.render_episode_stats(self)
        
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        features = self.raw_data.values[self.current_step]
        
        position_type = self.current_position["direction"] if self.current_position else 0
        normalized_hold_time = 0.0
        
        if self.current_position:
            hold_time = self.current_step - self.current_position["entry_step"]
            normalized_hold_time = min(hold_time / self.MAX_HOLD_BARS, 1.0)
            unrealized_pnl, _ = self.action_handler.manage_position()
            normalized_pnl = np.clip(unrealized_pnl / self.initial_balance, -1, 1)
        else:
            normalized_pnl = 0.0
            
        return np.append(features, [position_type, normalized_hold_time, normalized_pnl])
        
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        position_info = {}
        
        if self.current_position:
            unrealized_pnl, _ = self.action_handler.manage_position()
            position_info = {
                "direction": "long" if self.current_position["direction"] == 1 else "short",
                "entry_price": self.current_position["entry_price"],
                "lot_size": self.current_position["lot_size"],
                "unrealized_pnl": unrealized_pnl,
                "profit_pips": self.current_position.get("current_profit_pips", 0.0),
                "hold_time": self.current_step - self.current_position["entry_step"]
            }
            
        return {
            "balance": self.metrics.balance,
            "total_pnl": self.metrics.balance - self.initial_balance,
            "drawdown": self.metrics.get_drawdown() * 100,
            "position": position_info,
            "trade_metrics": self.metrics.metrics,
            "total_trades": len(self.trades),
            "win_count": self.metrics.win_count,
            "loss_count": self.metrics.loss_count
        }
