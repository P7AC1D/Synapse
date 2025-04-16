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
                 balance_per_lot: float = 1000.0, random_start: bool = False):
        """Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot
            random_start: Whether to start from random positions
        """
        super().__init__()
        EzPickle.__init__(self)
        
        # Trading constants
        self.POINT_VALUE = 0.01      # Gold moves in 0.01 increments
        self.PIP_VALUE = 0.01        # Gold pip and point values are the same
        self.MIN_LOTS = 0.01         # Minimum 0.01 lots
        self.MAX_LOTS = 100.0         # Maximum lots
        self.CONTRACT_SIZE = 100.0   # Standard gold contract size
        self.BALANCE_PER_LOT = balance_per_lot
        self.MAX_DRAWDOWN = 0.4      # Maximum drawdown
        self.initial_balance = initial_balance
        
        # Initialize components
        self.feature_processor = FeatureProcessor()
        self.metrics = MetricsTracker(initial_balance)
        self.action_handler = ActionHandler(self)
        self.renderer = Renderer()
        self.reward_calculator = RewardCalculator(self)
        
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
        
        # State tracking
        self.current_hold_time = 0  # Track position hold duration
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Process action and get current state
        action = self.action_handler.process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE
        
        # Get position info before action
        position_type = self.current_position["direction"] if self.current_position else 0
        current_atr = self.prices['atr'][self.current_step]
        
        # Detect invalid actions
        invalid_action = False
        if action in [Action.BUY, Action.SELL] and self.current_position is not None:
            invalid_action = True  # Trying to open position when one exists
        elif action == Action.CLOSE and self.current_position is None:
            invalid_action = True  # Trying to close non-existent position
            
        # Update counters
        self.episode_steps += 1
        self.current_step += 1
        
        
        # Update hold time for existing positions
        if self.current_position:
            self.current_hold_time += 1

        # Initialize state variables
        pnl = 0.0
        unrealized_pnl = 0.0
        trade_info = None

        # Handle different actions
        if action in [Action.BUY, Action.SELL]:
            if self.current_position is None:
                direction = 1 if action == Action.BUY else 2
                self.action_handler.execute_trade(direction, current_spread)
                self.current_hold_time = 0
                
        elif action == Action.CLOSE and self.current_position:
            pnl, trade_info = self.action_handler.close_position()
            if pnl != 0:
                self.trades.append(trade_info)
                self.metrics.add_trade(trade_info)
                self.metrics.update_balance(pnl)
            self.current_position = None
            self.current_hold_time = 0
                
        # Track unrealized PnL for any active position
        if self.current_position:
            unrealized_pnl, profit_pips = self.action_handler.manage_position()
            self.current_position["current_profit_pips"] = profit_pips
            # Update metrics with unrealized PnL for accurate drawdown tracking
            self.metrics.update_unrealized_pnl(unrealized_pnl)
        else:
            self.metrics.update_unrealized_pnl(0.0)

        if action == Action.HOLD:
            # HOLD action with position - PnL already updated above
            pass

        # Calculate reward using RewardCalculator
        reward = self.reward_calculator.calculate_reward(
            action=action,
            position_type=position_type,
            pnl=pnl if action == Action.CLOSE else unrealized_pnl,
            atr=current_atr,
            current_hold=self.current_hold_time,
            optimal_hold=None,
            invalid_action=invalid_action
        )
        
        # Calculate terminal conditions
        end_of_data = self.current_step >= self.data_length - 1
        max_drawdown = self.metrics.get_drawdown()
        done = end_of_data or self.metrics.balance <= 0 or max_drawdown >= self.MAX_DRAWDOWN
        
        # Auto-close position at end of episode and handle terminal rewards
        if done:
            if self.current_position:
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
            self.current_hold_time = 0  # Reset hold time at end of episode
            
            # Calculate terminal reward
            terminal_reward = self.reward_calculator.calculate_terminal_reward(self.metrics.balance, self.initial_balance)
            reward += terminal_reward
        
        # Get observation and check truncation
        obs = self.get_observation()
        truncated = self.current_step >= self.data_length - 1
        
        return obs, float(reward), done, truncated, self._get_info()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            
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
        self.reward_calculator.max_unrealized_pnl = 0.0  # Reset max unrealized PnL tracking
        
        return self.get_observation(), self._get_info()
        
    def render(self) -> None:
        """Render the environment."""
        self.renderer.render_episode_stats(self)
        
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        features = self.raw_data.values[self.current_step]
        
        position_type = self.current_position["direction"] if self.current_position else 0
        
        if self.current_position:
            unrealized_pnl, _ = self.action_handler.manage_position()
            normalized_pnl = np.clip(unrealized_pnl / self.metrics.balance, -1, 1)
        else:
            normalized_pnl = 0.0
            
        return np.append(features, [position_type, normalized_pnl])
        
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
            "equity": self.metrics.get_equity_drawdown() * 100,
            "drawdown": self.metrics.get_drawdown() * 100,
            "position": position_info,
            "trade_metrics": self.metrics.metrics,
            "total_trades": len(self.trades),
            "win_count": self.metrics.win_count,
            "loss_count": self.metrics.loss_count
        }
