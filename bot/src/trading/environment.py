"""Trading environment for single-position trading with PPO-LSTM."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
from gymnasium.utils import EzPickle
import gymnasium as gym

from .actions import Action, ActionHandler
from .enhanced_features import EnhancedFeatureProcessor as FeatureProcessor
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
                 predict_mode: bool = False, currency_conversion: Optional[float] = None,
                 point_value: float = 0.001,
                 min_lots: float = 0.01, max_lots: float = 200.0,
                 contract_size: float = 100.0,
                 spread_variation: float = 0.0,
                 slippage_range: float = 0.0):
        """Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot
            random_start: Whether to start from random positions
            predict_mode: Whether environment is being used for live prediction (True) or backtesting (False)
            currency_conversion: Optional conversion rate for account currency (e.g. USD/ZAR)
            point_value: Value of one price point movement (default: 0.001 for Gold)
            min_lots: Minimum lot size (default: 0.01)
            max_lots: Maximum lot size (default: 200.0)
            contract_size: Standard contract size (default: 100.0 for Gold)
        """
        super().__init__()
        EzPickle.__init__(self)
        
        # Trading constants
        self.POINT_VALUE = point_value
        self.MIN_LOTS = min_lots
        self.MAX_LOTS = max_lots
        self.CONTRACT_SIZE = contract_size
        self.BALANCE_PER_LOT = balance_per_lot
        self.MAX_DRAWDOWN = 0.4      # Maximum drawdown
        self.initial_balance = initial_balance
        self.currency_conversion = currency_conversion or 1.0  # Default to 1.0 if not provided
        
        # Set predict_context flag for live trading vs backtesting
        self.predict_context = predict_mode
        
        # Market simulation settings
        self.spread_variation = spread_variation
        self.slippage_range = slippage_range
        
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
        # Observation space = features + position_type + unrealized_pnl
        feature_count = len(self.raw_data.columns) + 2  # +2 for position_type and unrealized_pnl
        self.observation_space = self.feature_processor.setup_observation_space(feature_count)
        
        # Save original index for reference
        self.original_index = data.index
        self.data_length = len(self.raw_data)
        
        # CRITICAL FIX: Align price data to match features index after lookback removal
        # This prevents look-ahead bias by ensuring features and prices have same temporal alignment
        features_start_idx = len(data) - len(self.raw_data)
        aligned_data = data.iloc[features_start_idx:].copy()
        
        # Verify alignment - this prevents look-ahead bias
        assert len(aligned_data) == len(self.raw_data), f"Length mismatch: {len(aligned_data)} vs {len(self.raw_data)}"
        assert aligned_data.index.equals(self.raw_data.index), "Index mismatch between features and prices"
        
        # Store properly aligned price data - no more look-ahead bias!
        self.prices = {
            'close': aligned_data['close'].values,
            'high': aligned_data['high'].values,
            'low': aligned_data['low'].values,
            'spread': aligned_data['spread'].values,
            'atr': self.atr_values
        }
        
        # Store alignment info for debugging and validation
        self.alignment_info = {
            'original_length': len(data),
            'features_length': len(self.raw_data),
            'lookback_removed': features_start_idx,
            'start_date': self.raw_data.index[0],
            'end_date': self.raw_data.index[-1],
            'original_start_date': data.index[0],
            'original_end_date': data.index[-1]
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
        
    def validate_alignment(self) -> None:
        """Validate that features and prices are properly aligned to prevent look-ahead bias."""
        print("=== Alignment Validation ===")
        print(f"Original data length: {self.alignment_info['original_length']}")
        print(f"Features length: {self.alignment_info['features_length']}")
        print(f"Lookback removed: {self.alignment_info['lookback_removed']} bars")
        print(f"Original date range: {self.alignment_info['original_start_date']} to {self.alignment_info['original_end_date']}")
        print(f"Aligned date range: {self.alignment_info['start_date']} to {self.alignment_info['end_date']}")
        
        # Verify lengths match
        assert len(self.raw_data) == len(self.prices['close']), "Feature-price length mismatch"
        assert len(self.raw_data) == len(self.prices['atr']), "Feature-ATR length mismatch"
        
        # Test temporal alignment at key points
        test_indices = [0, len(self.raw_data)//2, len(self.raw_data)-1]
        for i in test_indices:
            feature_time = self.raw_data.index[i]
            price_close = self.prices['close'][i]
            print(f"Step {i}: Feature timestamp = {feature_time}, Price = {price_close:.5f}")
            
        print("✓ Alignment validation passed - no look-ahead bias detected")
        
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
            if self.predict_context and "profit" in self.current_position:
                # Use position's profit for live trading
                unrealized_pnl = self.current_position["profit"]
                profit_points = self.current_position.get("profit_points", 0.0)
                self.current_position["current_profit_points"] = profit_points
            else:
                # Calculate PnL for backtesting
                unrealized_pnl, profit_points = self.action_handler.manage_position()
                self.current_position["current_profit_points"] = profit_points
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
        max_drawdown = self.metrics.get_equity_drawdown()
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
        # Reset episode tracking for the new reward system
        if hasattr(self.reward_calculator, 'reset_episode_tracking'):
            self.reward_calculator.reset_episode_tracking()
        self.reward_calculator.max_unrealized_pnl = 0.0  # Reset max unrealized PnL tracking
        
        return self.get_observation(), self._get_info()
        
    def render(self) -> None:
        """Render the environment."""
        self.renderer.render_episode_stats(self)
        
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        # In predict context, we want the most recent data point
        features = self.raw_data.values[-1] if self.predict_context else self.raw_data.values[self.current_step]
        
        position_type = self.current_position["direction"] if self.current_position else 0
        
        if self.current_position:
            # For live trading, use position's profit if available
            if self.predict_context and "profit" in self.current_position:
                unrealized_pnl = self.current_position["profit"]
            else:
                # Fallback to calculating PnL for backtesting
                unrealized_pnl, _ = self.action_handler.manage_position()
            # Normalize unrealized PnL to be between -1 and 1 based on balance
            normalized_pnl = np.clip(unrealized_pnl / self.metrics.balance, -1, 1)
        else:
            normalized_pnl = 0.0
            
        return np.append(features, [position_type, normalized_pnl])
        
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information."""
        position_info = {}
        
        if self.current_position:
            if self.predict_context and "profit" in self.current_position:
                # Use position's profit for live trading
                unrealized_pnl = self.current_position["profit"]
                profit_points = self.current_position.get("profit_points", 0.0)
            else:
                # Calculate PnL for backtesting
                unrealized_pnl, profit_points = self.action_handler.manage_position()
                
            position_info = {
                "direction": "long" if self.current_position["direction"] == 1 else "short",
                "entry_price": self.current_position["entry_price"],
                "lot_size": self.current_position["lot_size"],
                "unrealized_pnl": unrealized_pnl,
                "profit_points": profit_points,
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
