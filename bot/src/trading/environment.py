"""Trading environment for single-position trading with PPO-LSTM."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
from gymnasium.utils import EzPickle
import gymnasium as gym
import logging

from .actions import Action, ActionHandler
from .features import FeatureProcessor
from .metrics import MetricsTracker
from .rendering import Renderer
from .rewards import RewardCalculator

# Feature indices for market state
TREND_STRENGTH_IDX = 5
VOLATILITY_BREAKOUT_IDX = 4
TREND_THRESHOLD = 0.3
VOLATILITY_THRESHOLD = 0.7

@dataclass
class TradingConfig:
    """Configuration for trading environment."""
    initial_balance: float = 10000.0
    balance_per_lot: float = 1000.0
    point_value: float = 0.001
    min_lots: float = 0.01
    max_lots: float = 200.0
    contract_size: float = 100.0
    spread_variation: float = 0.0
    slippage_range: float = 0.0
    window_size: int = 50
    max_drawdown: float = 0.4
    min_bars_per_episode: int = 240
    currency_conversion: float = 1.0


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
        
    def __init__(self, data: pd.DataFrame, predict_mode: bool = False, config: Optional[TradingConfig] = None):
        """Initialize trading environment.
        
        Args:
            data: DataFrame with OHLCV data
            predict_mode: Whether environment is being used for live prediction (True) or backtesting (False)
            config: Trading environment configuration. If None, uses default values.
        """
        super().__init__()
        EzPickle.__init__(self)
        
        # Input validation
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be None or empty")
        
        # Initialize configuration and logging
        self.config = config or TradingConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing trading environment...")
        
        # Validate configuration
        if self.config.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if self.config.window_size <= 0:
            raise ValueError("Window size must be positive")
        if self.config.min_bars_per_episode <= 0:
            raise ValueError("Minimum bars per episode must be positive")
            
        # Initialize core variables
        self.predict_context = predict_mode
        self.window_size = self.config.window_size
        self.initial_balance = self.config.initial_balance
        
        self.logger.debug(
            f"Configuration: balance=${self.initial_balance:.2f}, "
            f"window={self.window_size}, mode={'prediction' if predict_mode else 'training'}"
        )
        
        # Initialize environment components
        self._init_components()
        
        # Process and validate data
        self._process_market_data(data)
        
        # Set up spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.feature_processor.setup_observation_space(window_size=self.window_size)

    def _init_components(self) -> None:
        """Initialize environment components."""
        self.feature_processor = FeatureProcessor()
        self.metrics = MetricsTracker(self.initial_balance)
        self.action_handler = ActionHandler(self)
        self.renderer = Renderer()
        self.reward_calculator = RewardCalculator(self)

    def _process_market_data(self, data: pd.DataFrame) -> None:
        """Process and validate market data."""
        
        # Verify required columns
        required_columns = ['open', 'close', 'high', 'low', 'spread', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Process data and setup spaces
        self.features_df, self.atr_values, aligned_index = self.feature_processor.preprocess_data(data)
        self.action_space = spaces.Discrete(4)
        self.observation_space = self.feature_processor.setup_observation_space(window_size=self.window_size)
        
        # Save aligned datetime index and data length
        self.original_index = aligned_index
        self.data_length = len(self.features_df)
        
        # Store aligned price data
        self.prices = {
            'close': data.loc[aligned_index, 'close'].values,
            'high': data.loc[aligned_index, 'high'].values,
            'low': data.loc[aligned_index, 'low'].values,
            'spread': data.loc[aligned_index, 'spread'].values,
            'atr': self.atr_values
        }
        
        # Add validation check for alignment
        if any(len(price_data) != self.data_length for price_data in self.prices.values()):
            raise ValueError("Price data arrays must have same length as feature data")
        
        # Calculate starting point ensuring enough historical data
        self.start_step = max(self.feature_processor.lookback, self.window_size - 1)
        min_data_required = self.start_step + self.config.min_bars_per_episode
        
        if self.data_length < min_data_required:
            raise ValueError(
                f"Insufficient data: got {self.data_length} bars, need at least {min_data_required} "
                f"(lookback: {self.feature_processor.lookback}, window: {self.window_size}, "
                f"min episode: {self.config.min_bars_per_episode})"
            )
            
        self.logger.debug(
            f"Data processed successfully: {self.data_length} bars available, "
            f"starting from index {self.start_step}"
        )
        
        # Initialize to starting state
        self._reset_state()
        
    def _reset_state(self) -> None:
        """Reset environment state variables."""
        self.current_step = self.start_step
        self.episode_steps = 0
        self.completed_episodes = 0
        self.current_position = None
        self.trades = []
        self.trade_metrics = {'current_direction': 0}
        self.current_hold_time = 0
        
    def _handle_trade_action(self, action: int, current_spread: float) -> Tuple[float, bool, float]:
        """Handle trade action and return associated PnL and validity."""
        invalid_action = False
        pnl = 0.0
        unrealized_pnl = 0.0        
        
        # Validate action
        if action in [Action.BUY, Action.SELL] and self.current_position is not None:
            invalid_action = True
            self.logger.debug("Invalid action: Attempting to open position when one exists")
        elif action == Action.CLOSE and self.current_position is None:
            invalid_action = True
            self.logger.debug("Invalid action: Attempting to close non-existent position")
            
        # Execute valid actions
        if not invalid_action:
            if action in [Action.BUY, Action.SELL]:
                if self.current_position is None:
                    direction = 1 if action == Action.BUY else 2
                    self.action_handler.execute_trade(direction, current_spread)
                    self.current_hold_time = 0
                    self.logger.debug(f"Opened {'long' if direction == 1 else 'short'} position")
                    
            elif action == Action.CLOSE and self.current_position:
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
                    self.logger.debug(f"Closed position with PnL: {pnl:.2f}")
                self.current_position = None
                self.current_hold_time = 0
        
        # Update unrealized PnL for active position
        if self.current_position:
            if self.predict_context and "profit" in self.current_position:
                unrealized_pnl = self.current_position["profit"]
                profit_points = self.current_position.get("profit_points", 0.0)
            else:
                unrealized_pnl, profit_points = self.action_handler.manage_position()
            self.current_position["current_profit_points"] = profit_points
            self.metrics.update_unrealized_pnl(unrealized_pnl)
        else:
            self.metrics.update_unrealized_pnl(0.0)
            
        return pnl, invalid_action, unrealized_pnl

    def _get_terminal_state(self) -> Tuple[bool, float]:
        """Check if episode should terminate and handle terminal state."""
        end_of_data = self.current_step >= self.data_length - 1
        max_drawdown = self.metrics.get_equity_drawdown()
        bankrupt = self.metrics.balance <= 0
        drawdown_exceeded = max_drawdown >= self.config.max_drawdown
        
        done = end_of_data or bankrupt or drawdown_exceeded
        terminal_reward = 0.0
        
        if done:
            # Log termination reason
            if bankrupt:
                self.logger.debug("Episode terminated due to bankruptcy")
            elif drawdown_exceeded:
                self.logger.debug(f"Episode terminated due to max drawdown: {max_drawdown:.2%}")
            elif end_of_data:
                self.logger.debug("Episode completed normally")
            
            # Close any open position
            if self.current_position:
                pnl, trade_info = self.action_handler.close_position()
                if pnl != 0:
                    self.trades.append(trade_info)
                    self.metrics.add_trade(trade_info)
                    self.metrics.update_balance(pnl)
                    self.logger.debug(f"Closed final position with PnL: {pnl:.2f}")
                self.current_hold_time = 0
            
            terminal_reward = self.reward_calculator.calculate_terminal_reward(
                self.metrics.balance, self.initial_balance
            )
            
        return done, terminal_reward
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Pre-process action and update state
        action = self.action_handler.process_action(action)
        current_spread = self.prices['spread'][self.current_step] * self.config.point_value
        
        self.episode_steps += 1
        self.current_step += 1
        if self.current_position:
            self.current_hold_time += 1
            
        # Process trade
        pnl, invalid_action, unrealized_pnl = self._handle_trade_action(action, current_spread)
        
        # Calculate reward components
        reward = self.reward_calculator.calculate_reward(
            action=action,
            position_type=self.current_position["direction"] if self.current_position else 0,
            pnl=pnl if action == Action.CLOSE else unrealized_pnl,
            atr=self.prices['atr'][self.current_step],
            current_hold=self.current_hold_time,
            optimal_hold=None,
            invalid_action=invalid_action
        )
        
        # Check terminal state
        done, terminal_reward = self._get_terminal_state()
        reward += terminal_reward
        
        # Get final state
        obs = self.get_observation()
        truncated = self.current_step >= self.data_length - 1
        
        return obs, float(reward), done, truncated, self._get_info()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Optional random seed (ignored)
            options: Optional configuration dictionary
            
        Returns:
            Tuple of (initial observation, environment info)
        """
        self._reset_state()
        self.completed_episodes += 1
        
        # Reset component states
        self.metrics.reset(self.initial_balance)
        
        # Reset reward calculator state
        self.reward_calculator.reset(
            initial_balance=self.initial_balance,
            min_bars=self.config.min_bars_per_episode
        )
        
        self.logger.debug(f"Environment reset (episode {self.completed_episodes})")
        
        # Get initial observation and info
        obs = self.get_observation()
        info = self._get_info()
        
        return obs, info
        
    def render(self) -> None:
        """Render the environment."""
        self.renderer.render_episode_stats(self)
        
    def _get_market_features(self) -> np.ndarray:
        """Get market features window."""
        market_feature_names = self.feature_processor.get_feature_names()[:-3]
        
        if self.predict_context:
            # For live prediction
            features = self.features_df[market_feature_names].values[-self.window_size:]
            if len(features) < self.window_size:
                padding = np.zeros((self.window_size - len(features), len(market_feature_names)))
                features = np.vstack((padding, features))
        else:
            # For backtesting/training
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            features = self.features_df[market_feature_names].iloc[start_idx:end_idx].values
            
        return features.flatten()
        
    def _get_position_features(self) -> np.ndarray:
        """Get current position state features."""
        # Position type
        position_type = self.current_position["direction"] if self.current_position else 0
        
        # Normalized PnL
        if self.current_position:
            if self.predict_context and "profit" in self.current_position:
                unrealized_pnl = self.current_position["profit"]
            else:
                unrealized_pnl, _ = self.action_handler.manage_position()
            # Ensure non-zero denominator
            current_balance = max(self.metrics.balance, self.initial_balance)
            normalized_pnl = np.clip(unrealized_pnl / current_balance, -1, 1)
        else:
            normalized_pnl = 0.0
        
        # Normalized hold time
        normalized_hold_time = (
            min(self.current_hold_time / self.reward_calculator.TIME_PRESSURE_THRESHOLD, 1.0)
            if self.current_position else 0.0
        )
        
        return np.array([position_type, normalized_pnl, normalized_hold_time], dtype=np.float32)
    
    def get_observation(self) -> np.ndarray:
        """Get current observation, combining market and position features."""
        market_features = self._get_market_features()
        position_features = self._get_position_features()
        return np.concatenate((market_features, position_features))
        
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
