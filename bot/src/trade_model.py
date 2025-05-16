"""Models for trading prediction using reinforcement learning."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO

from trading.environment import TradingEnv, TradingConfig
from trading.actions import Action

@dataclass
class ModelConfig:
    """Configuration for trade model."""
    model_path: Path
    balance_per_lot: float = 1000.0
    initial_balance: float = 10000.0
    point_value: float = 0.01
    min_lots: float = 0.01
    max_lots: float = 200.0
    contract_size: float = 100.0
    window_size: int = 50
    warmup_window: int = 100
    device: str = 'cpu'

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.model_path.exists():
            raise ValueError(f"Model file not found: {self.model_path}")
        if self.balance_per_lot <= 0:
            raise ValueError("balance_per_lot must be positive")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.min_lots <= 0 or self.max_lots <= 0:
            raise ValueError("Lot sizes must be positive")
        if self.min_lots > self.max_lots:
            raise ValueError("min_lots cannot be greater than max_lots")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.warmup_window <= 0:
            raise ValueError("warmup_window must be positive")

class TradeModel:
    """Class for loading and making predictions with a trained PPO model."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the trade model.
        
        Args:
            config: Model configuration object
        """
        config.validate()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing trade model with {config.model_path}")
        
        # Initialize model state
        self.model = None
        # State management
        self.lstm_states = None
        self.is_recurrent = True
        
        # Required data columns - could be moved to config if needs to be customizable
        self.required_columns = {
            'open': 'Required for price action features',
            'close': 'Price data',
            'high': 'Price data',
            'low': 'Price data',
            'spread': 'Price data'
        }
        
        # Initialize model
        try:
            self._initialize_model()
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
            
    def _initialize_model(self) -> None:
        """Initialize and load the PPO model."""
        self.logger.debug(f"Loading model from {self.config.model_path}")
        
        try:
            self.model = RecurrentPPO.load(
                self.config.model_path,
                print_system_info=False,
                device=self.config.device
            )
            
            # Verify model architecture
            if not hasattr(self.model, 'lstm_hidden_size'):
                raise ValueError("Model does not have LSTM architecture")
                
            # Reset LSTM states
            self.lstm_states = None
            self.logger.debug("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise ValueError(f"Model loading failed: {str(e)}")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the model by validating required columns.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with all necessary columns for the model
        
        Raises:
            ValueError: If data is missing required columns
        """
        # Check if all required columns are present
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        
        if (missing_columns):
            error_msg = f"Data is missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Make sure we're not passing an empty dataset
        if len(data) < 2:
            raise ValueError(f"Data must contain at least 2 rows, got {len(data)}")
            
        return data
        
    def _generate_prediction_description(self, action: int, position_type: int) -> str:
        """Generate a human-readable description of the prediction.
        
        Args:
            action: The predicted action (0=hold, 1=buy, 2=sell, 3=close)
            position_type: Current position type (0=none, 1=long, -1=short)
            
        Returns:
            str: Description of the prediction
        """
        action_map = {0: 'hold', 1: 'buy', 2: 'sell', 3: 'close'}
        position_map = {0: 'no position', 1: 'long', -1: 'short'}
        
        base_desc = f"Model predicts {action_map[action]}"
        
        if action == 0:  # Hold
            if position_type != 0:
                return f"{base_desc} current {position_map[position_type]} position"
            return f"{base_desc} - stay out of market"
            
        elif action in [1, 2]:  # Buy or Sell
            if position_type != 0:
                return f"{base_desc} but rejected - {position_map[position_type]} position exists"
            return f"{base_desc} - open new {'long' if action == 1 else 'short'} position"
            
        else:  # Close
            if position_type != 0:
                return f"{base_desc} current {position_map[position_type]} position"
            return f"{base_desc} but rejected - no position exists"
            
    def _calculate_position_metrics(self, env: TradingEnv) -> Dict[str, Any]:
        """Calculate current position metrics."""
        position_metrics = {
            'active_positions': 1 if env.current_position is not None else 0,
        }
        
        if env.current_position:
            unrealized_pnl, profit_points = env.action_handler.manage_position()
            position_metrics['position'] = {
                'direction': env.current_position['direction'],
                'entry_price': env.current_position['entry_price'],
                'lot_size': env.current_position['lot_size'],
                'hold_time': env.current_step - env.current_position['entry_step'],
                'unrealized_pnl': unrealized_pnl,
                'profit_points': profit_points
            }
            
        return position_metrics
        
    def _calculate_streak_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade streak metrics."""
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0 and abs(pnl) >= 1e-8:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'current_consecutive_wins': current_win_streak,
            'current_consecutive_losses': current_loss_streak
        }
        
    def _calculate_hold_time_metrics(self, trades_df: pd.DataFrame, 
                                   winning_trades: pd.DataFrame, 
                                   losing_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate hold time metrics for trades."""
        metrics = {
            'avg_hold_time': float(trades_df['hold_time'].mean()),
            'median_hold_time': float(trades_df['hold_time'].median()),
        }
        
        if not winning_trades.empty and 'hold_time' in winning_trades.columns:
            metrics.update({
                'win_hold_time': float(winning_trades['hold_time'].mean()),
                'win_hold_time_0th': float(winning_trades['hold_time'].min()),
                'win_hold_time_1st': float(winning_trades['hold_time'].quantile(0.01)),
                'win_hold_time_10th': float(winning_trades['hold_time'].quantile(0.1)),
                'win_hold_time_20th': float(winning_trades['hold_time'].quantile(0.2)),
                'win_hold_time_median': float(winning_trades['hold_time'].median()),
                'win_hold_time_80th': float(winning_trades['hold_time'].quantile(0.8)),
                'win_hold_time_90th': float(winning_trades['hold_time'].quantile(0.9)),
                'win_hold_time_99th': float(winning_trades['hold_time'].quantile(0.99)),
                'win_hold_time_100th': float(winning_trades['hold_time'].max())
            })
            
        if not losing_trades.empty and 'hold_time' in losing_trades.columns:
            metrics.update({
                'loss_hold_time': float(losing_trades['hold_time'].mean()),
                'loss_hold_time_0th': float(losing_trades['hold_time'].min()),
                'loss_hold_time_1st': float(losing_trades['hold_time'].quantile(0.01)),
                'loss_hold_time_10th': float(losing_trades['hold_time'].quantile(0.1)),
                'loss_hold_time_20th': float(losing_trades['hold_time'].quantile(0.2)),
                'loss_hold_time_median': float(losing_trades['hold_time'].median()),
                'loss_hold_time_80th': float(losing_trades['hold_time'].quantile(0.8)),
                'loss_hold_time_90th': float(losing_trades['hold_time'].quantile(0.9)),
                'loss_hold_time_99th': float(losing_trades['hold_time'].quantile(0.99)),
                'loss_hold_time_100th': float(losing_trades['hold_time'].max())
            })
            
        return metrics
        
    def _calculate_points_metrics(self, winning_trades: pd.DataFrame, 
                                losing_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate profit/loss points metrics."""
        metrics = {}
        
        if not winning_trades.empty:
            metrics.update({
                'avg_win_points': float(winning_trades["profit_points"].mean()),
                'win_points_0th': float(winning_trades["profit_points"].min()),
                'win_points_1st': float(winning_trades["profit_points"].quantile(0.01)),
                'win_points_10th': float(winning_trades["profit_points"].quantile(0.1)),
                'win_points_20th': float(winning_trades["profit_points"].quantile(0.2)),
                'median_win_points': float(winning_trades["profit_points"].median()),
                'win_points_80th': float(winning_trades["profit_points"].quantile(0.8)),
                'win_points_90th': float(winning_trades["profit_points"].quantile(0.9)),
                'win_points_99th': float(winning_trades["profit_points"].quantile(0.99)),
                'win_points_100th': float(winning_trades["profit_points"].max())
            })
        
        if not losing_trades.empty:
            abs_loss_points = losing_trades["profit_points"].abs()
            metrics.update({
                'avg_loss_points': -float(abs_loss_points.mean()),
                'loss_points_0th': -float(abs_loss_points.min()),
                'loss_points_1st': -float(abs_loss_points.quantile(0.01)),
                'loss_points_10th': -float(abs_loss_points.quantile(0.1)),
                'loss_points_20th': -float(abs_loss_points.quantile(0.2)),
                'median_loss_points': -float(abs_loss_points.median()),
                'loss_points_80th': -float(abs_loss_points.quantile(0.8)),
                'loss_points_90th': -float(abs_loss_points.quantile(0.9)),
                'loss_points_99th': -float(abs_loss_points.quantile(0.99)),
                'loss_points_100th': -float(abs_loss_points.max())
            })
            
        return metrics

    def _calculate_backtest_metrics(self, env: TradingEnv, total_steps: int, total_reward: float) -> Dict[str, Any]:
        """
        Calculate metrics from backtest results.
        
        Args:
            env: Trading environment after backtest completion
            total_steps: Number of steps taken in backtest
            total_reward: Total cumulative reward from backtest
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'final_balance': float(env.balance),
            'initial_balance': float(env.initial_balance),
            'return_pct': ((env.balance / env.initial_balance) - 1) * 100,
            'total_trades': len(env.trades),
            'win_count': env.win_count,
            'loss_count': env.loss_count,
            'win_rate': 0.0,  # Will be updated if trades exist
            'total_steps': total_steps,
            'total_reward': total_reward,
            'active_positions': 1 if env.current_position is not None else 0,
        }
        
        # Add position details if there's an open position
        if env.current_position:
            unrealized_pnl, profit_points = env.action_handler.manage_position()
            metrics['position'] = {
                'direction': env.current_position['direction'],
                'entry_price': env.current_position['entry_price'],
                'lot_size': env.current_position['lot_size'],
                'hold_time': env.current_step - env.current_position['entry_step'],
                'unrealized_pnl': unrealized_pnl,
                'profit_points': profit_points
            }
        
        # Safely add win rate
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = (env.win_count / metrics['total_trades'] * 100)
        
        # Initialize optional metrics with default values
        metrics.update({
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0.0,
            'short_win_rate': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'profit_factor': 0.0,
            'expected_value': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'avg_hold_time': 0.0,
            'win_hold_time': 0.0,
            'loss_hold_time': 0.0,
            'avg_win_points': 0.0,
            'avg_loss_points': 0.0,
            'median_win_points': 0.0,
            'median_loss_points': 0.0,
        })
        
        # Only calculate detailed metrics if trades exist
        if env.trades:
            trades_df = pd.DataFrame(env.trades)
            
            # Split trades by direction
            long_trades = trades_df[trades_df['direction'] == 1]
            short_trades = trades_df[trades_df['direction'] == -1]
            
            # Calculate directional metrics - safely handle empty cases
            metrics['long_trades'] = len(long_trades)
            metrics['short_trades'] = len(short_trades)
            
            if len(long_trades) > 0:
                long_wins = long_trades[long_trades['pnl'].apply(lambda x: x > 0 and abs(x) >= 1e-8)]
                metrics['long_win_rate'] = (len(long_wins) / len(long_trades) * 100)
            
            if len(short_trades) > 0:
                short_wins = short_trades[short_trades['pnl'].apply(lambda x: x > 0 and abs(x) >= 1e-8)]
                metrics['short_win_rate'] = (len(short_wins) / len(short_trades) * 100)
            
            # Overall win/loss metrics with proper PnL thresholds
            winning_trades = trades_df[trades_df['pnl'].apply(lambda x: x > 0 and abs(x) >= 1e-8)]
            losing_trades = trades_df[trades_df['pnl'].apply(lambda x: x <= 0 or abs(x) < 1e-8)]
            
            # Calculate PnL metrics
            metrics['total_profit'] = float(winning_trades['pnl'].sum()) if not winning_trades.empty else 0.0
            metrics['total_loss'] = float(abs(losing_trades['pnl'].sum())) if not losing_trades.empty else 0.0
            
            # Safe profit factor calculation
            if metrics['total_loss'] > 0:
                metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss']
            else:
                metrics['profit_factor'] = float('inf') if metrics['total_profit'] > 0 else 0.0
                
            # Expected value
            if not trades_df.empty:
                metrics['expected_value'] = float(trades_df['pnl'].mean())
            
            # Calculate drawdowns using MetricsTracker's methods
            metrics['max_drawdown_pct'] = float(env.metrics.get_drawdown() * 100)  # Balance-based drawdown
            metrics['max_equity_drawdown_pct'] = float(env.metrics.get_max_equity_drawdown() * 100)  # Equity-based drawdown
            metrics['current_drawdown_pct'] = float(env.metrics.get_balance_drawdown() * 100)  # Current balance drawdown
            metrics['current_equity_drawdown_pct'] = float(env.metrics.get_equity_drawdown() * 100)  # Current equity drawdown (realized only)

            # Calculate current drawdown (realized only)
            if env.metrics.max_balance > 0:
                current_drawdown = (env.metrics.max_balance - env.balance) / env.metrics.max_balance
            else:
                current_drawdown = 0.0
            metrics['current_drawdown_pct'] = float(current_drawdown * 100)
            
            # For historical reference
            metrics['historical_max_drawdown_pct'] = float(metrics['max_equity_drawdown_pct'])
            
            # Calculate Sharpe ratio safely
            returns = pd.Series(trades_df['pnl']) / env.initial_balance
            if len(returns) > 1 and returns.std() > 0:
                metrics['sharpe_ratio'] = float((returns.mean() / returns.std()) * np.sqrt(252))
            
            # Calculate hold time metrics
            if 'hold_time' in trades_df.columns:
                metrics.update(self._calculate_hold_time_metrics(trades_df, winning_trades, losing_trades))
        
        # Calculate points and update metrics
        metrics.update(self._calculate_points_metrics(winning_trades, losing_trades))
        
        # Include trade history
        metrics['trades'] = env.trades
        
        # Calculate consecutive trade metrics
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        # Process trades chronologically to track streaks
        for trade in env.trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0 and abs(pnl) >= 1e-8:  # Clear win
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:  # Loss or zero PnL
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        # Add streak metrics
        metrics.update({
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'current_consecutive_wins': current_win_streak,
            'current_consecutive_losses': current_loss_streak
        })
        
        return metrics

    def evaluate(self, data: pd.DataFrame, initial_balance: Optional[float] = None,
                balance_per_lot: Optional[float] = None, spread_variation: float = 0.0,
                slippage_range: float = 0.0) -> Dict[str, Any]:
        """Evaluate model performance with backtesting.
        
        Args:
            data: DataFrame with market data
            initial_balance: Optional starting balance (defaults to config value)
            balance_per_lot: Optional balance per lot (defaults to config value)
            spread_variation: Random spread variation range (default: 0.0)
            slippage_range: Maximum price slippage range in points (default: 0.0)
                
        Returns:
            Dictionary with backtest metrics and trade summary
            
        Raises:
            ValueError: If model not loaded or data validation fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        try:
            # Validate and prepare data
            data = self.prepare_data(data)
            
            # Use config values if not overridden
            initial_balance = initial_balance or self.config.initial_balance
            balance_per_lot = balance_per_lot or self.config.balance_per_lot
            
            self.logger.info(
                f"Starting evaluation with balance=${initial_balance:.2f}, "
                f"balance_per_lot=${balance_per_lot:.2f}"
            )
            
            # Perform LSTM warm-up
            if self.is_lstm_model():
                warmup_window = min(self.config.warmup_window, len(data) // 10)
                self.warmup_lstm_state(data.iloc[:-warmup_window], warmup_window)
                self.logger.info(f"Warmed up LSTM states using {warmup_window} bars")
            
            # Create environment for evaluation
            env = TradingEnv(
                data=data,
                config=TradingConfig(
                    initial_balance=initial_balance,
                    balance_per_lot=balance_per_lot,
                    point_value=self.config.point_value,
                    min_lots=self.config.min_lots,
                    max_lots=self.config.max_lots,
                    contract_size=self.config.contract_size,
                    window_size=self.config.window_size,
                    spread_variation=spread_variation,
                    slippage_range=slippage_range
                )
            )
            
            # Run evaluation
            obs, _ = env.reset()
            done = False
            step = 0
            total_reward = 0.0
            
            while not done:
                raw_action, self.lstm_states = self.model.predict(
                    obs,
                    state=self.lstm_states,
                    deterministic=True
                )
                
                # Process action with improved error handling
                try:
                    if isinstance(raw_action, np.ndarray):
                        action_value = int(raw_action.item())
                    else:
                        action_value = int(raw_action)
                    discrete_action = action_value % 4
                    
                    if env.current_position is not None and discrete_action in [Action.BUY, Action.SELL]:
                        discrete_action = Action.HOLD 
                except (ValueError, TypeError):
                    discrete_action = 0
                
                # Execute step
                obs, reward, done, _, _ = env.step(discrete_action)
                total_reward += reward
                step += 1
            
            # Calculate final metrics
            metrics = self._calculate_backtest_metrics(env, step, total_reward)
            self.logger.info(
                f"Evaluation completed: {metrics['total_trades']} trades, "
                f"return {metrics['return_pct']:.2f}%, "
                f"win rate {metrics['win_rate']:.1f}%"
            )
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    def predict_single(self, data: pd.DataFrame, env: Optional[TradingEnv] = None) -> Dict[str, Any]:
        """
        Make a single prediction at the last data point.
        
        Args:
            data: DataFrame with market data
            env: Optional existing TradingEnv instance with position info
            
        Returns:
            Dictionary with prediction details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare data
        data = self.prepare_data(data)
        
        # Use existing environment if provided, otherwise create a new one
        if env is None:
            env = TradingEnv(
                data=data,
                config=TradingConfig(
                    initial_balance=self.config.initial_balance,
                    balance_per_lot=self.config.balance_per_lot,
                    point_value=self.config.point_value,
                    min_lots=self.config.min_lots,
                    max_lots=self.config.max_lots,
                    contract_size=self.config.contract_size,
                    window_size=self.config.window_size
                ),
                predict_mode=True,
                random_start=False
            )
            # Initialize environment
            obs, _ = env.reset()
            
        # Get observation (will include position info if env has it)
        obs = env.get_observation()
        
        # Make prediction with LSTM states
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            deterministic=True
        )
        
        # Convert to discrete action
        if isinstance(action, np.ndarray):
            action_value = int(action.item())
        else:
            action_value = int(action)
        discrete_action = action_value % 4
            
        # Get current position type for context
        current_position_type = 0
        if env.current_position:
            current_position_type = env.current_position['direction']
            
        # Generate human readable description
        description = self._generate_prediction_description(discrete_action, current_position_type)
        
        # Action mask (whether action is valid)
        valid_action = True
        
        # Check if action is valid (can't open a position if one exists, can't close without a position)
        if (current_position_type != 0 and discrete_action in [1, 2]) or (current_position_type == 0 and discrete_action == 3):
            valid_action = False
            
        # Return prediction details
        prediction = {
            'action': discrete_action,
            'action_raw': action_value,
            'description': description,
            'valid_action': valid_action,
            'position_type': current_position_type,
            'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None
        }
            
        return prediction

    def reset_lstm_states(self) -> None:
        """Reset LSTM states to None. Should be called when market gaps are detected or starting new sequences."""
        self.lstm_states = None
        self.logger.debug("LSTM states reset")
        
    def warmup_lstm_state(self, data: pd.DataFrame, warmup_window: int = 100) -> None:
        """Warm up LSTM states using historical data.
        
        Args:
            data: DataFrame with market data
            warmup_window: Number of observations to use for warm-up (default: 100)
        """
        if not self.is_lstm_model():
            return
            
        if len(data) < warmup_window:
            self.logger.warning(f"Not enough data for warm-up. Need {warmup_window} bars, got {len(data)}")
            warmup_window = len(data)
            
        # Create environment for warm-up
        env = TradingEnv(
            data=data.iloc[-warmup_window:],
            config=TradingConfig(
                initial_balance=self.config.initial_balance,
                balance_per_lot=self.config.balance_per_lot,
                point_value=self.config.point_value,
                min_lots=self.config.min_lots,
                max_lots=self.config.max_lots,
                contract_size=self.config.contract_size,
                window_size=self.config.window_size
            ),
            predict_mode=True
        )
        
        # Initialize environment
        obs, _ = env.reset()
        
        # Run warm-up sequence
        self.lstm_states = None  # Start fresh
        for _ in range(warmup_window):
            _, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                deterministic=True
            )
            obs, _, done, _, _ = env.step(0)  # Use HOLD action during warm-up
            if done:
                break
                
        self.logger.debug(f"LSTM states warmed up using {warmup_window} observations")

    def is_lstm_model(self) -> bool:
        """Check if this is a recurrent LSTM model.
        
        Returns:
            bool: True if model is recurrent LSTM, False otherwise
        """
        return self.is_recurrent and hasattr(self.model, 'lstm_hidden_size')
