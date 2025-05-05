"""Models for trading prediction using reinforcement learning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from trading.environment import TradingEnv
from trading.actions import Action

class TradeModel:
    """Class for loading and making predictions with a trained PPO model."""
    
    def __init__(self, model_path: str, balance_per_lot: float = 1000.0, initial_balance: float = 10000.0,
                 point_value: float = 0.01,
                 min_lots: float = 0.01, max_lots: float = 200.0,
                 contract_size: float = 100.0):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            balance_per_lot: Account balance required per 0.01 lot (default: 1000.0)
            initial_balance: Starting account balance (default: 10000.0)
            point_value: Value of one price point movement (default: 0.01)
            min_lots: Minimum lot size (default: 0.01)
            max_lots: Maximum lot size (default: 200.0)
            contract_size: Standard contract size (default: 100.0)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model = None
        self.balance_per_lot = balance_per_lot
        self.initial_balance = initial_balance
        self.point_value = point_value
        self.min_lots = min_lots
        self.max_lots = max_lots
        self.contract_size = contract_size
        self.required_columns = [
            'open',   # Required for price action features
            'close',  # Price data
            'high',   # Price data
            'low',    # Price data
            'spread'  # Price data
        ]
        
        # Load the model
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the pre-trained PPO model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load the PPO model with saved hyperparameters
            self.model = PPO.load(
                self.model_path,
                print_system_info=False,
                device='cpu'
            )
            self.logger.info(f"Model successfully loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
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
            
            # Hold time analysis with medians and 90th percentiles
            if 'hold_time' in trades_df.columns:
                metrics.update({
                    'avg_hold_time': float(trades_df['hold_time'].mean()),
                    'median_hold_time': float(trades_df['hold_time'].median()),
                })
                
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
        
        # Include trade history
        metrics['trades'] = env.trades

        # Calculate points metrics including averages, medians, and 90th percentiles
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
            # Calculate stats on absolute values, then make negative
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

    def evaluate(self, data: pd.DataFrame, initial_balance: float = 10000.0, balance_per_lot: Optional[float] = None,
                spread_variation: float = 0.0, slippage_range: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate model performance without verbose logging.
        
        Args:
            data: DataFrame with market data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot (if None, uses instance default)
            spread_variation: Random spread variation range (default: 0.0)
            slippage_range: Maximum price slippage range in points (default: 0.0)
                
        Returns:
            Dictionary with backtest metrics and trade summary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use provided balance_per_lot or fall back to instance default
        balance_per_lot = balance_per_lot if balance_per_lot is not None else self.balance_per_lot
        
        # Prepare data
        data = self.prepare_data(data)
        
        # Create environment for evaluation
        env = TradingEnv(
            data=data,
            initial_balance=initial_balance,
            balance_per_lot=balance_per_lot,
            random_start=False,
            point_value=self.point_value,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
            contract_size=self.contract_size,
            spread_variation=spread_variation,
            slippage_range=slippage_range
        )
        
        # Initialize environment
        obs, _ = env.reset()
        
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            # Make prediction (without LSTM states)
            raw_action, _ = self.model.predict(
                obs,
                deterministic=True
            )
            
            # Process action with improved error handling
            try:
                if isinstance(raw_action, np.ndarray):
                    action_value = int(raw_action.item())
                else:
                    action_value = int(raw_action)
                discrete_action = action_value % 4
                
                # Add explicit position check and force close if needed
                if env.current_position is not None and discrete_action in [Action.BUY, Action.SELL]:
                    discrete_action = Action.HOLD 
            except (ValueError, TypeError):
                discrete_action = 0
            
            # Execute step
            obs, reward, done, _, _ = env.step(discrete_action)
            total_reward += reward
            step += 1
        
        # Calculate metrics with additional error handling
        return self._calculate_backtest_metrics(env, step, total_reward)

    def predict_single(self, data: pd.DataFrame, warmup_bars: int = 0) -> Dict[str, Any]:
        """
        Make a single prediction at the last data point.
        
        Args:
            data: DataFrame with market data
            warmup_bars: Number of bars to warm up prediction (default: 0)
            
        Returns:
            Dictionary with prediction details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare data
        data = self.prepare_data(data)
        
        # Create environment for prediction
        env = TradingEnv(
            data=data,
            initial_balance=self.initial_balance,
            balance_per_lot=self.balance_per_lot,
            random_start=False,
            point_value=self.point_value,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
            contract_size=self.contract_size
        )
        
        # Get observation of the latest bar
        obs, _ = env.reset()
        
        # Make prediction
        action, _ = self.model.predict(
            obs, 
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
            
        # Calculate potential lot size if opening a position
        if discrete_action in [1, 2] and current_position_type == 0:
            lot_size = env.action_handler.calculate_lot_size(
                balance=env.balance,
                close_price=data['close'].iloc[-1],
                atr_value=None  # Let the action handler use its default ATR value
            )
            prediction['suggested_lot_size'] = float(lot_size)
            
        return prediction
