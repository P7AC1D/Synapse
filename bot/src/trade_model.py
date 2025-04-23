"""Models for trading prediction using reinforcement learning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO

from trading.environment import TradingEnv
from trading.actions import Action

class TradeModel:
    """Class for loading and making predictions with a trained PPO-LSTM model."""
    
    def __init__(self, model_path: str, balance_per_lot: float = 1000.0, initial_balance: float = 10000.0):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            balance_per_lot: Account balance required per 0.01 lot (default: 1000.0)
            initial_balance: Starting account balance (default: 10000.0)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model = None
        self.balance_per_lot = balance_per_lot
        self.initial_balance = initial_balance
        self.required_columns = [
            'open',   # Required for price action features
            'close',  # Price data
            'high',   # Price data
            'low',    # Price data
            'spread'  # Price data
        ]
        
        self.lstm_states = None  # Store LSTM states between predictions
        self.initial_warmup = 120  # Match backtest warmup period
        self.window_size = 200  # Match backtest rolling window
        
        # Load the model
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the pre-trained PPO-LSTM model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load the PPO model with saved hyperparameters
            self.model = RecurrentPPO.load(
                self.model_path,
                print_system_info=False
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
        
        if missing_columns:
            error_msg = f"Data is missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Make sure we're not passing an empty dataset
        if len(data) < 2:
            raise ValueError(f"Data must contain at least 2 rows, got {len(data)}")
            
        return data
        
    def predict_single(self, data: pd.DataFrame, current_position: Optional[Dict] = None, 
                      verbose: bool = False) -> Dict[str, Any]:
        """Make a single prediction for live trading.
        
        Args:
            data: DataFrame with market data
            current_position: Optional dictionary with current position info
            verbose: Whether to log detailed feature values
            
        Returns:
            Dictionary with prediction and description
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare data and environment
        data = self.prepare_data(data)
        
        # Make sure we have enough data for the window
        if len(data) < self.window_size:
            raise ValueError(f"Insufficient data: need at least {self.window_size} bars, got {len(data)}")
            
        # Use the last window_size bars for prediction
        window_data = data.iloc[-self.window_size:].copy()
        
        # Initialize states if needed
        if self.lstm_states is None and len(data) >= self.initial_warmup:
            self.logger.info("Initializing LSTM states with warmup data")
            warmup_data = data.iloc[:self.initial_warmup].copy()
            self.preload_states(warmup_data)
            
        # Create environment
        env = TradingEnv(
            data=window_data,
            initial_balance=self.initial_balance,
            balance_per_lot=self.balance_per_lot,
            random_start=False
        )
        obs, _ = env.reset()
        
        # Get current position type
        position_type = 0
        if current_position:
            position_type = current_position.get('direction', 0)
            env.current_position = current_position
        
        # Make prediction with state maintenance
        action, new_lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            deterministic=True  # Use deterministic mode for consistency
        )
        self.lstm_states = new_lstm_states
        
        # Generate prediction description
        description = self._generate_prediction_description(int(action), position_type)
        
        return {
            'action': int(action),
            'description': description
        }
        
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
            
    def reset_states(self) -> None:
        """Reset the LSTM states. Call this when starting a new prediction sequence."""
        self.lstm_states = None
        
    def preload_states(self, data: pd.DataFrame) -> None:
        """
        Preload LSTM states with historical data.
        
        Args:
            data: DataFrame with market data for warmup
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Create environment for state preloading
        env = TradingEnv(
            data=data,
            initial_balance=self.initial_balance,
            balance_per_lot=self.balance_per_lot,
            random_start=False
        )
        obs, _ = env.reset()
        
        # Run prediction steps to initialize states
        while True:
            # Action doesn't matter for preloading, we only care about state updates
            _, new_lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                deterministic=True
            )
            self.lstm_states = new_lstm_states  # Update states
            obs, _, done, truncated, _ = env.step(0)  # Use 0 (hold) to minimize impact
            if done or truncated:
                break
        self.logger.info(f"LSTM states preloaded with {len(data)} historical bars")
        
    def backtest(self, data: pd.DataFrame, initial_balance: float = 10000.0, balance_per_lot: float = 1000.0) -> Dict[str, Any]:
        """
        Run a backtest with the model.
        
        Args:
            data: DataFrame with market data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot
            
        Returns:
            Dictionary with backtest results and trade history
        
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data
        data = self.prepare_data(data)
        
        # Create environment for backtesting
        env = TradingEnv(
            data=data,
            initial_balance=initial_balance,
            balance_per_lot=balance_per_lot,
            random_start=False
        )
        
        # Initialize LSTM states
        lstm_states = None
        obs, _ = env.reset()
        
        # Do an initial prediction to get first LSTM states
        _, lstm_states = self.model.predict(
            obs,
            state=lstm_states,
            deterministic=True
        )
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            raw_action, new_lstm_states = self.model.predict(
                obs, 
                state=lstm_states,
                deterministic=True
            )
            lstm_states = new_lstm_states  # Update LSTM states
            
            # Process action (0=hold, 1=buy, 2=sell, 3=close)
            discrete_action = int(raw_action.item()) if isinstance(raw_action, np.ndarray) else int(raw_action)
            discrete_action = discrete_action % 4  # Ensure in range 0-3
            # Log pre-step information
            # Format features with better readability
            feature_names = ['returns', 'rsi', 'atr', 'volatility_breakout', 'trend_strength', 
                           'candle_pattern', 'sin_time', 'cos_time', 'position_type', 
                           'normalized_hold_time', 'normalized_pnl']
            feature_dict = dict(zip(feature_names, obs))
            feature_str = "\n    ".join([f"{k}: {v:.4f}" for k, v in feature_dict.items()])
            
            self.logger.info(f"\n=== Step {step} ===")
            self.logger.info(f"Price: {data.iloc[env.current_step]['close']:.2f}")
            # Format position information with safety checks
            position_status = "None"
            if env.current_position:
                direction = "Long" if env.current_position['direction'] == 1 else "Short"
                pnl = env._manage_position()  # Get current unrealized P&L
                hold_time = env.current_step - env.current_position['entry_step']
                position_status = (f"{direction} @ {env.current_position['entry_price']:.5f} "
                                 f"(P&L: {pnl:+.2f}, Hold: {hold_time} bars)")
            self.logger.info(f"Position: {position_status}")

            # Show detailed action context
            self.logger.info(f"Features:\n    {feature_str}")
            
            # Log action with context
            action_desc = {0: "Hold", 1: "Buy", 2: "Sell", 3: "Close"}[discrete_action]
            raw_action_value = float(raw_action.item()) if isinstance(raw_action, np.ndarray) else float(raw_action)
            
            if env.current_position:
                self.logger.info(f"Action: {action_desc} (raw: {raw_action_value:.4f}) [current: {position_status}]")
            else:
                self.logger.info(f"Action: {action_desc} (raw: {raw_action_value:.4f})")
            
            # Execute step with proper action handling
            obs, reward, done, _, info = env.step(discrete_action)  # Pass action directly to env
            
            # Log post-step results with more detail
            self.logger.info("\nResults:")
            self.logger.info(f"  Balance: {info['balance']:.2f} ({'+' if info['balance'] > env.previous_balance else ''}{info['balance'] - env.previous_balance:.2f})")
            self.logger.info(f"  Total Return: {((info['balance'] - env.initial_balance) / env.initial_balance * 100):.2f}%")
            self.logger.info(f"  Reward: {reward:.4f}")
            if info['position']:
                self.logger.info(f"  Position Update: {info['position']}")
            total_reward += reward
            step += 1
            
        return self._calculate_backtest_metrics(env, step, total_reward)

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
            'avg_win_pips': 0.0,
            'avg_loss_pips': 0.0,
            'median_win_pips': 0.0,
            'median_loss_pips': 0.0,
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
                        'win_hold_time_1st': float(winning_trades['hold_time'].quantile(0.01)),
                        'win_hold_time_10th': float(winning_trades['hold_time'].quantile(0.1)),
                        'win_hold_time_median': float(winning_trades['hold_time'].median()),
                        'win_hold_time_90th': float(winning_trades['hold_time'].quantile(0.9)),
                        'win_hold_time_99th': float(winning_trades['hold_time'].quantile(0.99))
                    })
                    
                if not losing_trades.empty and 'hold_time' in losing_trades.columns:
                    metrics.update({
                        'loss_hold_time': float(losing_trades['hold_time'].mean()),
                        'loss_hold_time_1st': float(losing_trades['hold_time'].quantile(0.01)),
                        'loss_hold_time_10th': float(losing_trades['hold_time'].quantile(0.1)),
                        'loss_hold_time_median': float(losing_trades['hold_time'].median()),
                        'loss_hold_time_90th': float(losing_trades['hold_time'].quantile(0.9)),
                        'loss_hold_time_99th': float(losing_trades['hold_time'].quantile(0.99))
                    })
        
        # Include trade history
        metrics['trades'] = env.trades

        # Calculate pips metrics including averages, medians, and 90th percentiles
        if not winning_trades.empty:
            metrics.update({
                'avg_win_pips': float(winning_trades["profit_pips"].mean()),
                'win_pips_1st': float(winning_trades["profit_pips"].quantile(0.01)),
                'win_pips_10th': float(winning_trades["profit_pips"].quantile(0.1)),
                'median_win_pips': float(winning_trades["profit_pips"].median()),
                'win_pips_90th': float(winning_trades["profit_pips"].quantile(0.9)),
                'win_pips_99th': float(winning_trades["profit_pips"].quantile(0.99))
            })
        
        if not losing_trades.empty:
            # Calculate stats on absolute values, then make negative
            abs_loss_pips = losing_trades["profit_pips"].abs()
            metrics.update({
                'avg_loss_pips': -float(abs_loss_pips.mean()),
                'loss_pips_1st': -float(abs_loss_pips.quantile(0.01)),
                'loss_pips_10th': -float(abs_loss_pips.quantile(0.1)),
                'median_loss_pips': -float(abs_loss_pips.median()),
                'loss_pips_90th': -float(abs_loss_pips.quantile(0.9)),
                'loss_pips_99th': -float(abs_loss_pips.quantile(0.99))
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

    def evaluate(self, data: pd.DataFrame, initial_balance: float = 10000.0, balance_per_lot: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate model performance without verbose logging.
        
        Args:
            data: DataFrame with market data
            initial_balance: Starting account balance
            balance_per_lot: Account balance required per 0.01 lot (if None, uses instance default)
                
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
            random_start=False
        )
        
        # Initialize LSTM states
        lstm_states = None
        obs, _ = env.reset()
        
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            # Make prediction
            raw_action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
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
