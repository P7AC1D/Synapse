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
    
    def __init__(self, model_path: str, balance_per_lot: float = 1000.0):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            balance_per_lot: Account balance required per 0.01 lot (default: 1000.0)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model = None
        self.balance_per_lot = balance_per_lot
        self.required_columns = [
            'open',   # Required for price action features
            'close',  # Price data
            'high',   # Price data
            'low',    # Price data
            'spread'  # Price data
        ]
        
        self.lstm_states = None  # Store LSTM states for continuous prediction
        
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
        
        if (missing_columns):
            error_msg = f"Data is missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        return data
        
    def predict_single(self, data_frame: pd.DataFrame, current_position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a prediction for the latest data point.
        
        Args:
            data_frame: DataFrame with market data containing at minimum:
                       'open', 'close', 'high', 'low', 'spread' columns
            current_position: Optional dictionary with current position info:
                            {
                                "direction": 1 for long/-1 for short,
                                "entry_price": float,
                                "lot_size": float,
                                "entry_step": int,
                                "entry_time": str
                            }
            
        Returns:
            Dictionary with prediction details including 'action' (0-3) and 'description'
        
        Raises:
            ValueError: If model not loaded or data preparation fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # If no LSTM states exist, preload with historical data
        if self.lstm_states is None and len(data_frame) > 1:
            historical_data = data_frame.iloc[:-1]  # All but the last bar
            self.preload_states(historical_data)
            
        # Prepare data
        data = self.prepare_data(data_frame)
        
        # Create environment with position state
        env = TradingEnv(
            data=data,
            random_start=False,
            balance_per_lot=self.balance_per_lot  # Use configured parameter
        )
        
        # Set current position if provided
        if current_position:
            env.current_position = current_position.copy()  # Use copy to avoid reference issues
        
        # Get normalized observation
        observation = env.get_observation()
        
        # Make prediction with LSTM state management and proper deterministic setting
        action, self.lstm_states = self.model.predict(
            observation, 
            state=self.lstm_states,
            deterministic=True     # Use deterministic for backtesting
        )
        
        # Process action (0=hold, 1=buy, 2=sell, 3=close) with better error handling
        discrete_action = int(action) % 4
        
        # Create prediction result
        result = {
            'action': discrete_action,
            'description': ['hold', 'buy', 'sell', 'close'][discrete_action]
        }

        self.logger.debug(f"Prediction: {result}")
        return result
    
    def reset_states(self) -> None:
        """Reset the LSTM states. Call this when starting a new prediction sequence."""
        self.lstm_states = None
        
    def preload_states(self, historical_data: pd.DataFrame) -> None:
        """
        Preload LSTM states with historical data to build context.
        
        This is critical for LSTM models as they need sequential context.
        Call this before making predictions on new data to ensure proper context.
        
        Args:
            historical_data: DataFrame with past market data containing at minimum:
                             'open', 'close', 'high', 'low', 'spread' columns
            
        Raises:
            ValueError: If model not loaded or data preparation fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare and validate data
        data = self.prepare_data(historical_data)
        
        # Reset states before preloading
        self.reset_states()
        
        # Create environment for preloading
        env = TradingEnv(
            data=data,
            random_start=False,
            balance_per_lot=self.balance_per_lot  # Use configured parameter
        )
        
        # Step through historical data to build up LSTM state
        obs, _ = env.reset()
        for _ in range(len(data)):
            # Action doesn't matter for preloading, we only care about state updates
            _, new_lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                deterministic=True
            )
            self.lstm_states = new_lstm_states  # Update states
            obs, _, done, _, _ = env.step(1)
            if done:
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
            'loss_hold_time': 0.0
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
                long_wins = long_trades[long_trades['pnl'] > 0]
                metrics['long_win_rate'] = (len(long_wins) / len(long_trades) * 100)
            
            if len(short_trades) > 0:
                short_wins = short_trades[short_trades['pnl'] > 0]
                metrics['short_win_rate'] = (len(short_wins) / len(short_trades) * 100)
            
            # Overall win/loss metrics
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
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
            
            # Hold time analysis
            if 'hold_time' in trades_df.columns:
                metrics['avg_hold_time'] = float(trades_df['hold_time'].mean())
                
                if not winning_trades.empty and 'hold_time' in winning_trades.columns:
                    metrics['win_hold_time'] = float(winning_trades['hold_time'].mean())
                    
                if not losing_trades.empty and 'hold_time' in losing_trades.columns:
                    metrics['loss_hold_time'] = float(losing_trades['hold_time'].mean())
        
        # Include trade history
        metrics['trades'] = env.trades
        
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
