"""Models for trading prediction using reinforcement learning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO

from trade_environment import TradingEnv

class TradeModel:
    """Class for loading and making predictions with a trained PPO-LSTM model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.model = None
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
        
        if missing_columns:
            error_msg = f"Data is missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        return data
        
    def predict_single(self, data_frame: pd.DataFrame) -> Dict[str, Any]:
        """
        Make a prediction for the latest data point.
        
        Args:
            data_frame: DataFrame with market data
            
        Returns:
            Dictionary with prediction details
        
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
        
        # Create a temporary environment with simplified features
        env = TradingEnv(
            data=data,
            random_start=False,
            balance_per_lot=1000.0  # Match training environment
        )
        
        # Get normalized observation
        observation = env.get_history()
        
        # Make prediction with LSTM state management and proper deterministic setting
        action, self.lstm_states = self.model.predict(
            observation, 
            state=self.lstm_states,
            deterministic=True     # Use deterministic for backtesting
        )
        
        # Process action (0=hold, 1=buy, 2=sell, 3=close)
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
        Preload LSTM states with historical data.
        
        Args:
            historical_data: DataFrame with past market data
            
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
            balance_per_lot=1000.0  # Match training environment
        )
        
        # Step through historical data to build up LSTM state
        obs, _ = env.reset()
        for _ in range(len(data)):
            # Action doesn't matter for preloading, we only care about state updates
            _, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                deterministic=True
            )
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
        
        # Preload LSTM states with initial data
        preload_bars = min(250, len(data) // 4)  # Use 25% of data or 250 bars, whichever is smaller
        if preload_bars > 0:
            preload_data = data.iloc[:preload_bars]
            self.preload_states(preload_data)
            lstm_states = self.lstm_states
        else:
            lstm_states = None
        
        # Reset states and run backtest
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            action, lstm_states = self.model.predict(
                obs, 
                state=lstm_states,
                deterministic=True
            )
            # Process action (0=hold, 1=buy, 2=sell, 3=close)
            discrete_action = int(action) % 4
            self.logger.info(f"Step {step}:")
            self.logger.info(f"  Observation: {obs}")
            self.logger.info(f"  Action: {discrete_action} (0=hold,1=buy,2=sell,3=close)")
            self.logger.info(f"  Price: {data.iloc[env.current_step]['close']:.2f}")
            obs, reward, done, _, _ = env.step(discrete_action)
            total_reward += reward
            step += 1
            
        return self._calculate_backtest_metrics(env, step, total_reward)

    def _calculate_backtest_metrics(self, env: TradingEnv, total_steps: int, total_reward: float) -> Dict[str, Any]:
        """Calculate metrics from backtest results."""
        metrics = {
            'final_balance': float(env.balance),
            'initial_balance': float(env.initial_balance),
            'return_pct': ((env.balance / env.initial_balance) - 1) * 100,
            'total_trades': len(env.trades),
            'win_count': env.win_count,
            'loss_count': env.loss_count,
            'win_rate': (env.win_count / len(env.trades) * 100) if env.trades else 0.0,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'active_positions': len(env.positions),
            'grid_metrics': env.grid_metrics,
            'trades': env.trades
        }
        
        if env.trades:
            trades_df = pd.DataFrame(env.trades)
            
            # Split trades by direction
            long_trades = trades_df[trades_df['direction'] == 1]
            short_trades = trades_df[trades_df['direction'] == -1]
            long_wins = long_trades[long_trades['pnl'] > 0]
            short_wins = short_trades[short_trades['pnl'] > 0]
            
            # Calculate directional metrics
            metrics['long_trades'] = len(long_trades)
            metrics['short_trades'] = len(short_trades)
            metrics['long_win_rate'] = (len(long_wins) / len(long_trades) * 100) if len(long_trades) > 0 else 0.0
            metrics['short_win_rate'] = (len(short_wins) / len(short_trades) * 100) if len(short_trades) > 0 else 0.0
            
            # Overall win/loss metrics
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            # Calculate PnL metrics
            metrics['total_profit'] = float(winning_trades['pnl'].sum()) if not winning_trades.empty else 0.0
            metrics['total_loss'] = float(abs(losing_trades['pnl'].sum())) if not losing_trades.empty else 0.0
            metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] > 0 else float('inf')
            metrics['expected_value'] = float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0
            
            # Initialize default metrics for zero-trade case
            metrics['max_drawdown_pct'] = 0.0
            metrics['current_drawdown_pct'] = 0.0
            metrics['historical_max_drawdown_pct'] = 0.0

            # Calculate drawdown only if there are trades
            if env.trades:
                balance_history = []
                current_balance = env.initial_balance
                peak_balance = current_balance
                max_drawdown = 0.0
                
                for trade in env.trades:
                    current_balance += trade['pnl']
                    balance_history.append(current_balance)
                    peak_balance = max(peak_balance, current_balance)
                    drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                
                metrics['max_drawdown_pct'] = float(max_drawdown * 100)
            
            # Calculate Sharpe ratio
            returns = pd.Series(trades_df['pnl']) / env.initial_balance
            metrics['sharpe_ratio'] = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 else 0.0
            
            # Hold time analysis
            metrics['avg_hold_time'] = float(trades_df['hold_time'].mean())
            metrics['win_hold_time'] = float(winning_trades['hold_time'].mean()) if not winning_trades.empty else 0.0
            metrics['loss_hold_time'] = float(losing_trades['hold_time'].mean()) if not losing_trades.empty else 0.0
        
        return metrics
