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
            'close', 'high', 'low', 'spread',  # Price data
            'EMA_fast', 'EMA_slow', 'RSI', 'ATR'  # Indicators for our simplified features
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
            # Create a temporary environment for model loading with dummy data
            dummy_data = pd.DataFrame({
                'close': [1.0] * 100,
                'high': [1.0] * 100,
                'low': [1.0] * 100,
                'spread': [0.0001] * 100,
                'EMA_fast': [1.0] * 100,
                'EMA_slow': [1.0] * 100,
                'RSI': [50.0] * 100,
                'ATR': [0.001] * 100
            })
            
            env = TradingEnv(
                data=dummy_data,
                random_start=False,
                balance_per_lot=1000.0  # Match training environment
            )
            
            # Define custom objects needed for model loading
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "clip_range_vf": None,
            }
            
            # Load the PPO model with custom objects
            self.model = RecurrentPPO.load(
                self.model_path,
                env=env,
                custom_objects=custom_objects,
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
            
        # Prepare data
        data = self.prepare_data(data_frame)
        
        # Create a temporary environment with simplified features
        env = TradingEnv(
            data=data,
            random_start=False,
            balance_per_lot=1000.0  # Match training environment
        )
        
        # Get the observation
        observation = env.get_history()
        
        # Make prediction with LSTM state management
        action, self.lstm_states = self.model.predict(
            observation, 
            state=self.lstm_states,
            deterministic=True
        )
        
        # Process single action value for position
        position = np.sign(action[0]) if abs(action[0]) > 0.1 else 0
        
        # Get ATR and current market conditions
        current_atr = float(data.iloc[-1]['ATR'])
        
        # Extract grid size from action
        grid_multiplier = float(np.clip(action[1], 0.1, 3.0))  # Match TradingEnv limits
        grid_size_pips = current_atr * grid_multiplier
        
        # Create prediction result with grid parameters
        result = {
            'position': int(position),  # -1 for sell, 0 for hold, 1 for buy
            'grid_size_pips': grid_size_pips,
            'grid_multiplier': grid_multiplier,
            'atr': current_atr
        }

        self.logger.debug(f"Prediction: {result}")
        return result
    
    def reset_states(self) -> None:
        """Reset the LSTM states. Call this when starting a new prediction sequence."""
        self.lstm_states = None
    
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
        
        # Reset LSTM states for backtest
        lstm_states = None
        
        # Run backtest
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
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
            
            # Calculate position metrics
            metrics['long_trades'] = len(trades_df[trades_df['direction'] == 1])
            metrics['short_trades'] = len(trades_df[trades_df['direction'] == -1])
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            # Calculate PnL metrics
            metrics['total_profit'] = float(winning_trades['pnl'].sum()) if not winning_trades.empty else 0.0
            metrics['total_loss'] = float(abs(losing_trades['pnl'].sum())) if not losing_trades.empty else 0.0
            metrics['profit_factor'] = metrics['total_profit'] / metrics['total_loss'] if metrics['total_loss'] > 0 else float('inf')
            metrics['expected_value'] = float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0
            
            # Calculate drawdown
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
