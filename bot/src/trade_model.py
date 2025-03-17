"""Models for trading prediction using reinforcement learning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from trade_environment import TradingEnv, ActionType


class TradeModel:
    """Class for loading and making predictions with a trained DQN model."""
    
    def __init__(self, model_path: str, bar_count: int = 50, normalization_window: int = 100):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            bar_count: Number of bars in each observation
            normalization_window: Window size for normalization
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.bar_count = bar_count
        self.normalization_window = normalization_window
        self.model = None
        self.required_columns = [
            'open', 'high', 'low', 'close', 'spread', 'volume', 
            'EMA_fast', 'EMA_slow', 'RSI', 'ATR', 'OBV', 'VWAP'
        ]
        
        # Load the model
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the pre-trained DQN model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.model = DQN.load(self.model_path)
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
        
        # Create a temporary environment to get the observation
        env = TradingEnv(
            data=data,
            bar_count=self.bar_count,
            normalization_window=self.normalization_window,
            random_start=False
        )
        
        # Get the observation
        observation = env.get_history()
        
        # Make prediction
        action, _states = self.model.predict(observation, deterministic=True)
        
        # Decode the action
        action_type, rrr_idx, risk_reward_ratio = env._decode_action(action)        
        
        # Create prediction result
        result = {
            'action': int(action),
            'action_type': action_type.name,
            'risk_reward_ratio': float(risk_reward_ratio),
            'atr': float(data.iloc[-1]['ATR'])
        }

        self.logger.debug(f"Prediction: {result}")
        return result
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a backtest with the model.
        
        Args:
            data: DataFrame with market data
            
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
            bar_count=self.bar_count,
            normalization_window=self.normalization_window,
            random_start=False
        )
        
        # Run backtest
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            step += 1
            
        return self._calculate_backtest_metrics(env)

    def _calculate_backtest_metrics(self, env: TradingEnv) -> Dict[str, Any]:
        """
        Calculate metrics from backtest results.
        
        Args:
            env: Trading environment with completed backtest
            
        Returns:
            Dictionary with calculated metrics
        """
        if not env.trades:
            return {
                'final_balance': float(env.balance),
                'initial_balance': float(env.initial_balance),
                'return_pct': 0.0,
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'trades': []
            }
            
        # Calculate profit metrics
        winning_trades = [t['pnl'] for t in env.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in env.trades if t['pnl'] < 0]
        
        avg_profit = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
        
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'final_balance': float(env.balance),
            'initial_balance': float(env.initial_balance),
            'return_pct': float(((env.balance / env.initial_balance) - 1) * 100),
            'total_trades': len(env.trades),
            'win_count': env.win_count,
            'loss_count': env.loss_count,
            'win_rate': float((env.win_count / len(env.trades)) * 100) if env.trades else 0.0,
            'avg_profit': float(avg_profit),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'trades': env.trades
        }