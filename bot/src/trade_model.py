import pandas as pd
import numpy as np
import logging
from stable_baselines3 import DQN

from trade_environment import TradingEnv, ActionType

class TradeModel:
    """Class for loading and making predictions with a trained DQN model."""
    
    def __init__(self, model_path, bar_count=50, normalization_window=100):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            bar_count: Number of bars in each observation
            normalization_window: Window size for normalization
        """
        self.model_path = model_path
        self.bar_count = bar_count
        self.normalization_window = normalization_window
        self.model = None
        
        # Load the model
        self.load_model()
        
    def load_model(self):
        """Load the pre-trained DQN model."""
        try:
            self.model = DQN.load(self.model_path)
            print(f"Model successfully loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def prepare_data(self, data):
        """
        Prepare data for the model.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with all necessary columns for the model
        """
        # Check if all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'spread', 'volume', 'EMA_fast', 'EMA_slow', 'RSI', 'ATR', 'OBV', 'VWAP']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")
            
        return data
        
    def predict_single(self, data_frame):
        """
        Make a prediction for the latest data point.
        
        Args:
            data_frame: DataFrame with market data
            
        Returns:
            Dictionary with prediction details
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

        logging.debug(f"Prediction: {result}")
        
        return result
    
    def backtest(self, data):
        """
        Run a backtest with the model.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with backtest results and trade history
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
            
        # Calculate metrics
        if len(env.trades) > 0:
            avg_profit = np.mean([t['pnl'] for t in env.trades if t['pnl'] > 0]) if env.win_count > 0 else 0
            avg_loss = abs(np.mean([t['pnl'] for t in env.trades if t['pnl'] < 0])) if env.loss_count > 0 else 0
            profit_factor = (sum(t['pnl'] for t in env.trades if t['pnl'] > 0) / 
                           abs(sum(t['pnl'] for t in env.trades if t['pnl'] < 0))) if env.loss_count > 0 else float('inf')
        else:
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
            
        results = {
            'final_balance': float(env.balance),
            'initial_balance': float(env.initial_balance),
            'return_pct': float(((env.balance / env.initial_balance) - 1) * 100),
            'total_trades': len(env.trades),
            'win_count': env.win_count,
            'loss_count': env.loss_count,
            'win_rate': float((env.win_count / max(1, len(env.trades))) * 100),
            'avg_profit': float(avg_profit),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'trades': env.trades
        }
        
        return results