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
    
    def __init__(self, model_path: str, bar_count: int = 10):
        """
        Initialize the trade model.
        
        Args:
            model_path: Path to the saved model file
            bar_count: Number of bars in each observation
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.bar_count = bar_count
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
            dummy_data = pd.DataFrame(
                np.zeros((self.bar_count + 1, len(self.required_columns))),
                columns=self.required_columns
            )
            env = TradingEnv(
                data=dummy_data,
                bar_count=self.bar_count,
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
            bar_count=self.bar_count,
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
        
        env_params = {
            'initial_balance': initial_balance,
            'bar_count': self.bar_count,
            'balance_per_lot': balance_per_lot,  # Standard balance per lot ratio
            'random_start': False
        }

        # Create environment for backtesting
        env = TradingEnv(data=data, **env_params)
        
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
        """
        Calculate metrics from backtest results.
        
        Args:
            env: Trading environment with completed backtest
            total_steps: Total number of steps in the backtest
            total_reward: Total reward accumulated during backtest
            
        Returns:
            Dictionary with calculated metrics
        """
        # Handle case with no trades
        if not env.trades:
            return {
                'final_balance': float(env.balance),
                'initial_balance': float(env.initial_balance),
                'return_pct': 0.0,
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'total_steps': total_steps,
                'total_reward': total_reward,
            'grid_positions': len(env.long_positions) + len(env.short_positions),
                'trades': []
            }
            
        # Create DataFrame from trades for easier analysis
        trades_df = pd.DataFrame(env.trades)
        
        # Basic profit metrics
        winning_trades = [t['pnl'] for t in env.trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in env.trades if t['pnl'] < 0]
        
        # Winning/losing trades statistics
        total_trades = len(env.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
        
        # Profit/loss statistics
        avg_profit = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Position-specific statistics
        if 'position' in trades_df.columns:
            num_buy = trades_df[trades_df["position"] == 1].shape[0]
            num_sell = trades_df[trades_df["position"] == -1].shape[0]
            buy_win_count = trades_df[(trades_df["position"] == 1) & (trades_df["pnl"] > 0.0)].shape[0]
            sell_win_count = trades_df[(trades_df["position"] == -1) & (trades_df["pnl"] > 0.0)].shape[0]
            buy_win_rate = (buy_win_count / num_buy * 100) if num_buy > 0 else 0.0
            sell_win_rate = (sell_win_count / num_sell * 100) if num_sell > 0 else 0.0
        else:
            num_buy = num_sell = 0
            buy_win_rate = sell_win_rate = 0.0
        
        # Risk metrics
        avg_rrr = trades_df["rrr"].mean() if 'rrr' in trades_df.columns else 0.0
        expected_value = trades_df["pnl"].mean() if total_trades > 0 else 0.0
        
        # Kelly criterion
        kelly = (win_rate/100.0) - ((1 - (win_rate/100.0)) / avg_rrr) if avg_rrr > 0 else 0.0
        
        # Calculate drawdowns
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
        
        # Sharpe ratio calculation
        if total_trades > 1:
            daily_returns = trades_df["pnl"] / env.initial_balance
            excess_returns = np.array(daily_returns)
            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252) if np.std(excess_returns, ddof=1) > 0 else 0.0
        else:
            sharpe = 0.0
        
        return {
            'final_balance': float(env.balance),
            'initial_balance': float(env.initial_balance),
            'return_pct': float(((env.balance / env.initial_balance) - 1) * 100),
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': float(win_rate),
            'avg_profit': float(avg_profit),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'max_drawdown_pct': float(max_drawdown * 100),
            'long_trades': int(num_buy),
            'short_trades': int(num_sell),
            'long_win_rate': float(buy_win_rate),
            'short_win_rate': float(sell_win_rate),
            'avg_rrr': float(avg_rrr),
            'expected_value': float(expected_value),
            'kelly_criterion': float(kelly),
            'sharpe_ratio': float(sharpe),
            'total_steps': total_steps,
            'total_reward': total_reward,
            'grid_positions': len(env.long_positions) + len(env.short_positions),
            'grid_metrics': env.grid_metrics,
            'trades': env.trades
        }
