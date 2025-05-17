"""Enhanced evaluation callback with comprehensive balance and equity tracking.

This callback provides:
- Train/validation/test set evaluation
- Separate balance and equity drawdown tracking
- Unrealized PnL monitoring
- Enhanced model selection based on both equity and balance metrics
- Comprehensive performance summary from MetricsTracker
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from trading.environment import TradingEnv

class ValidationCallback(BaseCallback):
    """Specialized evaluation callback for monitoring training progress.

    Features:
    - Validation set evaluation during training
    - Tracks balance and equity metrics
    - Monitors training progress and convergence
    - Provides detailed performance summaries
    """
    def __init__(self, eval_env, eval_freq=100000, model_save_path=None, log_path=None, 
                 deterministic=True, verbose=0, iteration=0):
        super(ValidationCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env  # Validation environment
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration
        
        # Initialize validation tracking only (removed test tracking)
        self.best_val_score = -float("inf")
        print("\nInitialized validation callback with validation-only tracking")

    def _evaluate_model(self, eval_seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Run complete evaluation episode."""
        obs, _ = self.eval_env.reset(seed=eval_seed)
        done = False
        episode_reward = 0
        lstm_states = None
        
        while not done:
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                deterministic=self.deterministic
            )
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated        
            episode_reward += reward
        
        # Get metrics
        performance = self.eval_env.env.metrics.get_performance_summary()
        trade_metrics = self.eval_env.env.trade_metrics
        
        metrics = {
            'return': performance['return_pct'] / 100,
            'max_balance_drawdown': performance['max_drawdown_pct'] / 100,
            'max_equity_drawdown': performance['max_equity_drawdown_pct'] / 100,
            'reward': episode_reward,
            'win_rate': performance['win_rate'] / 100,
            'balance': self.eval_env.env.metrics.balance,
            'trades': self.eval_env.env.trades,
            'profit_factor': performance['profit_factor'],
            'unrealized_pnl': self.eval_env.env.metrics.current_unrealized_pnl,
            'metrics': {
                'performance': performance,
                'env': self.eval_env
            }
        }
        return metrics

    def _calculate_score(self, metric_data: Dict[str, Any]) -> float:
        """Calculate model score based on validation metrics."""
        # Extract core metrics
        returns = metric_data['return']  # Already in decimal form
        max_dd = max(metric_data['max_balance_drawdown'], metric_data['max_equity_drawdown'])  # Already in decimal form
        profit_factor = metric_data['profit_factor']
        win_rate = metric_data['win_rate']  # Already in decimal form
        trades = len(metric_data['trades'])
        
        # Simple scoring based on risk-adjusted returns
        score = returns / (max_dd + 0.05)  # Add small constant to avoid division by zero
        
        if self.verbose > 0:
            print(f"\nScore components:")
            print(f"  Returns: {returns:.2%}")
            print(f"  Max DD: {max_dd:.2%}")
            print(f"  Trades: {trades}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Profit Factor: {profit_factor:.2f}")
            print(f"  Final Score: {score:.4f}")
        
        return score
    
    
    def _on_step(self) -> bool:
        """Execute validation step."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run validation
            metrics = self._evaluate_model(eval_seed=np.random.randint(0, 1000000))
            score = self._calculate_score(metrics)
            
            # Print validation metrics with model stats
            self.eval_env.env.metrics.print_evaluation_metrics(
                phase="Validation",
                timestep=self.num_timesteps,
                model=self.model
            )
            
            # Save if validation score improved
            if score > self.best_val_score:
                self.best_val_score = score
                if self.model_save_path:
                    # Save as validation checkpoint
                    path = os.path.join(self.model_save_path, f"best_validation_iter_{self.iteration}.zip")
                    self.model.save(path)
                    print(f"\nNew best validation model saved: {path}")
                    print(f"Validation Score: {score:.4f}")
                    
                    # Save validation metrics
                    metrics_path = path.replace(".zip", "_metrics.json")
                    metrics = {
                        'score': score,
                        'metrics': metrics
                    }
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            self.last_time_trigger = self.n_calls
            print(f"Completed validation evaluation at step {self.n_calls}")
            
        return True
