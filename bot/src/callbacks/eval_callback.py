"""Enhanced evaluation callback with comprehensive balance and equity tracking.

This module provides:
- Training and validation evaluation
- Balance and equity drawdown tracking
- Unrealized PnL monitoring
- Enhanced model selection based on comprehensive metrics
- Full performance tracking using MetricsTracker
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
    """
    Specialized evaluation callback for monitoring training progress.

    Features:
    - Validation set evaluation during training
    - Real-time balance and equity metrics tracking
    - Comprehensive regime consistency analysis
    - Robust model selection with multi-factor scoring
    """
    def __init__(self, eval_env, eval_freq=100000, model_save_path=None, log_path=None, 
                 deterministic=True, verbose=0, iteration=0):
        super(ValidationCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.model_save_path = model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration
        
        self.best_val_score = -float("inf")
        print("\nInitialized validation callback with enhanced scoring system")

    def _evaluate_model(self, eval_seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Run complete evaluation episode with regime analysis."""
        obs, _ = self.eval_env.reset(seed=eval_seed)
        done = False
        episode_reward = 0
        lstm_states = None
        
        # For regime consistency tracking
        regime_pnl = {'trending': [], 'ranging': []}
        last_balance = self.eval_env.env.balance
        
        while not done:
            # Get trend strength and volatility breakout from observation
            trend_strength = obs[5]  # Index 5 is trend_strength
            volatility_breakout = obs[4]  # Index 4 is volatility_breakout
            is_trending = trend_strength > 0.3 and volatility_breakout > 0.7

            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                deterministic=self.deterministic
            )
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated        
            episode_reward += reward

            # Track PnL per step for regime analysis
            step_pnl = self.eval_env.env.balance - last_balance
            last_balance = self.eval_env.env.balance

            if is_trending:
                regime_pnl['trending'].append(step_pnl)
            else:
                regime_pnl['ranging'].append(step_pnl)
        
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
            },
            'regime_pnl': regime_pnl
        }
        return metrics

    def _calculate_score(self, metric_data: Dict[str, Any]) -> float:
        """
        Calculate model score using enhanced scoring system that considers:
        - Risk-adjusted returns (35%)
        - Raw returns (25%)
        - Regime consistency (20%)
        - Profit factor (10%)
        - Win rate (10%)
        """
        # Extract core metrics
        returns = metric_data['return']  # Already in decimal form
        max_dd = max(metric_data['max_balance_drawdown'], metric_data['max_equity_drawdown'])
        profit_factor = metric_data['profit_factor']
        win_rate = metric_data['win_rate']  # Already in decimal form
        regime_pnl = metric_data['regime_pnl']
        initial_balance = self.eval_env.env.initial_balance

        # Calculate regime consistency bonus
        total_trending_pnl = sum(regime_pnl['trending'])
        total_ranging_pnl = sum(regime_pnl['ranging'])
        
        # Normalize PnL by initial balance
        norm_trending_pnl = total_trending_pnl / initial_balance
        norm_ranging_pnl = total_ranging_pnl / initial_balance

        regime_consistency_bonus = 0.0
        if norm_trending_pnl > 0 and norm_ranging_pnl > 0:
            regime_consistency_bonus += 0.5  # Both regimes profitable
            if min(abs(norm_trending_pnl), abs(norm_ranging_pnl)) / (max(abs(norm_trending_pnl), abs(norm_ranging_pnl)) + 1e-6) > 0.25:
                regime_consistency_bonus += 0.5  # Balanced performance
        elif (norm_trending_pnl > 0 and norm_ranging_pnl < -0.05 * abs(norm_trending_pnl)) or \
             (norm_ranging_pnl > 0 and norm_trending_pnl < -0.05 * abs(norm_ranging_pnl)):
            regime_consistency_bonus -= 0.5  # One regime subsidizing losses

        # Calculate score components
        risk_adj_return = returns / (max_dd + 0.05)  # Add small constant to avoid division by zero
        scaled_regime_consistency = (regime_consistency_bonus + 0.5) / 1.5  # Scale to [0,1]
        
        # Profit factor bonus
        pf_bonus = min(max(profit_factor - 1.0, 0.0), 2.0) / 2.0  # Scale to [0,1]

        # Final weighted score
        score = (
            risk_adj_return * 0.35 +           # Risk-adjusted returns (35%)
            returns * 0.25 +                    # Raw returns (25%)
            scaled_regime_consistency * 0.20 +  # Regime consistency (20%)
            pf_bonus * 0.10 +                  # Profit factor (10%)
            win_rate * 0.10                    # Win rate (10%)
        )
        
        if self.verbose > 0:
            print(f"\nScore components:")
            print(f"  Risk-Adjusted Return: {risk_adj_return:.4f}")
            print(f"  Raw Returns: {returns:.2%}")
            print(f"  Regime Consistency: {scaled_regime_consistency:.4f}")
            print(f"  Profit Factor Bonus: {pf_bonus:.4f}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Final Score: {score:.4f}")
        
        return score
    
    def _on_step(self) -> bool:
        """Execute validation step."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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
                    # Save best model
                    path = os.path.join(self.model_save_path, "best_model.zip")
                    self.model.save(path)
                    print(f"\nNew best model saved: {path}")
                    print(f"Validation Score: {score:.4f}")
                    
                    # Save validation metrics
                    metrics_path = path.replace(".zip", "_metrics.json")
                    metrics = {
                        'score': score,
                        'metrics': metrics,
                        'iteration': self.iteration,
                        'timesteps': self.num_timesteps,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            self.last_time_trigger = self.n_calls
            print(f"Completed validation evaluation at step {self.n_calls}")
            
        return True
