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
from typing import Dict
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from trading.environment import TradingEnv

class UnifiedEvalCallback(BaseCallback):
    """Specialized evaluation callback for iterative model improvement.

    Features:
    - Test set evaluation at iteration end
    - Tracks both balance and equity drawdowns separately
    - Monitors unrealized PnL for current positions
    - Uses integrated MetricsTracker for consistent metrics
    - Enhanced model selection based on worst-case drawdown

    Model Selection Process:
    1. Models with positive returns on validation set are saved as curr_best_model.zip
    2. At iteration end, curr_best_model is evaluated on test set
    3. Models with positive test returns are considered for final selection
    4. Best performing model on test set becomes best_model for warm start
    """
    def __init__(self, eval_env, test_env, train_data, val_data, test_data, eval_freq=100000, 
                 best_model_save_path=None, log_path=None, deterministic=True, verbose=0, 
                 iteration=0, training_timesteps=200000):
        super(UnifiedEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env  # Validation environment
        self.test_env = test_env  # Test environment
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration
        self.training_timesteps = training_timesteps

        # Store separate datasets
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # Initialize tracking metrics
        self.best_val_score = -float("inf")
        self.best_test_score = -float("inf")
        self.best_metrics = {}

    def _run_eval_episode(self, env, eval_seed: int = None, start_pos: int = None) -> Dict[str, float]:
        """Run a complete evaluation episode on given environment."""
        if start_pos is not None:
            env.env.current_step = start_pos
        obs, _ = env.reset(seed=eval_seed)
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated        
            episode_reward += reward

            # Track running metrics
            running_balance = env.env.balance
            max_balance = max(max_balance, running_balance)
        
        # Get comprehensive metrics from MetricsTracker
        performance = env.env.metrics.get_performance_summary()
        trade_metrics = env.env.trade_metrics

        # Use performance summary directly
        return {
            'return': performance['return_pct'] / 100,
            'max_balance_drawdown': performance['max_drawdown_pct'] / 100,
            'max_equity_drawdown': performance['max_equity_drawdown_pct'] / 100,
            'reward': episode_reward,
            'win_rate': performance['win_rate'] / 100,
            'avg_profit': performance['avg_win'],
            'avg_loss': performance['avg_loss'],
            'balance': env.env.metrics.balance,
            'trades': env.env.trades,
            'current_direction': trade_metrics['current_direction'],
            'profit_factor': performance['profit_factor'],
            'unrealized_pnl': env.env.metrics.current_unrealized_pnl
        }
        
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate model performance score using financial DRL evaluation criteria.
        
        Implements evaluation based on:
        - Sortino ratio (downside risk-adjusted returns)
        - Calmar ratio (drawdown-adjusted returns)
        - Trade consistency measures
        - Risk-adjusted profit factor
        
        Returns negative infinity for models with unacceptable risk characteristics.
        """
        returns = metrics['return']
        max_dd = max(metrics['max_drawdown'], metrics['max_equity_drawdown'])
        profit_factor = metrics['profit_factor']
        win_rate = metrics['win_rate']
        # Extract metrics with safe defaults
        metrics_data = metrics.get('metrics', {})
        trades = metrics_data.get('total_trades', 0)
          # Calculate average trade PnL and std if we have trades
        if trades > 0:
            trade_pnls = metrics_data.get('trade_pnls', [])
            # Safety check to avoid empty lists
            if not trade_pnls:
                trade_pnls = [0]
                
            avg_trade = np.mean(trade_pnls)
            trade_std = np.std(trade_pnls) if len(trade_pnls) > 1 else 0.0
        else:
            avg_trade = 0.0
            trade_std = 0.0
        
        # Reject models with unacceptable characteristics
        if any([
            returns <= 0,          # Must be profitable
            max_dd > 0.15,        # Max 15% drawdown
            profit_factor < 1.2,   # Minimum profit factor
            trades < 20,          # Minimum trade count for significance
            win_rate < 0.40       # Minimum win rate
        ]):
            return float('-inf')
              # 1. Sortino Ratio (30% weight)
        # Only consider downside deviation
        losing_trades = []
        if 'trade_pnls' in metrics['metrics'] and metrics['metrics']['trade_pnls']:
            losing_trades = [t for t in metrics['metrics']['trade_pnls'] if t < 0]
            
        downside_std = np.std(losing_trades) if losing_trades else 0.001
        sortino_ratio = returns / (downside_std + 0.001)
        score = min(sortino_ratio, 4.0) / 4.0 * 0.30  # Cap at 4.0
        
        # 2. Calmar Ratio (25% weight)
        # Returns / Max Drawdown
        calmar_ratio = returns / (max_dd + 0.001)
        score += min(calmar_ratio, 3.0) / 3.0 * 0.25  # Cap at 3.0
        
        # 3. Trade Consistency (25% weight)
        # Coefficient of variation of trade PnL (adjusted)
        cv = abs(trade_std / (avg_trade + 0.001))
        consistency_score = 1.0 / (1.0 + cv)  # Transform to 0-1 range
        score += consistency_score * 0.25
        
        # 4. Risk-Adjusted Profit Factor (20% weight)
        # Profit factor penalized by drawdown severity
        risk_adj_pf = profit_factor * (1.0 - max_dd)
        score += min(risk_adj_pf, 2.0) / 2.0 * 0.20  # Cap at 2.0
        
        return score
                
    def _evaluate_performance(self, is_final_eval: bool = False) -> Dict[str, Dict[str, float]]:
        """Run evaluation on validation and optionally test sets."""
        # Generate consistent seed
        eval_seed = np.random.randint(0, 1000000)
        
        # Always evaluate on validation set
        val_metrics = self._run_eval_episode(self.eval_env, eval_seed=eval_seed)
        
        result = {'validation': val_metrics}
        
        # Run test set evaluation only at iteration end
        if is_final_eval:
            test_metrics = self._run_eval_episode(self.test_env, eval_seed=eval_seed)
            result['test'] = test_metrics
        
        return result
    
    def _should_save_model(self, metrics: Dict[str, Dict[str, float]], is_final_eval: bool) -> bool:
        """
        Determine if current model should be saved based on financial evaluation criteria.
        
        For validation phase:
        - Must meet minimum criteria (positive returns, max DD, etc.)
        - Must improve on best validation score
        
        For test phase:
        - Must meet stricter criteria for production deployment
        - Must improve on best test score with statistical significance
        """
        val_metrics = metrics['validation']
        val_score = self._calculate_score(val_metrics)
        
        # Check minimum criteria for validation
        if not is_final_eval:
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                return True
            return False
        
        # Stricter criteria for test phase
        test_metrics = metrics['test']
        test_score = self._calculate_score(test_metrics)
        
        # Additional test phase requirements
        min_trades = 20  # Minimum trades for statistical significance
        # Handle datetime or string timestamps
        metrics_data = test_metrics.get('metrics', {})
        start_time = metrics_data.get('start_time')
        end_time = metrics_data.get('end_time')
        
        # Convert string timestamps to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        test_period_days = (end_time - start_time).days if start_time and end_time else 0
        
        # Reject if not enough trades or too short test period
        if test_metrics['metrics']['total_trades'] < min_trades or test_period_days < 30:
            return False
            
        # Check for improvement with higher threshold
        if test_score > self.best_test_score * 1.05:  # Require 5% improvement
            self.best_test_score = test_score
            self.best_metrics = metrics
            return True
            
        return False
    
    def _on_step(self) -> bool:
        """Helper function to collect and print detailed metrics for each dataset"""
        def get_detailed_metrics(env, name, metrics):
            # Get comprehensive metrics from MetricsTracker
            performance = env.env.metrics.get_performance_summary()

            print(f"\n===== {name} Metrics (Timestep {self.num_timesteps:,d}) =====")
            print(f"  Balance: ${metrics['balance']:.2f} (${metrics['balance'] + metrics['unrealized_pnl']:.2f})")
            print(f"  Unrealized PnL: {metrics['unrealized_pnl']:.2f}")
            print(f"  Return: {metrics['return']*100:.2f}%")
            print(f"  Max Drawdown: {performance['max_drawdown_pct']:.2f}% ({performance['max_equity_drawdown_pct']:.2f}%)")
            print(f"  Total Reward: {metrics['reward']:.2f}")
            print(f"  Steps Completed: {env.env.current_step:,d} / {len(env.env.raw_data):,d}")
            
            # Training Metrics
            try:
                # Debug logging at a finer level
                if self.verbose > 1:
                    print("\nAvailable Training Metrics:", list(self.model.logger.name_to_value.keys()))
                
                # Collect training metrics
                training_stats = {
                    # Value network stats
                    "value_loss": float(self.model.logger.name_to_value.get('train/value_loss', 0.0)),
                    "explained_variance": float(self.model.logger.name_to_value.get('train/explained_variance', 0.0)),
                    # Policy network stats
                    "policy_loss": float(self.model.logger.name_to_value.get('train/policy_gradient_loss', 0.0)),
                    "entropy_loss": float(self.model.logger.name_to_value.get('train/entropy_loss', 0.0)),
                    "approx_kl": float(self.model.logger.name_to_value.get('train/approx_kl', 0.0)),
                    # Training stats
                    "total_loss": float(self.model.logger.name_to_value.get('train/loss', 0.0)),
                    "clip_fraction": float(self.model.logger.name_to_value.get('train/clip_fraction', 0.0)),
                    "learning_rate": float(self.model.logger.name_to_value.get('train/learning_rate', 0.0)),
                    "n_updates": int(self.model.logger.name_to_value.get('train/n_updates', 0))
                }
                
                print("\n  Network Stats:")
                print(f"    Value Network:")
                print(f"      Loss: {training_stats['value_loss']:.4f}")
                print(f"      Explained Var: {training_stats['explained_variance']:.2f}")
                print(f"    Policy Network:")
                print(f"      Loss: {training_stats['policy_loss']:.4f}")
                print(f"      Entropy: {training_stats['entropy_loss']:.4f}")
                print(f"      KL Div: {training_stats['approx_kl']:.4f}")
                print(f"    Training:")
                print(f"      Total Loss: {training_stats['total_loss']:.4f}")
                print(f"      Clip Fraction: {training_stats['clip_fraction']:.4f}")
                print(f"      Learning Rate: {training_stats['learning_rate']:.6f}")
                print(f"      Updates: {training_stats['n_updates']}")
            except (KeyError, AttributeError):
                print("    Training stats not yet available")
                training_stats = {
                    # Value network stats
                    "value_loss": 0.0,
                    "explained_variance": 0.0,
                    # Policy network stats
                    "policy_loss": 0.0,
                    "entropy_loss": 0.0,
                    "approx_kl": 0.0,
                    # Training stats
                    "total_loss": 0.0,
                    "clip_fraction": 0.0,
                    "learning_rate": 0.0,
                    "n_updates": 0
                }
            
            # Performance Metrics
            print("\n  Performance Metrics:")
            print(f"    Total Trades: {performance['total_trades']} ({performance['win_rate']:.2f}% win)")
            print(f"    Average Win: {performance['avg_win_points']:.1f} points ({performance['win_hold_time']:.1f} bars)")
            print(f"    Average Loss: {performance['avg_loss_points']:.1f} points ({performance['loss_hold_time']:.1f} bars)")
            print(f"    Long Trades: {performance['long_trades']} ({performance['long_win_rate']:.1f}% win)")
            print(f"    Short Trades: {performance['short_trades']} ({performance['short_win_rate']:.1f}% win)")
            print(f"    Profit Factor: {performance['profit_factor']:.2f}")

            # Group metrics into categories
            account_stats = {
                "balance": metrics['balance'],
                "equity": metrics['balance'] + metrics['unrealized_pnl'],
                "unrealized_pnl": metrics['unrealized_pnl'],
                "return": metrics['return'] * 100,
                "max_dd": performance['max_drawdown_pct'],
                "max_equity_dd": performance['max_equity_drawdown_pct']
            }

            trading_stats = {
                "win_rate": performance['win_rate'],
                "total_trades": len(metrics['trades']),
                "steps_completed": env.env.current_step,
                "total_steps": len(env.env.raw_data),
                "total_reward": metrics['reward']
            }

            performance_stats = {
                "total_trades": performance['total_trades'],
                "average_win": performance['avg_win'],
                "average_loss": performance['avg_loss'],
                "avg_win_points": performance['avg_win_points'],
                "avg_loss_points": performance['avg_loss_points'],
                "profit_factor": performance['profit_factor'],
                "long_trades": performance['long_trades'],
                "long_win_rate": performance['long_win_rate'],
                "short_trades": performance['short_trades'],
                "short_win_rate": performance['short_win_rate'],
                "avg_hold_time": performance['avg_hold_time'],
                "win_hold_time": performance['win_hold_time'],
                "loss_hold_time": performance['loss_hold_time']
            }

            return {
                "name": name,
                "account": account_stats,
                "trading": trading_stats,
                "performance": performance_stats,
                "training": training_stats
            }
            
        """Execute evaluation steps."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run evaluation
            is_final_eval = self.num_timesteps >= self.training_timesteps - self.eval_freq
            metrics = self._evaluate_performance(is_final_eval)
            val_metrics = get_detailed_metrics(self.eval_env, "Validation Results", metrics)
            
            if is_final_eval:
                test_metrics = get_detailed_metrics(self.test_env, "Test Results", metrics)
            
            # Save if performance improved
            if self._should_save_model(metrics, is_final_eval):
                # Save model
                save_path = os.path.join(self.best_model_save_path,
                                       "best_test_model.zip" if is_final_eval else "curr_best_model.zip")
                self.model.save(save_path)
                
                # Save metrics
                metrics_path = save_path.replace(".zip", "_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"\nNew best {'test' if is_final_eval else 'validation'} model saved!")
            
            self.last_time_trigger = self.n_calls
            
        return True
