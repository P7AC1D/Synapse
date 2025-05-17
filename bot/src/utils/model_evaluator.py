"""Model evaluation utilities for PPO-LSTM models.

This module provides utilities for evaluating trained models on validation and test sets,
with comprehensive metrics tracking and model selection logic. It handles:
- Model evaluation on different datasets
- Performance scoring and model selection
- Metrics logging and model saving
- Best model tracking across iterations
"""
import os
import json
import numpy as np
from utils.trading_visualizer import TradingVisualizer
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv

class ModelEvaluator:
    """Handles model evaluation, selection and saving.
    
    Features:
    - Comprehensive model evaluation on validation/test sets
    - Enhanced scoring system for model selection
    - Consistent metrics tracking and logging
    - Best model management
    """
    def __init__(self, 
                 save_path: str,
                 device: str = "auto",
                 verbose: int = 0):
        self.save_path = save_path
        self.device = device
        self.verbose = verbose
        
        # Initialize tracking metrics
        self.best_val_score = -float("inf")
        self.best_test_score = -float("inf")
        self.best_metrics = {}

    def evaluate_model(self, 
                      model: RecurrentPPO,
                      env: Monitor,
                      deterministic: bool = True,
                      eval_seed: Optional[int] = None) -> Dict[str, float]:
        """Run complete evaluation of model on given environment."""
        obs, _ = env.reset(seed=eval_seed)
        done = False
        running_balance = env.env.initial_balance
        max_balance = running_balance
        episode_reward = 0
        lstm_states = None
        
        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                deterministic=deterministic
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated        
            episode_reward += reward

            # Track running metrics
            running_balance = env.env.balance
            max_balance = max(max_balance, running_balance)
        
        # Get comprehensive metrics
        performance = env.env.metrics.get_performance_summary()
        trade_metrics = env.env.trade_metrics
        
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
            'unrealized_pnl': env.env.metrics.current_unrealized_pnl,
            'performance': performance,
            'env': env
        }

    def calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate performance score using financial criteria."""
        # Extract core metrics
        returns = metrics['return']
        max_dd = max(metrics['max_balance_drawdown'], metrics['max_equity_drawdown'])
        profit_factor = metrics['profit_factor']
        win_rate = metrics['win_rate']
        trades = len(metrics['trades'])
        
        # Reject models with unacceptable characteristics
        if any([
            returns <= 0,          # Must be profitable
            max_dd > 0.15,        # Max 15% drawdown
            profit_factor < 1.2,   # Minimum profit factor
            trades < 20,          # Minimum trade count
            win_rate < 0.40       # Minimum win rate
        ]):
            return float('-inf')
        
        # Calculate component scores
        score = 0.0
        
        # 1. Risk-adjusted returns (35%)
        risk_adj_return = returns / (max_dd + 0.05)
        score += (risk_adj_return / 4.0) * 0.35 # Cap at 4.0
        
        # 2. Raw returns (25%)
        score += min(returns / 0.15, 1.0) * 0.25 # Cap at 15%
        
        # 3. Trade consistency (20%)
        trade_pnls = [t['pnl'] for t in metrics['trades']]
        avg_trade = np.mean(trade_pnls)
        trade_std = np.std(trade_pnls) if len(trade_pnls) > 1 else 0.0
        
        # Coefficient of variation (lower is better)
        cv = abs(trade_std / (avg_trade + 0.001))
        consistency_score = 1.0 / (1.0 + cv)
        score += consistency_score * 0.20
        
        # 4. Risk-adjusted profit factor (20%)
        risk_adj_pf = profit_factor * (1.0 - max_dd)
        score += min(risk_adj_pf / 2.0, 1.0) * 0.20
        
        return score

    def evaluate_historical(self,
                          model: RecurrentPPO,
                          full_data: pd.DataFrame,
                          config: Any,
                          eval_seed: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on full historical dataset."""
        historical_env = Monitor(TradingEnv(full_data, predict_mode=False, config=config))
        return self.evaluate_model(model, historical_env, eval_seed=eval_seed)

    def plot_historical_results(self, results: Dict[str, Any], iteration: int) -> None:
        """Plot historical evaluation results."""
        if 'trades' not in results or not results['trades']:
            return
            
        plots_dir = os.path.join(self.save_path, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'iteration_{iteration}.png')
        
        TradingVisualizer.plot_results(
            results=results,
            save_path=plot_path,
            title=f'Historical Performance - Iteration {iteration}'
        )

    def select_best_model(self,
                         model: RecurrentPPO,
                         val_env: Monitor,
                         full_data: pd.DataFrame,
                         config: Any,
                         iteration: int,
                         is_final_eval: bool = False) -> Dict[str, Any]:
        """Evaluate and potentially select model as new best."""
        # Store model reference for network stats
        self.model = model
        
        # Generate consistent seed
        eval_seed = np.random.randint(0, 1000000)
        
        # Always evaluate on validation set
        val_metrics = self.evaluate_model(model, val_env, eval_seed=eval_seed)
        val_score = self.calculate_score(val_metrics)
        
        result = {
            'validation': {
                'metrics': val_metrics,
                'score': val_score
            }
        }
        
        save_model = False
        
        # For validation phase
        if not is_final_eval:
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                save_model = True
                save_name = "curr_best_model.zip"
        
        # For test phase (full historical evaluation)
        else:
            historical_metrics = self.evaluate_historical(model, full_data, config, eval_seed=eval_seed)
            historical_score = self.calculate_score(historical_metrics)
            
            result['historical'] = {
                'metrics': historical_metrics,
                'score': historical_score
            }
            
            # Check for significant improvement (5%)
            if historical_score > self.best_test_score * 1.05:
                self.best_test_score = historical_score
                self.best_metrics = result
                save_model = True
                save_name = "best_historical_model.zip"
        
            # Save if performance improved
            if save_model:
                iter_path = os.path.join(self.save_path, f"iteration_{iteration}")
                os.makedirs(iter_path, exist_ok=True)
                
                # Save model
                model_path = os.path.join(iter_path, save_name)
                model.save(model_path)
                
                # Generate and save plot for historical evaluation
                if is_final_eval:
                    self.plot_historical_results(historical_metrics, iteration)
            
            # Create serializable metrics dictionary by removing env object
            serializable_result = {}
            for phase in result:
                serializable_result[phase] = {
                    'score': result[phase]['score'],
                    'metrics': {
                        k: v for k, v in result[phase]['metrics'].items() 
                        if k != 'env'  # Exclude env object
                    }
                }
            
            # Save serializable metrics
            metrics_path = model_path.replace(".zip", "_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            result['saved_model'] = model_path
            
            if self.verbose > 0:
                phase = "test" if is_final_eval else "validation"
                print(f"\nNew best {phase} model saved at: {model_path}")
        
        return result

    def print_evaluation_results(self, data: Dict[str, Dict[str, Any]], phase: str, timestep: Optional[int] = None) -> None:
        """Print complete evaluation results.
        
        Args:
            data: Evaluation results dictionary with metrics and environment data
            phase: Name of the evaluation phase (e.g. "Training", "Validation", "Test")
            timestep: Optional current timestep for progress tracking
        """
        metrics = data['metrics']  # Get the metrics dictionary
        
        # Print metrics from performance data if env is not available
        if 'env' not in metrics:
            perf = metrics.get('performance', {})
            print(f"\n=== {phase} Results ===")
            print(f"Return: {metrics.get('return', 0)*100:.2f}%")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_balance_drawdown', 0)*100:.2f}%")
            print(f"Total Trades: {len(metrics.get('trades', []))}")
        else:
            # Use metrics tracker if environment is available
            monitor_env = metrics['env']  # This is the Monitor wrapper
            metrics_tracker = monitor_env.env.metrics  # Access the actual TradingEnv's metrics
            metrics_tracker.print_evaluation_metrics(
                phase=phase,
                timestep=timestep,
                model=self.model if hasattr(self, 'model') else None
            )
        
        # Print score separately since it's evaluator-specific
        print(f"Score: {data['score']:.4f}")
