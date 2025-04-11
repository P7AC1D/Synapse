"""Enhanced evaluation callback with comprehensive balance and equity tracking.

This callback provides:
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
    """Optimized evaluation callback with comprehensive balance and equity tracking.

    Features:
    - Tracks both balance and equity drawdowns separately
    - Monitors unrealized PnL for current positions
    - Uses integrated MetricsTracker for consistent metrics
    - Enhanced model selection based on worst-case drawdown
    - Includes combined dataset evaluation for consistency

    The callback saves models based on:
    - Validation set performance (60% weight)
    - Maximum drawdown penalty (30% weight)
    - Training/validation consistency (10% weight)
    - Profit factor bonus (up to +20%)
    """
    def __init__(self, eval_env, train_data, val_data, eval_freq=100000, best_model_save_path=None, 
                 log_path=None, deterministic=True, verbose=1, iteration=0, training_timesteps=200000):
        super(UnifiedEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration
        
        # Store separate datasets
        self.train_data = train_data
        self.val_data = val_data
        
        # Create combined evaluation environment
        self.combined_data = pd.concat([train_data, val_data])

        base_env = eval_env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        env_params = {
            'initial_balance': base_env.initial_balance,
            'balance_per_lot': base_env.BALANCE_PER_LOT,
            'random_start': False
        }
        self.combined_env = Monitor(TradingEnv(self.combined_data, **env_params))
        
        # Initialize tracking metrics
        self.best_score = -float("inf")
        self.best_metrics = {}
        self.max_drawdown = 0.0
        self.training_timesteps = training_timesteps
        
        # Back up raw data for reference
        if hasattr(self.eval_env, 'env'):
            self.eval_env.env.raw_data_backup = self.eval_env.env.raw_data.copy()
        else:
            self.eval_env.raw_data_backup = self.eval_env.raw_data.copy()
            
    def _run_eval_episode(self, env, eval_seed: int = None, start_pos: int = None) -> Dict[str, float]:
        """Run a complete evaluation episode on given environment."""
        if start_pos is not None:
            env.env.current_step = start_pos
        obs, _ = env.reset(seed=eval_seed)
        done = False
        lstm_states = None
        running_balance = env.env.initial_balance
        max_balance = running_balance
        episode_reward = 0
        
        while not done:
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, deterministic=self.deterministic
            )
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
            'balance': env.env.metrics.balance,  # Use actual balance from metrics tracker
            'trades': env.env.trades,
            'current_direction': trade_metrics['current_direction'],
            'profit_factor': performance['profit_factor'],
            'unrealized_pnl': env.env.metrics.current_unrealized_pnl
        }
        
    def _calculate_trade_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall trade quality score with enhanced metrics."""
        win_rate_score = metrics['win_rate']
        profit_factor = metrics.get('profit_factor', 0.0)  # Now directly from performance
        # Use max of both balance and equity drawdowns for penalty
        balance_dd = metrics.get('max_balance_drawdown', 0) / 100  # Convert from percentage
        equity_dd = metrics.get('max_equity_drawdown', 0) / 100  # Convert from percentage
        max_dd = max(balance_dd, equity_dd)
        drawdown_penalty = max(0, 1 - max_dd * 2)
        
        # Ensure directories exist
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            
        # Calculate quality score with adjusted weights
        return (win_rate_score * 0.35 + 
                min(profit_factor, 4) / 4 * 0.45 + 
                drawdown_penalty * 0.2)
                
    def _evaluate_performance(self) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation on all datasets."""
        # Generate consistent seed and start position for both evaluations
        eval_seed = np.random.randint(0, 1000000)
        if self.eval_env.env.random_start:
            max_start = max(0, self.eval_env.env.data_length - 100)
            start_pos = np.random.randint(0, max_start)
        else:
            start_pos = 0
        
        # Use same seed and start position for both evaluations
        combined_metrics = self._run_eval_episode(self.combined_env, eval_seed=eval_seed, start_pos=start_pos)
        val_metrics = self._run_eval_episode(self.eval_env, eval_seed=eval_seed, start_pos=start_pos)
        
        # Calculate consistency score
        consistency_score = val_metrics['return'] / (combined_metrics['return'] + 1e-8)
        
        # Calculate trade quality scores
        val_quality = self._calculate_trade_quality(val_metrics)
        combined_quality = self._calculate_trade_quality(combined_metrics)
        
        # Create comprehensive metrics
        result = {
            'validation': val_metrics,
            'combined': combined_metrics,
            'scores': {
                'consistency': consistency_score,
                'val_quality': val_quality,
                'combined_quality': combined_quality,
                'validation_quality': val_quality  # Add validation quality directly to scores
            }
        }
        
        return result
    
    def _should_save_model(self, metrics: Dict[str, Dict[str, float]]) -> bool:
        """Determine if current model should be saved as best."""
        validation = metrics['validation']
        combined = metrics['combined']
        scores = metrics.get('scores', {})
        
        # IMPORTANT: Reject any model with negative validation return
        if validation['return'] < 0:
            return False
        
        # Properly handle consistency for different scenarios
        consistency = scores.get('consistency', 0)
        if combined['return'] < 0 and validation['return'] > 0:
            # Model improved on validation data - this is good!
            adjusted_consistency = 1.0  # Reward this case
        else:
            # Normal case - use bounded consistency
            adjusted_consistency = max(-0.5, min(consistency, 1.0))
        
        # Calculate validation-focused score
        score = (
            validation['return'] * 0.6 +                  # Higher weight on validation
            -max(
                validation.get('max_balance_drawdown', 0),
                validation.get('max_equity_drawdown', 0)
            ) / 100 * 0.3 +    # Penalize worst drawdown
            adjusted_consistency * 0.1                    # Small reward for consistency
        )
        
        # Add profit factor bonus if available
        if 'performance' in validation and 'profit_factor' in validation['performance']:
            pf = validation['performance']['profit_factor']
            if pf > 1.0:  # Only reward profit factors above 1.0
                profit_factor_bonus = min(pf - 1.0, 2.0) * 0.1  # Up to 20% bonus
                score += profit_factor_bonus
        

        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics
            return True
        return False
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run comprehensive evaluation
            metrics = self._evaluate_performance()
            combined = metrics['combined']
            val = metrics['validation']
            
            def get_detailed_metrics(env, name, metrics):
                """Helper function to collect and print detailed metrics for each dataset"""
                # Get comprehensive metrics from MetricsTracker
                performance = env.env.metrics.get_performance_summary()

                print(f"\n===== {name} Metrics (Timestep {self.num_timesteps:,d}) =====")
                print(f"  Balance: {metrics['balance']:.2f}")
                print(f"  Equity: {metrics['balance'] + metrics['unrealized_pnl']:.2f}")
                print(f"  Unrealized PnL: {metrics['unrealized_pnl']:.2f}")
                print(f"  Return: {metrics['return']*100:.2f}%")
                print(f"  Max Balance Drawdown: {performance['max_drawdown_pct']:.2f}%")
                print(f"  Max Equity Drawdown: {performance['max_equity_drawdown_pct']:.2f}%")
                print(f"  Win Rate: {performance['win_rate']:.2f}%")
                print(f"  Total Reward: {metrics['reward']:.2f}")
                print(f"  Steps Completed: {env.env.current_step:,d} / {len(env.env.raw_data):,d}")
                
                # Performance Metrics
                print("\n  Performance Metrics:")
                print(f"    Total Trades: {performance['total_trades']}")
                print(f"    Average Win: {performance['avg_win']:.2f}")
                print(f"    Average Loss: {performance['avg_loss']:.2f}")
                print(f"    Profit Factor: {performance['profit_factor']:.2f}")
                print(f"    Long Trades: {performance['long_trades']} ({performance['long_win_rate']:.1f}% win)")
                print(f"    Short Trades: {performance['short_trades']} ({performance['short_win_rate']:.1f}% win)")
                print(f"    Average Hold Time: {performance['avg_hold_time']:.1f} bars")
                print(f"    Winners Hold Time: {performance['win_hold_time']:.1f} bars")
                print(f"    Losers Hold Time: {performance['loss_hold_time']:.1f} bars")

                # Return metrics without redundant basic_metrics dictionary
                # Calculate equity value
                equity = metrics['balance'] + metrics['unrealized_pnl']
                
                return {
                    "name": name,
                    "balance": metrics['balance'],
                    "equity": equity,
                    "unrealized_pnl": metrics['unrealized_pnl'],
                    "return": metrics['return'] * 100,
                    "max_balance_drawdown": performance['max_drawdown_pct'],
                    "max_equity_drawdown": performance['max_equity_drawdown_pct'],
                    "win_rate": performance['win_rate'],
                    "total_reward": metrics['reward'],
                    "performance": {
                        "total_trades": performance['total_trades'],
                        "average_win": performance['avg_win'],
                        "average_loss": performance['avg_loss'],
                        "profit_factor": performance['profit_factor'],
                        "long_trades": performance['long_trades'],
                        "long_win_rate": performance['long_win_rate'],
                        "short_trades": performance['short_trades'],
                        "short_win_rate": performance['short_win_rate'],
                        "avg_hold_time": performance['avg_hold_time'],
                        "win_hold_time": performance['win_hold_time'],
                        "loss_hold_time": performance['loss_hold_time']
                    }
                }

            # Get detailed metrics for both datasets
            combined_metrics = get_detailed_metrics(self.combined_env, "Combined Dataset", combined)
            val_metrics = get_detailed_metrics(self.eval_env, "Validation Set", val)
            
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'combined': combined_metrics,
                    'validation': val_metrics
                })
                
                iteration_file = os.path.join(self.log_path, f"eval_results_iter_{self.iteration}.json")
                with open(iteration_file, "w") as f:
                    json.dump(self.eval_results, f, indent=2)
                    
                combined_file = os.path.join(self.log_path, "eval_results_all.json")
                try:
                    with open(combined_file, "r") as f:
                        all_results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_results = {}
                    
                eval_env = self.eval_env
                while hasattr(eval_env, 'env'):
                    eval_env = eval_env.env
                    if isinstance(eval_env, TradingEnv):
                        break

                # Calculate drawdown metrics
                running_balance = eval_env.initial_balance
                max_balance = eval_env.initial_balance
                period_max_drawdown = 0.0
                
                # Calculate running drawdown using trade history
                for trade in eval_env.trades:
                    running_balance += trade['pnl']
                    max_balance = max(max_balance, running_balance)
                    if max_balance > 0:
                        current_drawdown = (max_balance - running_balance) / max_balance
                        period_max_drawdown = max(period_max_drawdown, current_drawdown)

                # Update historical max drawdown
                self.max_drawdown = max(self.max_drawdown, period_max_drawdown)

                # Calculate basic metrics
                active_position = 1 if eval_env.current_position else 0
                num_winning_trades = len([t for t in eval_env.trades if t['pnl'] > 0])
                num_losing_trades = len([t for t in eval_env.trades if t['pnl'] <= 0])
                
                try:
                    period_start = str(eval_env.original_index[0])
                    period_end = str(eval_env.original_index[-1])
                except (AttributeError, IndexError) as e:
                    period_start = period_end = "NA"
                    print(f"Warning: Could not get period timestamps: {str(e)}")

                # Calculate win rate directly from trades
                win_rate = (num_winning_trades / len(eval_env.trades) * 100) if eval_env.trades else 0.0
                
                period_info = {
                    'results': self.eval_results,
                    'iteration': str(self.iteration),
                    'balance': str(float(eval_env.balance)),
                    'total_trades': str(len(eval_env.trades)),
                    'active_position': str(active_position),
                    'win_count': str(num_winning_trades),
                    'loss_count': str(num_losing_trades),
                    'win_rate': str(win_rate),
                    'period_start': period_start,
                    'period_end': period_end,
                    'trade_metrics': eval_env.trade_metrics,
                    'max_drawdown': str(period_max_drawdown * 100),
                    'historical_max_drawdown': str(self.max_drawdown * 100)
                }

                all_results[f"iteration_{self.iteration}"] = period_info
                
                with open(combined_file, "w") as f:
                    json.dump(all_results, f, indent=2)
            
            # Check if model should be saved as best
            if self._should_save_model(metrics) and self.best_model_save_path is not None:
                model_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(model_path)
                
                print(f"\n=== New Best Model Saved at Timestep {self.num_timesteps:,d} ===")
                print(f"Combined Return: {metrics['combined']['return']*100:.2f}%")
                print(f"Validation Return: {metrics['validation']['return']*100:.2f}%")
                print(f"Consistency Score: {metrics['scores']['consistency']:.2f}")
                print(f"Trade Quality: {metrics['scores']['combined_quality']:.2f}")
                print(f"Training Progress: {self.num_timesteps/self.training_timesteps*100:.1f}% complete")

            self.last_time_trigger = self.n_calls
        
        return True
