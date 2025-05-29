"""Enhanced evaluation callback with improved model selection scoring.

This callback provides:
- 80/20 validation weighting for base performance
- Risk-to-reward ratio scoring that rewards R:R ratios above 1.0
- Directional balance scoring that rewards balanced long/short trading
- Consistency component
- Profit factor bonus

The enhanced scoring system addresses issues with the original simple average approach
that favored models overfitting to training data over those with better validation performance.
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

class EnhancedEvalCallback(BaseCallback):
    """Enhanced evaluation callback with improved model selection scoring.

    Features:
    - 80/20 validation weighting for base performance (heavily favors unseen data)
    - Risk-to-reward ratio scoring that rewards good risk management
    - Directional balance scoring that rewards balanced long/short strategies
    - Consistency component that rewards models performing well on both datasets
    - No hard minimums - continuous scoring that naturally rewards better behavior

    Model Selection Process:
    1. Models with positive returns on both validation and combined sets are candidates
    2. Enhanced scoring considers multiple factors beyond just returns
    3. Current best model is compared against existing best model using enhanced score
    4. Better performing model becomes the new best_model.zip
    """
    def __init__(self, eval_env, train_data, val_data, eval_freq=100000, best_model_save_path=None, 
                 log_path=None, deterministic=True, verbose=0, iteration=0, training_timesteps=200000):
        super(EnhancedEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
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
            'unrealized_pnl': env.env.metrics.current_unrealized_pnl,
            'performance': performance  # Include full performance data
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
    
    def _calculate_directional_balance_score(self, long_trades: int, short_trades: int) -> float:
        """Calculate directional balance score that rewards balanced long/short trading.
        
        Returns:
            float: Score from 0.0 to 1.0, with 1.0 being perfect balance
        """
        if long_trades == 0 and short_trades == 0:
            return 0.0  # No trades at all
        if long_trades == 0 or short_trades == 0:
            return 0.3  # One-sided trading gets low score but not zero
        
        # Calculate balance ratio (always >= 1.0)
        ratio = max(long_trades, short_trades) / min(long_trades, short_trades)
        
        # Smooth decay function: perfect balance (1.0) = score 1.0
        # Gradually decreases but never hits zero
        return 1.0 / (1.0 + (ratio - 1.0) * 0.5)
    
    def _calculate_risk_reward_score(self, avg_win_points: float, avg_loss_points: float) -> float:
        """Calculate risk-reward score that rewards R:R ratios above 1.0.
        
        Returns:
            float: Score from 0.1 to 1.0, with bonus above 1.0 R:R ratio
        """
        if avg_loss_points >= 0 or avg_win_points <= 0:
            return 0.5  # Neutral score for edge cases
        
        rr_ratio = avg_win_points / abs(avg_loss_points)
        
        # Sigmoid-like curve: 1.0 R:R = 0.5 score, bonus above 1.0
        if rr_ratio >= 1.0:
            # Exponential reward above 1.0, capped at 1.0
            return min(1.0, 0.5 + (rr_ratio - 1.0) * 0.3)
        else:
            # Linear scaling below 1.0, minimum 0.1
            return max(0.1, rr_ratio * 0.5)

    def _calculate_enhanced_score(self, metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate enhanced final score for model comparison.
        
        Enhanced scoring includes:
        - 80/20 validation weighting for base performance
        - Risk-to-reward ratio scoring
        - Directional balance scoring
        - Consistency component
        - Profit factor bonus
        """
        validation = metrics['validation']
        combined = metrics['combined']
        
        # 80/20 validation weighting for base performance
        base_score = (validation['return'] * 0.80 + combined['return'] * 0.20)
        
        # Only proceed if base performance is positive
        if base_score <= 0:
            return float('-inf')
        
        # Get performance data for additional metrics
        val_performance = validation.get('performance', {})
        
        # Risk-to-reward component (from validation data)
        rr_score = self._calculate_risk_reward_score(
            val_performance.get('avg_win_points', 0), 
            val_performance.get('avg_loss_points', 0)
        )
        
        # Directional balance component (from validation data)
        balance_score = self._calculate_directional_balance_score(
            val_performance.get('long_trades', 0),
            val_performance.get('short_trades', 0)
        )
        
        # Consistency component (bonus for consistent performance)
        if validation['return'] > 0 and combined['return'] > 0:
            consistency = min(validation['return'], combined['return']) / max(validation['return'], combined['return'])
        else:
            consistency = 0.0
        
        # Profit factor bonus (existing)
        pf_bonus = 0.0
        if 'profit_factor' in val_performance:
            pf = val_performance['profit_factor']
            if pf > 1.0:  # Only reward profit factors above 1.0
                pf_bonus = min(pf - 1.0, 2.0) * 0.05  # Reduced bonus
        
        # Final weighted score
        final_score = (
            base_score * 0.60 +           # 60% performance (80/20 weighted)
            rr_score * 0.20 +             # 20% risk-reward
            balance_score * 0.15 +        # 15% directional balance  
            consistency * 0.05 +          # 5% consistency bonus
            pf_bonus                      # Small profit factor bonus
        )
        
        return final_score

    def _evaluate_against_previous(self) -> bool:
        """
        Compare current model against best_model.zip using enhanced scoring.
        
        Returns:
            bool: True if current model scores better than best_model
        """
        if not self.best_model_save_path:
            return True  # No path to check existing models
            
        best_model_path = os.path.join(self.best_model_save_path, "best_model.zip")
        if not os.path.exists(best_model_path):
            return True  # No previous best model to compare against
            
        # Get current model performance
        current_metrics = self._evaluate_performance()
        current_score = self._calculate_enhanced_score(current_metrics)
        
        try:
            # Load and evaluate previous best model
            from sb3_contrib.ppo_recurrent import RecurrentPPO
            prev_model = RecurrentPPO.load(best_model_path)
            
            # Store current model temporarily
            temp_model = self.model
            self.model = prev_model
            
            # Evaluate previous model
            prev_metrics = self._evaluate_performance()
            prev_score = self._calculate_enhanced_score(prev_metrics)
            
            # Restore current model
            self.model = temp_model
            
            # Log comparison results
            self.logger.info(f"\nEnhanced Model Comparison (End of Iteration {self.iteration}):")
            self.logger.info(f"  Current Model Score: {current_score:.4f}")
            self.logger.info(f"  Best Model Score: {prev_score:.4f}")
            
            # Compare scores
            return current_score > prev_score
            
        except Exception as e:
            self.logger.warning(f"Error comparing with best model: {e}")
            return True  # Default to accepting current model on error
        
    def _should_save_model(self, metrics: Dict[str, Dict[str, float]]) -> bool:
        """
        Determine if current model should be saved as curr_best_model.
        
        A model is saved as curr_best_model if:
        1. Both validation and combined returns are positive
        2. Enhanced score is tracked for logging and comparison
        3. At iteration end, curr_best_model is compared against best_model using enhanced scoring
        
        Returns:
            bool: True if model should be saved as curr_best_model
        """
        validation = metrics['validation']
        combined = metrics['combined']
        
        # Reject models with negative returns on either dataset
        if validation['return'] <= 0 or combined['return'] <= 0:
            self.logger.debug(
                f"Model rejected - Negative returns: "
                f"Validation: {validation['return']*100:.2f}%, "
                f"Combined: {combined['return']*100:.2f}%"
            )
            return False
        
        # Calculate enhanced score
        score = self._calculate_enhanced_score(metrics)
        
        # Save if score is positive and improves on previous best
        if score > float('-inf'):
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

            # Get detailed metrics for both datasets
            combined_metrics = get_detailed_metrics(self.combined_env, "Combined Dataset", combined)
            val_metrics = get_detailed_metrics(self.eval_env, "Validation Set", val)
            
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'combined': combined_metrics,
                    'validation': val_metrics
                })
                
                # Create iterations subdirectory
                iterations_dir = os.path.join(self.log_path, "iterations")
                os.makedirs(iterations_dir, exist_ok=True)
                
                # Save iteration results in the subdirectory
                iteration_file = os.path.join(iterations_dir, f"eval_results_iter_{self.iteration}.json")
                with open(iteration_file, "w") as f:
                    json.dump(self.eval_results, f, indent=2)
            
            should_save = False
            
            # Check if model should be saved based on current metrics
            if self._should_save_model(metrics) and self._evaluate_against_previous():
                self.logger.info("Model passed both current iteration and previous best comparison")
                should_save = True
            
            # Save model if criteria met
            if should_save and self.best_model_save_path is not None:
                model_path = os.path.join(self.best_model_save_path, "curr_best_model")
                metrics_path = os.path.join(self.best_model_save_path, "curr_best_metrics.json")
                
                # Save model
                self.model.save(model_path)
                
                # Load existing metrics if available
                try:
                    with open(metrics_path, 'r') as f:
                        all_best_metrics = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_best_metrics = {}
                
                # Calculate enhanced score components for logging
                val_performance = metrics['validation'].get('performance', {})
                rr_score = self._calculate_risk_reward_score(
                    val_performance.get('avg_win_points', 0), 
                    val_performance.get('avg_loss_points', 0)
                )
                balance_score = self._calculate_directional_balance_score(
                    val_performance.get('long_trades', 0),
                    val_performance.get('short_trades', 0)
                )
                base_score = (metrics['validation']['return'] * 0.80 + metrics['combined']['return'] * 0.20)
                consistency = min(metrics['validation']['return'], metrics['combined']['return']) / max(metrics['validation']['return'], metrics['combined']['return']) if metrics['validation']['return'] > 0 and metrics['combined']['return'] > 0 else 0.0
                
                # Create metrics for current iteration
                best_metrics = {
                    'timestep': self.num_timesteps,
                    'training': val_metrics['training'],
                    'combined': combined_metrics,
                    'validation': val_metrics,
                    'scores': metrics['scores'],
                    'enhanced_selection': {
                        'enhanced_score': self.best_score,
                        'base_score': base_score * 100,  # 80/20 weighted
                        'validation_return': metrics['validation']['return'] * 100,
                        'combined_return': metrics['combined']['return'] * 100,
                        'risk_reward_score': rr_score,
                        'balance_score': balance_score,
                        'consistency_score': consistency,
                        'profit_factor_bonus': min(max(0, metrics['validation'].get('profit_factor', 1.0) - 1.0), 2.0) * 0.05,
                        'rr_ratio': val_performance.get('avg_win_points', 0) / abs(val_performance.get('avg_loss_points', -1)) if val_performance.get('avg_loss_points', 0) < 0 else 0,
                        'directional_ratio': max(val_performance.get('long_trades', 1), val_performance.get('short_trades', 1)) / min(val_performance.get('long_trades', 1), val_performance.get('short_trades', 1)) if val_performance.get('long_trades', 0) > 0 and val_performance.get('short_trades', 0) > 0 else 0
                    }
                }
                
                # Add current iteration metrics to collection
                all_best_metrics[f"iteration_{self.iteration}"] = best_metrics
                
                # Save updated metrics
                with open(metrics_path, 'w') as f:
                    json.dump(all_best_metrics, f, indent=2)
                
                # Get latest training stats for debugging
                training_stats = val_metrics['training']

                # Header
                print(f"\n{'='*70}")
                print(f"  🚀 New Best Model Saved (Enhanced Scoring) - Timestep {self.num_timesteps:,d}")
                print(f"{'='*70}")

                # Returns section
                print(f"\n  Returns:")
                print(f"    Combined: {metrics['combined']['return']*100:.2f}%")
                print(f"    Validation: {metrics['validation']['return']*100:.2f}%")

                # Separator
                print(f"\n  {'-'*50}")
                
                # Enhanced selection metrics
                print(f"  Enhanced Selection (Score: {self.best_score:.3f}):")
                print(f"    Base Score (80/20): {base_score*100:.2f}%")
                print(f"      - Validation (80%): {metrics['validation']['return']*100:.2f}%")
                print(f"      - Combined (20%): {metrics['combined']['return']*100:.2f}%")
                print(f"    Risk-Reward Score: {rr_score:.3f} (R:R = {val_performance.get('avg_win_points', 0) / abs(val_performance.get('avg_loss_points', -1)) if val_performance.get('avg_loss_points', 0) < 0 else 0:.2f})")
                print(f"    Balance Score: {balance_score:.3f} (L:{val_performance.get('long_trades', 0)} S:{val_performance.get('short_trades', 0)})")
                print(f"    Consistency: {consistency:.3f}")
                print(f"    Progress: {self.num_timesteps/self.training_timesteps*100:.1f}%")
                
                # Separator
                print(f"\n  {'-'*50}")
                
                print("  Network Stats:")
                print(f"    Value Network:")
                print(f"      Loss: {training_stats['value_loss']:.4f}")
                print(f"      Explained Var: {training_stats['explained_variance']:.2f}")
                print(f"    Policy Network:")
                print(f"      Loss: {training_stats['policy_loss']:.4f}")
                print(f"      Entropy: {training_stats['entropy_loss']:.4f}")
                print(f"      KL Div: {training_stats['approx_kl']:.4f}")

            self.last_time_trigger = self.n_calls
        
        return True
