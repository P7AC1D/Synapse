"""
Training Utilities for Walk-Forward Optimization

This module provides utilities for training a DRL trading bot using walk-forward optimization.
The implementation includes:
- Data splitting and window management
- Model training and evaluation
- Performance tracking
- Early stopping based on validation performance
"""

import os
import json
import pandas as pd
import numpy as np
import time
import shutil
from datetime import datetime
from typing import Tuple, Dict, Any
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.anti_collapse_callback import AntiCollapseCallback
from configs.config import (
    TRAINING_CONFIG,
    POLICY_KWARGS,
    MODEL_KWARGS,
    VALIDATION_CONFIG
)
from utils.adaptive_validation_utils import AdaptiveValidationManager

def save_validation_state(save_path: str, state: Dict[str, Any]) -> None:
    """
    Save validation state to continue tracking between sessions.
    
    Args:
        save_path: Directory path where to save the state
        state: Dictionary containing validation state to save
    """
    state_path = os.path.join(save_path, "validation_state.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

def load_validation_state(load_path: str) -> Dict[str, Any]:
    """
    Load validation state from previous session.
    
    Args:
        load_path: Directory path where state file is located
        
    Returns:
        Dictionary containing loaded validation state
    """
    state_path = os.path.join(load_path, "validation_state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return None

class EvalCallback(BaseCallback):
    """
    Evaluation callback that tracks model performance during training.
    Uses validation data to assess model performance and save the best models.    """
    
    def __init__(self, eval_env, train_data, val_data, eval_freq=5000, 
                 best_model_save_path=None, log_path=None, deterministic=True, 
                 verbose=0, iteration=0, training_timesteps=40000, use_live_sim_validation=True):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.train_data = train_data
        self.val_data = val_data
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.use_live_sim_validation = use_live_sim_validation
        self.verbose = verbose
        self.iteration = iteration
        self.training_timesteps = training_timesteps
        
        # Create enhanced directory structure
        self._create_directory_structure()
        
        # Initialize adaptive validation if enabled
        self.adaptive_validation = None
        if best_model_save_path and VALIDATION_CONFIG.get('adaptive', {}).get('enabled', False):
            self.adaptive_validation = AdaptiveValidationManager(best_model_save_path, iteration)
            if self.verbose > 0:
                print(f"🧠 Adaptive validation enabled for iteration {iteration}")
        
        # Initialize validation state
        self._initialize_validation_state()
            
        self.early_stopping_patience = VALIDATION_CONFIG['early_stopping']['patience']
        self.n_calls = 0
    
    def _create_directory_structure(self):
        """Create enhanced directory structure for organized storage."""
        if not self.best_model_save_path:
            return
            
        # Create main directories
        self.iterations_dir = os.path.join(self.best_model_save_path, "iterations")
        self.checkpoints_dir = os.path.join(self.best_model_save_path, "checkpoints")
        self.validation_dir = os.path.join(self.best_model_save_path, "validation_results")
        
        # Create directories if they don't exist
        for directory in [self.iterations_dir, self.checkpoints_dir, self.validation_dir]:
            os.makedirs(directory, exist_ok=True)
            
        if self.verbose > 0:
            print(f"📁 Enhanced directory structure created:")
            print(f"   Iterations: {self.iterations_dir}")
            print(f"   Checkpoints: {self.checkpoints_dir}")
            print(f"   Validation: {self.validation_dir}")
    
    def _initialize_validation_state(self):
        """Initialize or load validation state."""
        if self.best_model_save_path:
            # Try to load existing state
            state = load_validation_state(self.best_model_save_path)
            if state:
                loaded_iteration = state.get('iteration', -1)
                self.best_validation_score = state.get('best_validation_score', -float('inf'))
                self.validation_history = state.get('validation_history', [])
                self.best_validation_metrics = state.get('best_validation_metrics', None)
                self.n_calls = state.get('n_calls', 0)
                # Reset validation state if starting a new iteration
                if loaded_iteration != self.iteration:
                    self.no_improvement_count = 0
                    # Start fresh for new iteration - each iteration is independent
                    self.best_validation_score = -float('inf')
                    self.best_validation_metrics = None
                    self.validation_history = []
                    if self.verbose > 0:
                        print(f"🔄 Starting fresh validation for iteration {self.iteration}")
                        print(f"   Previous iteration: {loaded_iteration} (score: {state.get('best_validation_score', 0)*100:.2f}%)")
                        print(f"   Early stopping counter reset (was {state.get('no_improvement_count', 0)})")
                else:
                    self.no_improvement_count = state.get('no_improvement_count', 0)
            else:
                # Initialize new state
                self.best_validation_score = -float('inf')
                self.best_validation_metrics = None
                self.validation_history = []
                self.no_improvement_count = 0

    def _save_current_state(self):
        """Save current validation state."""
        if self.best_model_save_path:
            state = {
                'best_validation_score': self.best_validation_score,
                'validation_history': self.validation_history,
                'no_improvement_count': self.no_improvement_count,
                'iteration': self.iteration,  # Always save current iteration
                'n_calls': self.n_calls,
                'best_validation_metrics': self.best_validation_metrics
            }
            save_validation_state(self.best_model_save_path, state)
    
    def _evaluate_on_validation(self) -> Dict[str, Any]:
        """Evaluate model on validation data."""
        try:
            # Reset environment
            obs, _ = self.eval_env.reset()
            lstm_states = None
            episode_actions = []
            
            # Single episode evaluation
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done and step_count < len(self.val_data):
                # Get action from model
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, deterministic=self.deterministic
                )
                
                # Take step
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_actions.append(action)
                step_count += 1
                
            # Extract metrics from environment
            if hasattr(self.eval_env, 'env'):
                env_metrics = self.eval_env.env.metrics.get_performance_summary()
            else:
                env_metrics = self.eval_env.metrics.get_performance_summary()
            
            validation_return = env_metrics.get('return_pct', 0.0) / 100.0            # Validation metrics
            validation_metrics = {
                'return': validation_return,
                'total_trades': env_metrics.get('total_trades', 0),
                'win_rate': env_metrics.get('win_rate', 0.0) / 100.0,
                'profit_factor': env_metrics.get('profit_factor', 0.0),
                'max_drawdown': env_metrics.get('max_drawdown_pct', 0.0) / 100.0,
                'max_equity_drawdown': env_metrics.get('max_equity_drawdown_pct', 0.0) / 100.0,
                'sharpe_ratio': env_metrics.get('sharpe_ratio', 0.0),
                'episode_reward': episode_reward,
                'steps': step_count,
                'long_trades': env_metrics.get('long_trades', 0),
                'short_trades': env_metrics.get('short_trades', 0),
                'long_win_rate': env_metrics.get('long_win_rate', 0.0) / 100.0,
                'short_win_rate': env_metrics.get('short_win_rate', 0.0) / 100.0,
                'avg_win_points': env_metrics.get('avg_win_points', 0.0),
                'avg_loss_points': env_metrics.get('avg_loss_points', 0.0),
                'avg_win': env_metrics.get('avg_win', 0.0),
                'avg_loss': env_metrics.get('avg_loss', 0.0),
                'win_hold_time': env_metrics.get('win_hold_time', 0.0),
                'loss_hold_time': env_metrics.get('loss_hold_time', 0.0),
                'avg_hold_time': env_metrics.get('avg_hold_time', 0.0)            }
            
            return validation_metrics
            
        except Exception as e:
            print(f"⚠️ Validation evaluation error: {e}")
            return {
                'return': -1.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,                
                'max_drawdown': 1.0,
                'max_equity_drawdown': 1.0,
                'sharpe_ratio': -1.0,
                'episode_reward': -1000,
                'steps': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0.0,
                'short_win_rate': 0.0,
                'avg_win_points': 0.0,
                'avg_loss_points': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_hold_time': 0.0,
                'loss_hold_time': 0.0,
                'avg_hold_time': 0.0            }
    
    def _evaluate_on_validation_live_sim(self) -> Dict[str, Any]:
        """
        Evaluate model on validation data using live trading simulation.
        This method maintains LSTM states properly and simulates real trading conditions.
        """
        try:
            # Create a fresh trading environment for validation
            from trade_model import TradeModel
            
            # Extract underlying environment attributes safely
            def get_env_attr(env, attr_name, default_value):
                """Safely extract environment attribute from potentially wrapped environment."""
                # Try direct access first
                if hasattr(env, attr_name):
                    return getattr(env, attr_name)
                # Try unwrapped environment
                elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, attr_name):
                    return getattr(env.unwrapped, attr_name)
                # Try env attribute (for Monitor wrapper)
                elif hasattr(env, 'env') and hasattr(env.env, attr_name):
                    return getattr(env.env, attr_name)
                # Try going deeper if needed
                elif hasattr(env, 'env') and hasattr(env.env, 'unwrapped') and hasattr(env.env.unwrapped, attr_name):
                    return getattr(env.env.unwrapped, attr_name)
                else:
                    return default_value
            
            # Extract environment parameters with safe defaults
            balance_per_lot = get_env_attr(self.eval_env, 'BALANCE_PER_LOT', 500.0)
            initial_balance = get_env_attr(self.eval_env, 'initial_balance', 10000.0)
            point_value = get_env_attr(self.eval_env, 'POINT_VALUE', 0.001)
            min_lots = get_env_attr(self.eval_env, 'MIN_LOTS', 0.01)
            max_lots = get_env_attr(self.eval_env, 'MAX_LOTS', 200.0)
            contract_size = get_env_attr(self.eval_env, 'CONTRACT_SIZE', 100.0)
            max_loss_points = get_env_attr(self.eval_env, 'max_loss_points', 25000.0)
            
            # Create temporary model wrapper for evaluation
            temp_model = TradeModel(
                model_path=None,  # We'll set the model directly
                balance_per_lot=balance_per_lot,
                initial_balance=initial_balance,
                point_value=point_value,
                min_lots=min_lots,
                max_lots=max_lots,
                contract_size=contract_size
            )
            temp_model.model = self.model  # Use the current training model
            
            # Create validation data DataFrame
            val_data_df = self.val_data.copy()
            if not isinstance(val_data_df.index, pd.DatetimeIndex):
                # Ensure we have a datetime index for the simulation
                val_data_df.index = pd.date_range(start='2020-01-01', periods=len(val_data_df), freq='15min')
              # Initialize LSTM states for live simulation
            if hasattr(temp_model, 'reset_states'):
                temp_model.reset_states()
            temp_model.lstm_states = None
            
            # Create environment for the validation run
            env = TradingEnv(
                data=val_data_df,
                initial_balance=temp_model.initial_balance,
                balance_per_lot=temp_model.balance_per_lot,
                random_start=False,
                point_value=temp_model.point_value,
                min_lots=temp_model.min_lots,
                max_lots=temp_model.max_lots,
                contract_size=temp_model.contract_size,
                max_loss_points=max_loss_points
            )
            
            obs, _ = env.reset()
              # Main prediction loop - similar to predict_single but simplified
            total_steps = 0
            total_reward = 0.0  # Track total episode reward
            current_position = None
            
            while total_steps < len(val_data_df):
                try:
                    # Get prediction with maintained LSTM states
                    action, new_lstm_states = temp_model.model.predict(
                        obs,
                        state=temp_model.lstm_states,
                        deterministic=True
                    )
                    temp_model.lstm_states = new_lstm_states  # Maintain states like in live trading
                    
                    # Convert action to discrete value
                    if isinstance(action, np.ndarray):
                        action_value = int(action.item())
                    else:
                        action_value = int(action)
                    discrete_action = action_value % 4
                    
                    # Force HOLD if trying to open new position while one exists
                    if env.current_position is not None and discrete_action in [1, 2]:  # Buy or Sell
                        discrete_action = 0  # Force HOLD
                      # Execute step
                    obs, reward, done, truncated, info = env.step(discrete_action)
                    total_reward += reward  # Accumulate episode reward
                    total_steps += 1
                    
                    if done or truncated:
                        break
                        
                except Exception as step_e:
                    # Continue on step errors like in predict_single
                    total_steps += 1
                    continue
            
            # Handle any open position at the end
            if env.current_position:
                env.current_step = min(total_steps - 1, len(val_data_df) - 1)
                pnl, trade_info = env.action_handler.close_position()
                if pnl != 0:
                    env.trades.append(trade_info)
                    env.metrics.add_trade(trade_info)
                    env.metrics.update_balance(pnl)
              # Calculate metrics using the same method as live trading
            results = temp_model._calculate_backtest_metrics(env, total_steps, total_reward)
            
            # Convert to validation format
            validation_metrics = {
                'return': results.get('return_pct', 0.0) / 100.0,
                'total_trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0.0) / 100.0,
                'profit_factor': results.get('profit_factor', 0.0),
                'max_drawdown': results.get('max_drawdown_pct', 0.0) / 100.0,
                'max_equity_drawdown': results.get('max_equity_drawdown_pct', 0.0) / 100.0,
                'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                'episode_reward': total_reward,  # Now properly tracked
                'steps': total_steps,
                'long_trades': results.get('long_trades', 0),
                'short_trades': results.get('short_trades', 0),
                'long_win_rate': results.get('long_win_rate', 0.0) / 100.0,
                'short_win_rate': results.get('short_win_rate', 0.0) / 100.0,
                'avg_win_points': results.get('avg_win_points', 0.0),
                'avg_loss_points': results.get('avg_loss_points', 0.0),
                'avg_win': results.get('avg_win', 0.0),
                'avg_loss': results.get('avg_loss', 0.0),
                'win_hold_time': results.get('win_hold_time', 0.0),
                'loss_hold_time': results.get('loss_hold_time', 0.0),
                'avg_hold_time': results.get('avg_hold_time', 0.0)
            }
            
            return validation_metrics
            
        except Exception as e:
            print(f"⚠️ Live simulation validation error: {e}")
            # Return the original validation method as fallback
            return self._evaluate_on_validation()

    def _should_save_model(self, validation_metrics: Dict[str, float]) -> bool:
        """Determine if model should be saved based on validation performance."""
        validation_return = validation_metrics['return']
          # Use adaptive validation if available
        if self.adaptive_validation:
            decision_info = self.adaptive_validation.should_save_model(
                validation_metrics, self.best_validation_score
            )
            should_save = decision_info.get('should_save', False)
            
            if should_save:
                # Update best scores
                score = decision_info.get('composite_score', validation_return)
                if score > self.best_validation_score:
                    self.best_validation_score = score
                    self.best_validation_metrics = validation_metrics
                    self.no_improvement_count = 0
                    
                    if self.verbose > 0:
                        threshold = decision_info.get('adaptive_threshold', 0)
                        print(f"✅ Enhanced validation - New best score: {score*100:.2f}%")
                        print(f"   Return: {validation_return*100:.2f}% (threshold: {threshold*100:.2f}%)")
                        print(f"   Composite score: {decision_info.get('composite_score', 0)*100:.2f}%")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"✅ Enhanced validation - Model saved (meets criteria)")
                        print(f"   Return: {validation_return*100:.2f}%, Score: {score*100:.2f}%")
            else:
                self.no_improvement_count += 1
                if self.verbose > 0:
                    reasoning = decision_info.get('reasoning', 'Unknown')
                    threshold = decision_info.get('adaptive_threshold', 0)
                    print(f"❌ Enhanced validation - Model rejected")
                    print(f"   Return: {validation_return*100:.2f}% (threshold: {threshold*100:.2f}%)")
                    print(f"   Reason: {reasoning}")
                    print(f"   No improvement: {self.no_improvement_count}/{self.early_stopping_patience}")
            
            # Save validation state after processing
            self._save_current_state()
            return should_save
        
        # Fallback to original logic if adaptive validation not available
        # Only save models with non-negative validation returns
        if validation_return < 0:
            self.no_improvement_count += 1
            # Save state after rejection to maintain consistency
            self._save_current_state()
            
            if self.verbose > 0:
                print(f"❌ Model rejected - Negative return: {validation_return*100:.2f}%")
            return False
        
        # Calculate validation score
        validation_score = validation_return
        
        # Save if validation performance improves
        if validation_score > self.best_validation_score:
            self.best_validation_score = validation_score
            self.best_validation_metrics = validation_metrics
            self.no_improvement_count = 0
              # Save validation state after improvement
            self._save_current_state()
            
            if self.verbose > 0:
                print(f"✅ New best score: {validation_score*100:.2f}%")
            return True
        else:
            self.no_improvement_count += 1
            self._save_current_state()
            
            if self.verbose > 0:
                print(f"📊 No improvement - Current: {validation_score*100:.2f}%, "
                      f"Best: {self.best_validation_score*100:.2f}% "
                      f"(No improvement: {self.no_improvement_count}/{self.early_stopping_patience})")
            return False
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if not VALIDATION_CONFIG['early_stopping']['enabled']:
            return False
            
        if self.no_improvement_count >= self.early_stopping_patience:
            print(f"\n🛑 EARLY STOPPING - No improvement for {self.early_stopping_patience} evaluations")
            return True
        
        return False
    
    def _on_step(self) -> bool:
        """Evaluation logic called during training."""
        self.n_calls += 1
        
        # Use num_timesteps (environment steps) instead of n_calls for proper eval frequency
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            # Debug: Log evaluation trigger
            print(f"🔍 Evaluation triggered: n_calls={self.n_calls}, eval_freq={self.eval_freq}, timestep={self.num_timesteps}")            # Evaluate on validation set using appropriate method
            if self.use_live_sim_validation:
                validation_metrics = self._evaluate_on_validation_live_sim()
                if self.verbose > 0:
                    print(f"🔄 Using live trading simulation for validation")
            else:
                validation_metrics = self._evaluate_on_validation()
                if self.verbose > 0:
                    print(f"📊 Using standard environment-based validation")
            self.validation_history.append(validation_metrics)
            
            # Save detailed validation results for analysis
            self._save_validation_results(validation_metrics)
              # Save checkpoint model for analysis
            self._save_checkpoint_model()
              # Log validation performance
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"📊 VALIDATION RESULTS (Step {self.num_timesteps:,d})")
                print(f"{'='*60}")
                
                # Performance Overview Section
                print(f"\n💰 PERFORMANCE OVERVIEW:")
                print(f"   Return: {validation_metrics['return']*100:.2f}%")
                print(f"   Total Reward: {validation_metrics['episode_reward']:.2f}")
                print(f"   Sharpe Ratio: {validation_metrics['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {validation_metrics['max_drawdown']*100:.1f}% (Equity: {validation_metrics['max_equity_drawdown']*100:.1f}%)")
                
                # Trading Statistics Section
                print(f"\n📈 TRADING STATISTICS:")
                print(f"   Total Trades: {validation_metrics['total_trades']}")
                print(f"   Overall Win Rate: {validation_metrics['win_rate']*100:.1f}%")
                print(f"   Profit Factor: {validation_metrics['profit_factor']:.2f}")
                print(f"   Average Win: ${validation_metrics['avg_win']:.2f} ({validation_metrics['avg_win_points']:.1f} points)")
                print(f"   Average Loss: ${validation_metrics['avg_loss']:.2f} ({validation_metrics['avg_loss_points']:.1f} points)")
                
                # Directional Analysis Section
                print(f"\n🎯 DIRECTIONAL ANALYSIS:")
                print(f"   Long Trades: {validation_metrics['long_trades']} ({validation_metrics['long_win_rate']*100:.1f}% win)")
                print(f"   Short Trades: {validation_metrics['short_trades']} ({validation_metrics['short_win_rate']*100:.1f}% win)")
                
                # Timing Analysis Section
                print(f"\n⏱️ TIMING ANALYSIS:")
                print(f"   Average Hold Time: {validation_metrics['avg_hold_time']:.1f} bars")
                print(f"   Winning Trades Hold: {validation_metrics['win_hold_time']:.1f} bars")
                print(f"   Losing Trades Hold: {validation_metrics['loss_hold_time']:.1f} bars")
                
                # Execution Summary
                print(f"\n🔄 EXECUTION SUMMARY:")
                print(f"   Steps Completed: {validation_metrics['steps']:,d}")
                print(f"   Data Utilization: {(validation_metrics['steps'] / len(self.val_data) * 100):.1f}%")
                
                print(f"{'='*60}\n")
            
            # Save model if performance is good
            if self._should_save_model(validation_metrics):
                if self.best_model_save_path:
                    # Save as current best model
                    curr_best_path = os.path.join(self.best_model_save_path, "curr_best_model.zip")
                    self.model.save(curr_best_path)
                      # Save validation metrics
                    metrics_path = curr_best_path.replace(".zip", "_metrics.json")
                    metrics_to_save = {
                        'validation_score': self.best_validation_score,
                        'validation_metrics': validation_metrics,
                        'iteration': self.iteration,
                        'training_step': self.num_timesteps,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics_to_save, f, indent=2)
                    
                    print(f"💾 Model saved with score: {self.best_validation_score*100:.2f}%")
            
            # Check for early stopping
            if self._check_early_stopping():
                return False  # Stop training
        
        return True  # Continue training

    def _save_validation_results(self, validation_metrics: Dict[str, Any]):
        """Save detailed validation results for analysis."""
        if not self.best_model_save_path:
            return
              # Create comprehensive validation result
        result = {
            'iteration': self.iteration,
            'training_step': self.num_timesteps,
            'timestamp': datetime.now().isoformat(),
            'progress': {
                'step_progress': self.num_timesteps / self.training_timesteps * 100,
                'eval_number': len(self.validation_history),
                'no_improvement_count': self.no_improvement_count,
                'early_stopping_patience': self.early_stopping_patience
            },
            'validation_metrics': validation_metrics,
            'best_validation_score': self.best_validation_score,
            'adaptive_validation': None
        }
        
        # Add adaptive validation info if available
        if self.adaptive_validation:
            try:
                result['adaptive_validation'] = self.adaptive_validation.get_diagnostic_info()
            except:
                result['adaptive_validation'] = {'status': 'error_getting_info'}
          # Save to iterations directory
        iteration_file = os.path.join(
            self.iterations_dir, 
            f"iteration_{self.iteration}_step_{self.num_timesteps}_validation.json"
        )
        with open(iteration_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save to validation results directory with timestamped name
        validation_file = os.path.join(
            self.validation_dir,
            f"validation_step_{self.num_timesteps:06d}.json"
        )
        with open(validation_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if self.verbose > 1:
            print(f"📁 Validation results saved:")
            print(f"   Iteration: {iteration_file}")
            print(f"   Validation: {validation_file}")
    
    def _save_checkpoint_model(self):
        """Save checkpoint model for analysis and recovery."""
        if not self.best_model_save_path:
            return
              # Create checkpoint filename with detailed info
        checkpoint_name = f"checkpoint_iter_{self.iteration}_step_{self.num_timesteps:06d}.zip"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        
        # Save model checkpoint
        self.model.save(checkpoint_path)
        
        # Create checkpoint metadata
        metadata = {
            'iteration': self.iteration,
            'training_step': self.num_timesteps,
            'timestamp': datetime.now().isoformat(),
            'model_path': checkpoint_path,
            'validation_history_length': len(self.validation_history),
            'best_validation_score': self.best_validation_score,
            'no_improvement_count': self.no_improvement_count,
            'progress_percent': self.num_timesteps / self.training_timesteps * 100
        }
        
        # Save metadata alongside model
        metadata_path = checkpoint_path.replace('.zip', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose > 1:
            print(f"🔄 Checkpoint saved: {checkpoint_name}")
    
    # ...existing code...
def create_data_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create data splits for training, validation, and testing.
    
    ⚠️ DEPRECATED FOR WALK-FORWARD OPTIMIZATION ⚠️
    This function is for traditional ML training only. 
    WFO training should use full iteration windows without internal splits.
    
    Args:
        data: Full dataset for training
        
    Returns:
        Tuple of training, validation, and test data
    """
    n_samples = len(data)
    
    # Create temporal splits
    train_end = int(n_samples * 0.7)  # 70% training
    val_end = int(n_samples * 0.9)    # 20% validation
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"\n📊 DATA SPLITS:")
    print(f"   Training: {len(train_data):,} samples ({len(train_data)/n_samples:.1%})")
    print(f"   Validation: {len(val_data):,} samples ({len(val_data)/n_samples:.1%})")
    print(f"   Test: {len(test_data):,} samples ({len(test_data)/n_samples:.1%})")
    print(f"   Total: {n_samples:,} samples")
    
    return train_data, val_data, test_data

def train_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """
    Walk-forward training implementation - FIXED for proper WFO.
    
    ✅ ARCHITECTURAL FIX APPLIED:
    - Removed double-split issue (no more create_data_splits() within iterations)
    - Uses FULL iteration windows for training (no data waste)
    - Implements proper temporal validation (next period out-of-sample)
    - Eliminates the 10% test data waste per iteration
    - Prevents temporal leakage through proper data separation
    
    Args:
        data: Full dataset for training
        initial_window: Size of initial training window
        step_size: Step size for moving window forward
        args: Training arguments
          Returns:
        RecurrentPPO: Final trained model
    """
    total_periods = len(data)
    total_iterations = (total_periods - initial_window) // step_size + 1
    
    # Create results path using proper cross-platform path handling
    results_path = os.path.join("..", "results", str(args.seed))
    os.makedirs(results_path, exist_ok=True)
    best_model_path = os.path.join(results_path, "best_model.zip")
    
    # Initialize or load training state
    state_path = os.path.join(results_path, "training_state.json")
    training_start = 0
    model = None
    
    # Initialize default training metrics
    training_metrics = {
        'iterations_completed': 0,
        'early_stops_triggered': 0,
        'best_validation_score': -float('inf'),
        'validation_improvements': 0,
        'total_training_time': 0
    }
    
    # Try to load existing training state
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                saved_state = json.load(f)
                training_start = saved_state.get('iteration', 0)
                training_metrics = saved_state.get('training_metrics', training_metrics)
                print(f"\n✅ Restored training state from iteration {training_start}")
                print(f"   Best validation score: {training_metrics['best_validation_score']*100:.2f}%")
                print(f"   Validation improvements: {training_metrics['validation_improvements']}")
                
                # Try to load best model if it exists
                best_model_path = saved_state.get('best_model_path')
                if best_model_path and os.path.exists(best_model_path):
                    model = RecurrentPPO.load(best_model_path)
                    print(f"✅ Loaded best model from: {best_model_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load training state - {e}")
            print("Starting fresh training run")
    
    try:
        if training_start == 0:
            print(f"\n🎯 Starting new walk-forward training...")
        else:
            print(f"\n⚡ Resuming training from iteration {training_start}...")
        print(f"📊 Target iterations: {total_iterations}")
        
        for iteration in range(training_start, total_iterations):
            iteration_start_time = time.time()            # Create data splits for this iteration
            # Use validation_split from config, training gets the remainder
            val_ratio = TRAINING_CONFIG['validation_split']  # e.g., 0.2 (20%)
            train_ratio = 1.0 - val_ratio  # e.g., 0.8 (80%) - automatically calculated
            
            val_size = int(initial_window * val_ratio)  # 20% for validation (~3,456 samples)
            train_size = int(initial_window * train_ratio)  # 80% for training (~13,824 samples)
            
            train_start = iteration * step_size
            train_end = train_start + train_size
            val_end = min(train_end + val_size, total_periods)
            
            if val_end - train_end < val_size * 0.5:  # Require at least 50% of intended validation size
                print(f"\n⚠️ Insufficient validation data at iteration {iteration + 1} ({val_end - train_end} < {val_size * 0.5:.0f}), stopping")
                break
            
            # Get iteration data - PROPER WFO Implementation
            # Use FULL iteration window for training (no internal splits)
            train_data = data.iloc[train_start:train_end].copy()
            
            # Use NEXT temporal period for validation (proper WFO)
            val_data = data.iloc[train_end:val_end].copy()
            
            print(f"📊 WFO Window {iteration + 1}:")
            print(f"   Training: {len(train_data):,} samples ({train_start}-{train_end}) [{train_ratio*100:.0f}%]")
            print(f"   Validation: {len(val_data):,} samples ({train_end}-{val_end}) [{val_ratio*100:.0f}%]")
            print(f"   ✅ Using config-based splits (train: {train_ratio}, val: {val_ratio})")
            print(f"   ✅ Temporal separation - No future leakage")
            
            # Environment parameters
            env_params = {
                'initial_balance': getattr(args, 'initial_balance', 10000),
                'balance_per_lot': getattr(args, 'balance_per_lot', 500),
                'random_start': getattr(args, 'random_start', False),
                'point_value': getattr(args, 'point_value', 0.01),
                'min_lots': getattr(args, 'min_lots', 0.01),
                'max_lots': getattr(args, 'max_lots', 1.0),
                'contract_size': getattr(args, 'contract_size', 100000),
                'max_loss_points': getattr(args, 'max_loss_points', 25000.0)
            }
            
            # Create environments
            train_env = Monitor(TradingEnv(train_data, **env_params))
            val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
              # Get training timesteps 
            current_timesteps = TRAINING_CONFIG['total_timesteps']
            if model is None:
                # Check for warm-start model
                warm_start_path = getattr(args, 'warm_start_model_path', None)
                warm_start_lr = getattr(args, 'warm_start_learning_rate', None)
                
                if warm_start_path and os.path.exists(warm_start_path):
                    print(f"\n� Loading warm-start model from: {warm_start_path}")
                    
                    try:
                        # Load the existing model
                        model = RecurrentPPO.load(warm_start_path, env=train_env)
                        
                        # Override learning rate if specified
                        if warm_start_lr:
                            print(f"🎯 Overriding learning rate: {model.learning_rate:.2e} → {warm_start_lr:.2e}")
                            model.learning_rate = warm_start_lr
                            # Also update the optimizer's learning rate
                            for param_group in model.policy.optimizer.param_groups:
                                param_group['lr'] = warm_start_lr
                        else:
                            print(f"📊 Using original learning rate: {model.learning_rate:.2e}")
                        
                        print(f"✅ Warm-start model loaded successfully")
                        
                        # Clear any existing validation state for fresh start
                        validation_state_file = os.path.join(results_path, 'validation_state.json')
                        if os.path.exists(validation_state_file):
                            os.remove(validation_state_file)
                            print(f"🗑️ Cleared previous validation state for fresh warm-start")
                            
                    except Exception as e:
                        print(f"❌ Failed to load warm-start model: {e}")
                        print(f"💡 Falling back to creating new model...")
                        model = None  # Force creation of new model below
                        
                elif warm_start_path:
                    print(f"❌ Warm-start model file not found: {warm_start_path}")
                    print(f"💡 Creating new model instead...")
                
                # Create new model if warm-start failed or wasn't requested
                if model is None:
                    print(f"\n�� Creating new model...")
                    # Clear any existing validation state for fresh start
                    validation_state_file = os.path.join(results_path, 'validation_state.json')
                    if os.path.exists(validation_state_file):
                        os.remove(validation_state_file)
                        print(f"🗑️ Cleared previous validation state for fresh start")
                    
                    # Prepare model kwargs (handle warm-start learning rate override for new models)
                    model_kwargs = MODEL_KWARGS.copy()
                    if warm_start_lr:
                        print(f"🎯 Using warm-start learning rate for new model: {warm_start_lr:.2e}")
                        model_kwargs['learning_rate'] = warm_start_lr
                    
                    model = RecurrentPPO(
                        "MlpLstmPolicy",
                        train_env,
                        policy_kwargs=POLICY_KWARGS,
                        device=getattr(args, 'device', 'auto'),
                        seed=getattr(args, 'seed', None),
                        **model_kwargs
                    )
            else:
                print(f"\n⚡ Continuing training with warm start...")
                model.set_env(train_env)
            
            # Create evaluation callback
            eval_cb = EvalCallback(
                val_env,
                train_data=train_data,
                val_data=val_data,
                eval_freq=TRAINING_CONFIG['eval_freq'],
                best_model_save_path=results_path,
                verbose=1,
                iteration=iteration,
                training_timesteps=current_timesteps,
                use_live_sim_validation=VALIDATION_CONFIG.get('use_live_sim_validation', True)
            )
            
            # Debug: Print eval frequency configuration
            print(f"🔧 Evaluation Configuration:")
            print(f"   Config eval_freq: {TRAINING_CONFIG['eval_freq']}")
            print(f"   EvalCallback eval_freq: {eval_cb.eval_freq}")
            print(f"   Current timesteps: {current_timesteps}")
            print(f"   Step size: {step_size}")
            
            # Create anti-collapse callback
            anti_collapse_cb = AntiCollapseCallback(
                min_entropy_threshold=-1.0,
                min_trades_per_eval=3,
                collapse_detection_window=3,
                emergency_epsilon=0.8,
                log_path=results_path,
                iteration=iteration,
                verbose=1
            )
            
            # Determine if this is warm-start mode
            warm_start_mode = hasattr(args, 'warm_start_model_path') and args.warm_start_model_path is not None
            
            # Create epsilon callback with appropriate parameters
            if warm_start_mode:
                epsilon_cb = CustomEpsilonCallback(
                    start_eps=0.15,  # Default will be overridden by warm-start config
                    end_eps=0.05,    # Default will be overridden by warm-start config
                    decay_timesteps=int(current_timesteps * 0.7),
                    iteration=iteration,
                    warm_start_mode=True,
                    warm_start_eps_start=getattr(args, 'warm_start_eps_start', None),
                    warm_start_eps_end=getattr(args, 'warm_start_eps_end', None),
                    warm_start_eps_min=getattr(args, 'warm_start_eps_min', None)
                )
            else:
                epsilon_cb = CustomEpsilonCallback(
                    start_eps=0.15,
                    end_eps=0.05,
                    decay_timesteps=int(current_timesteps * 0.7),
                    iteration=iteration,
                    warm_start_mode=False
                )
            
            callbacks = [
                anti_collapse_cb,
                epsilon_cb,
                eval_cb
            ]
            
            # Train model
            print(f"🎯 Training with {current_timesteps:,} timesteps")
            training_successful = True
            
            try:
                model.learn(
                    total_timesteps=current_timesteps,
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=True
                )
            except Exception as e:
                if "early stopping" in str(e).lower():
                    print(f"✅ Early stopping triggered: {e}")
                    training_metrics['early_stops_triggered'] += 1
                    training_successful = True
                else:
                    print(f"❌ Training error: {e}")
                    training_successful = False
            
            if not training_successful:
                break            # Model selection - IMPROVED WFO Model Selection
            curr_best_path = os.path.join(results_path, "curr_best_model.zip")
            
            if os.path.exists(curr_best_path):
                # Define metrics_path early to avoid "referenced before assignment" error
                metrics_path = curr_best_path.replace(".zip", "_metrics.json")
                
                # Check if improved model selection is enabled
                use_improved_selection = VALIDATION_CONFIG.get('wfo_model_selection', {}).get('enabled', True)
                selection_strategy = VALIDATION_CONFIG.get('wfo_model_selection', {}).get('strategy', 'ensemble_validation')
                
                # Override with command line arguments if provided
                if hasattr(args, 'disable_improved_selection') and args.disable_improved_selection:
                    use_improved_selection = False
                elif hasattr(args, 'model_selection'):
                    if args.model_selection == 'legacy':
                        use_improved_selection = False
                    else:
                        selection_strategy = args.model_selection
                
                if use_improved_selection:
                    # Import improved model selection (with fallback)
                    try:
                        from utils.wfo_model_selection import apply_improved_model_selection
                        
                        # Calculate current position in dataset for validation sets
                        current_end_idx = min(train_end + val_size, len(data))
                        
                        # Apply improved model selection with cross-validation
                        model_updated = apply_improved_model_selection(
                            results_path=results_path,
                            current_model_path=curr_best_path,
                            best_model_path=best_model_path,
                            data=data,
                            current_end_idx=current_end_idx,
                            env_params=env_params,
                            iteration=iteration,
                            strategy=selection_strategy
                        )
                        
                        if model_updated:
                            training_metrics['validation_improvements'] += 1
                            # Note: best_validation_score tracking is now handled by the selector
                            
                    except ImportError as e:
                        print(f"⚠️ Could not import improved model selection: {e}")
                        fallback_enabled = VALIDATION_CONFIG.get('wfo_model_selection', {}).get('fallback_to_legacy', True)
                        if fallback_enabled:
                            print("   Falling back to legacy validation comparison...")
                            use_improved_selection = False
                        else:
                            print("   Fallback disabled - skipping model selection")
                            
                if not use_improved_selection:                    # Legacy comparison with configurable warnings
                    warn_about_legacy = VALIDATION_CONFIG.get('wfo_model_selection', {}).get('warn_about_legacy', True)
                    
                    if warn_about_legacy:
                        print(f"💾 Processing model selection (LEGACY - comparing different validation periods)...")
                        print(f"⚠️  WARNING: This compares models on different time periods!")
                        print(f"⚠️  Results may not be meaningful for walk-forward optimization.")
                        print(f"⚠️  Consider enabling improved model selection in config.")
                    else:
                        print(f"💾 Processing model selection...")
                    
                    # Load validation metrics (metrics_path already defined above)
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            curr_metrics = json.load(f)
                        
                        curr_validation_score = curr_metrics.get('validation_score', 0)
                        
                        # Compare with best model
                        if os.path.exists(best_model_path):
                            best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                            if os.path.exists(best_metrics_path):
                                with open(best_metrics_path, 'r') as f:
                                    best_metrics = json.load(f)
                                
                                best_validation_score = best_metrics.get('validation_score', 0)
                                
                                if curr_validation_score > best_validation_score:
                                    # New best model based on validation
                                    shutil.copy2(curr_best_path, best_model_path)
                                    shutil.copy2(metrics_path, best_metrics_path)
                                    training_metrics['validation_improvements'] += 1
                                    training_metrics['best_validation_score'] = curr_validation_score
                                    print(f"🎯 NEW BEST MODEL: {curr_validation_score*100:.2f}% > {best_validation_score*100:.2f}%")
                                else:
                                    print(f"📊 Keeping previous best: {best_validation_score*100:.2f}% >= {curr_validation_score*100:.2f}%")
                            else:
                                best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                                shutil.copy2(curr_best_path, best_model_path)
                                shutil.copy2(metrics_path, best_metrics_path)
                                training_metrics['validation_improvements'] += 1
                                training_metrics['best_validation_score'] = curr_validation_score
                                print(f"🎯 First model saved: {curr_validation_score*100:.2f}%")
                        else:
                            best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                            shutil.copy2(curr_best_path, best_model_path)
                            shutil.copy2(metrics_path, best_metrics_path)
                            training_metrics['validation_improvements'] += 1
                            training_metrics['best_validation_score'] = curr_validation_score
                            print(f"🎯 Initial model saved: {curr_validation_score*100:.2f}%")
                
                # Clean up temporary files
                os.remove(curr_best_path)
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)
            else:
                print(f"⚠️ No model met validation criteria for iteration {iteration}")
            
            # Update training metrics
            iteration_time = time.time() - iteration_start_time
            training_metrics['total_training_time'] += iteration_time
            training_metrics['iterations_completed'] = iteration + 1
            
            print(f"\n⚡ ITERATION SUMMARY:")
            print(f"   Iteration time: {iteration_time/60:.1f} minutes")
            print(f"   Progress: {iteration+1}/{total_iterations} ({(iteration+1)/total_iterations*100:.1f}%)")
            print(f"   Best score: {training_metrics['best_validation_score']*100:.2f}%")
            print(f"   Improvements: {training_metrics['validation_improvements']}")
            print(f"   Early stops: {training_metrics['early_stops_triggered']}")
            
            # Save training state
            state = {
                'iteration': iteration + 1,
                'best_model_path': best_model_path,
                'timestamp': datetime.now().isoformat(),
                'training_metrics': training_metrics
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted. Progress saved.")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    finally:
        # Load final best model
        if os.path.exists(best_model_path):
            model = RecurrentPPO.load(best_model_path)
            print(f"\n✅ Loaded final model from: {best_model_path}")
        
        # Print final summary
        print(f"\n🏁 TRAINING COMPLETE:")
        print(f"   Iterations completed: {training_metrics['iterations_completed']}/{total_iterations}")
        print(f"   Best score: {training_metrics['best_validation_score']*100:.2f}%")
        print(f"   Improvements: {training_metrics['validation_improvements']}")
        print(f"   Early stops: {training_metrics['early_stops_triggered']}")
        print(f"   Total time: {training_metrics['total_training_time']/3600:.1f} hours")
        print(f"   Avg iteration: {training_metrics['total_training_time']/max(training_metrics['iterations_completed'], 1)/60:.1f} minutes")
    
    return model
