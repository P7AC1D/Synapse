"""
Regularized Training Utilities - Addressing Overfitting Issues

This module implements the critical fixes identified in the generalization analysis:
1. Validation-based model selection (instead of combined dataset selection)
2. Improved data splitting (70/20/10 instead of 90/10/0)
3. Early stopping based on validation performance
4. Stronger regularization and reduced architecture complexity
5. Enhanced monitoring and validation tracking

Based on analysis showing 1,169% performance gap between training (+1,146%) and validation (-23.7%).
"""

import os
import json
import pandas as pd
import numpy as np
import time
import shutil
import torch as th
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.anti_collapse_callback import AntiCollapseCallback
from configs.regularized_training_config import (
    REGULARIZED_TRAINING_CONFIG, 
    REGULARIZED_POLICY_KWARGS,
    REGULARIZED_MODEL_KWARGS,
    REGULARIZED_DATA_CONFIG,
    REGULARIZED_VALIDATION_CONFIG
)

class RegularizedEvalCallback(BaseCallback):
    """
    Regularized evaluation callback that uses VALIDATION-ONLY model selection.
    
    This is the critical fix: instead of using combined dataset scores, we use
    only validation performance for model selection to prevent overfitting.
    """
    
    def __init__(self, eval_env, train_data, val_data, eval_freq=5000, 
                 best_model_save_path=None, log_path=None, deterministic=True, 
                 verbose=0, iteration=0, training_timesteps=40000):
        super().__init__(verbose)  # Initialize BaseCallback
        
        self.eval_env = eval_env
        self.train_data = train_data
        self.val_data = val_data
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.verbose = verbose
        self.iteration = iteration
        self.training_timesteps = training_timesteps
          # Validation-only tracking
        self.best_validation_score = -float('inf')
        self.best_validation_metrics = None
        self.validation_history = []
        self.no_improvement_count = 0
        self.early_stopping_patience = REGULARIZED_VALIDATION_CONFIG['early_stopping']['patience']
        
        self.n_calls = 0
    
    def _evaluate_on_validation(self) -> Dict[str, Any]:
        """
        Evaluate model ONLY on validation data.
        This is the key change: no combined dataset evaluation.
        """
        try:
            # Reset environment (handle tuple return)
            obs, _ = self.eval_env.reset()
            lstm_states = None
            episode_rewards = []
            episode_actions = []
            
            # Single episode evaluation on validation data
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
                step_count += 1              # Extract metrics from environment
            # Handle different environment wrapper structures
            if hasattr(self.eval_env, 'env'):
                env_metrics = self.eval_env.env.metrics.get_performance_summary()
            else:
                env_metrics = self.eval_env.metrics.get_performance_summary()
              # Calculate validation score (validation return only)
            validation_return = env_metrics.get('return_pct', 0.0) / 100.0
            
            # Additional validation metrics
            validation_metrics = {
                'return': validation_return,
                'total_trades': env_metrics.get('total_trades', 0),
                'win_rate': env_metrics.get('win_rate', 0.0) / 100.0,
                'profit_factor': env_metrics.get('profit_factor', 0.0),
                'max_drawdown': env_metrics.get('max_drawdown_pct', 0.0) / 100.0,
                'sharpe_ratio': env_metrics.get('sharpe_ratio', 0.0),
                'episode_reward': episode_reward,
                'steps': step_count
            }
            
            return validation_metrics
            
        except Exception as e:
            print(f"⚠️ Validation evaluation error: {e}")
            return {
                'return': -1.0,  # Heavily penalize failed evaluations
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 1.0,
                'sharpe_ratio': -1.0,
                'episode_reward': -1000,
                'steps': 0
            }
    
    def _should_save_model(self, validation_metrics: Dict[str, float]) -> bool:
        """
        CRITICAL CHANGE: Model selection based ONLY on validation performance.
        
        No combined dataset evaluation - pure validation-based selection.
        """
        validation_return = validation_metrics['return']
        
        # Only save models with non-negative validation returns
        if validation_return < 0:
            if self.verbose > 0:
                print(f"❌ Model rejected - Negative validation return: {validation_return*100:.2f}%")
            return False
          # Calculate validation score (could add profit factor bonus later)
        validation_score = validation_return
        
        # Save if validation performance improves
        if validation_score > self.best_validation_score:
            self.best_validation_score = validation_score
            self.best_validation_metrics = validation_metrics
            self.no_improvement_count = 0  # Reset early stopping counter
            
            if self.verbose > 0:
                print(f"✅ New best validation score: {validation_score*100:.2f}%")
            return True
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"📊 No improvement - Current: {validation_score*100:.2f}%, "
                      f"Best: {self.best_validation_score*100:.2f}% "
                      f"(No improvement: {self.no_improvement_count}/{self.early_stopping_patience})")
            return False
    
    def _check_early_stopping(self) -> bool:
        """
        Check if early stopping should trigger based on validation performance.
        """
        if not REGULARIZED_VALIDATION_CONFIG['early_stopping']['enabled']:
            return False
            
        if self.no_improvement_count >= self.early_stopping_patience:
            print(f"\n🛑 EARLY STOPPING TRIGGERED - No validation improvement for {self.early_stopping_patience} evaluations")
            return True
        
        return False
    
    def _on_step(self) -> bool:
        """
        Called during training to evaluate and potentially save the model.
        """
        self.n_calls += 1
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate on validation set only
            validation_metrics = self._evaluate_on_validation()
            self.validation_history.append(validation_metrics)
            
            # Log validation performance
            if self.verbose > 0:
                print(f"\n📊 Validation Evaluation (Step {self.n_calls}):")
                print(f"   Return: {validation_metrics['return']*100:.2f}%")
                print(f"   Trades: {validation_metrics['total_trades']}")
                print(f"   Win Rate: {validation_metrics['win_rate']*100:.1f}%")
                print(f"   Profit Factor: {validation_metrics['profit_factor']:.2f}")
                print(f"   Max Drawdown: {validation_metrics['max_drawdown']*100:.1f}%")
            
            # Save model if validation performance is good
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
                        'training_step': self.n_calls,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics_to_save, f, indent=2)
                    
                    print(f"💾 Model saved with validation score: {self.best_validation_score*100:.2f}%")
            
            # Check for early stopping
            if self._check_early_stopping():
                return False  # Stop training
        
        return True  # Continue training


def create_regularized_data_splits(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create regularized data splits with 70/20/10 distribution.
    
    This addresses the critical overfitting issue of 90/10 split.
    """
    n_samples = len(data)
    
    # Use temporal split to maintain time series order
    train_end = int(n_samples * REGULARIZED_DATA_CONFIG['train_split'])
    val_end = int(n_samples * (REGULARIZED_DATA_CONFIG['train_split'] + REGULARIZED_DATA_CONFIG['validation_split']))
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"\n📊 REGULARIZED DATA SPLITS:")
    print(f"   Training: {len(train_data):,} samples ({len(train_data)/n_samples:.1%})")
    print(f"   Validation: {len(val_data):,} samples ({len(val_data)/n_samples:.1%})")
    print(f"   Test: {len(test_data):,} samples ({len(test_data)/n_samples:.1%})")
    print(f"   Total: {n_samples:,} samples")
    
    return train_data, val_data, test_data


def train_regularized_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """
    Walk-forward training with regularization fixes for overfitting.
    
    Key Changes:
    1. Validation-only model selection (no combined dataset)
    2. 70/20/10 data splits instead of 90/10
    3. Early stopping based on validation performance
    4. Reduced architecture complexity
    5. Stronger regularization
    
    Args:
        data: Full dataset for training
        initial_window: Size of initial training window  
        step_size: Step size for moving window forward
        args: Training arguments
        
    Returns:
        RecurrentPPO: Final trained model with regularization
    """
    total_periods = len(data)
    total_iterations = (total_periods - initial_window) // step_size + 1
    
    print(f"\n🛡️ REGULARIZED TRAINING FEATURES:")
    print(f"✅ Validation-only model selection (NO combined dataset bias)")
    print(f"✅ 70/20/10 data splits (improved from 90/10)")
    print(f"✅ Early stopping on validation performance")
    print(f"✅ Reduced architecture: 128x128 networks, 256 LSTM (from 256x256, 512 LSTM)")
    print(f"✅ Learning rate: 0.0005 (reduced from 0.001)")
    print(f"✅ Gradient clipping: 0.5")
    print(f"✅ Weight decay: 1e-3")
    
    results_path = f"../results/{args.seed}"
    os.makedirs(results_path, exist_ok=True)
    best_model_path = os.path.join(results_path, "best_model_regularized.zip")
    
    # Initialize training state
    state_path = os.path.join(results_path, "regularized_training_state.json")
    training_start = 0
    model = None
    
    # Track metrics
    training_metrics = {
        'iterations_completed': 0,
        'early_stops_triggered': 0,
        'best_validation_score': -float('inf'),
        'validation_improvements': 0,
        'total_training_time': 0
    }
    
    try:
        print(f"\n🎯 Starting regularized walk-forward training...")
        print(f"📊 Target iterations: {total_iterations}")
        
        for iteration in range(training_start, total_iterations):
            iteration_start_time = time.time()
            
            # Create regularized data splits for this iteration
            val_size = min(step_size, 2000)  # Limit validation size
            train_size = initial_window - val_size
            
            train_start = iteration * step_size
            train_end = train_start + train_size
            val_end = min(train_end + val_size, total_periods)
            
            if val_end - train_end < val_size * 0.3:
                print(f"\n⚠️ Insufficient validation data at iteration {iteration + 1}, stopping")
                break
            
            # Get iteration data
            iteration_train_data = data.iloc[train_start:train_end].copy()
            iteration_val_data = data.iloc[train_end:val_end].copy()
            
            # Apply regularized data splitting to training data
            train_data, val_data, _ = create_regularized_data_splits(iteration_train_data)
            
            print(f"\n=== REGULARIZED Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
            print(f"=== Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
            print(f"=== Iteration: {iteration+1}/{total_iterations} ===")
            
            # Environment parameters with regularization
            env_params = {
                'initial_balance': getattr(args, 'initial_balance', 10000),
                'balance_per_lot': getattr(args, 'balance_per_lot', 500),
                'random_start': getattr(args, 'random_start', False),
                'point_value': getattr(args, 'point_value', 0.01),
                'min_lots': getattr(args, 'min_lots', 0.01),
                'max_lots': getattr(args, 'max_lots', 1.0),
                'contract_size': getattr(args, 'contract_size', 100000)
            }
            
            # Create environments
            train_env = Monitor(TradingEnv(train_data, **env_params))
            val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
            
            # Get training timesteps (reduced from 50k to 40k)
            current_timesteps = REGULARIZED_TRAINING_CONFIG['total_timesteps']
            
            if model is None:
                print(f"\n🚀 Creating REGULARIZED model...")
                
                # Create regularized model with reduced architecture
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    train_env,
                    policy_kwargs=REGULARIZED_POLICY_KWARGS,
                    device=getattr(args, 'device', 'auto'),
                    seed=getattr(args, 'seed', None),
                    **REGULARIZED_MODEL_KWARGS
                )
            else:
                print(f"\n⚡ Continuing REGULARIZED training with warm start...")
                model.set_env(train_env)
            
            # Create regularized evaluation callback
            regularized_eval_cb = RegularizedEvalCallback(
                val_env,
                train_data=train_data,
                val_data=val_data,
                eval_freq=REGULARIZED_TRAINING_CONFIG['eval_freq'],
                best_model_save_path=results_path,
                verbose=1,
                iteration=iteration,
                training_timesteps=current_timesteps
            )
            
            # Create anti-collapse callback (preserve this important feature)
            anti_collapse_cb = AntiCollapseCallback(
                min_entropy_threshold=-1.0,
                min_trades_per_eval=3,
                collapse_detection_window=3,
                emergency_epsilon=0.8,
                log_path=results_path,
                iteration=iteration,
                verbose=1
            )
            
            callbacks = [
                anti_collapse_cb,
                CustomEpsilonCallback(
                    start_eps=0.15,  # Reduced exploration for regularization
                    end_eps=0.05,
                    decay_timesteps=int(current_timesteps * 0.7),
                    iteration=iteration
                ),
                regularized_eval_cb
            ]
            
            # Train model
            print(f"🎯 Training with {current_timesteps:,} timesteps (REGULARIZED)")
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
                break
            
            # Model selection based on validation performance only
            curr_best_path = os.path.join(results_path, "curr_best_model.zip")
            
            if os.path.exists(curr_best_path):
                print(f"\n💾 Processing regularized model selection...")
                
                # Load validation metrics
                metrics_path = curr_best_path.replace(".zip", "_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        curr_metrics = json.load(f)
                    
                    curr_validation_score = curr_metrics.get('validation_score', 0)
                      # Compare with best model (validation-only comparison)
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
                                print(f"🎯 NEW BEST MODEL: validation score {curr_validation_score*100:.2f}% > {best_validation_score*100:.2f}%")
                            else:
                                print(f"📊 Keeping previous best: validation score {best_validation_score*100:.2f}% >= {curr_validation_score*100:.2f}%")
                        else:
                            # First model with validation metrics - define best_metrics_path
                            best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                            shutil.copy2(curr_best_path, best_model_path)
                            shutil.copy2(metrics_path, best_metrics_path)
                            training_metrics['validation_improvements'] += 1
                            training_metrics['best_validation_score'] = curr_validation_score
                            print(f"🎯 First regularized model saved: validation score {curr_validation_score*100:.2f}%")
                    else:
                        # First model ever - define best_metrics_path
                        best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                        shutil.copy2(curr_best_path, best_model_path)
                        shutil.copy2(metrics_path, best_metrics_path)
                        training_metrics['validation_improvements'] += 1
                        training_metrics['best_validation_score'] = curr_validation_score
                        print(f"🎯 Initial regularized model saved: validation score {curr_validation_score*100:.2f}%")
                
                # Clean up temporary files
                os.remove(curr_best_path)
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)
            else:
                print(f"⚠️ No model met regularized validation criteria for iteration {iteration}")
            
            # Update training metrics
            iteration_time = time.time() - iteration_start_time
            training_metrics['total_training_time'] += iteration_time
            training_metrics['iterations_completed'] = iteration + 1
            
            print(f"\n⚡ REGULARIZED ITERATION SUMMARY:")
            print(f"   Iteration time: {iteration_time/60:.1f} minutes")
            print(f"   Progress: {iteration+1}/{total_iterations} ({(iteration+1)/total_iterations*100:.1f}%)")
            print(f"   Best validation score: {training_metrics['best_validation_score']*100:.2f}%")
            print(f"   Validation improvements: {training_metrics['validation_improvements']}")
            print(f"   Early stops triggered: {training_metrics['early_stops_triggered']}")
            
            # Save regularized training state
            state = {
                'iteration': iteration + 1,
                'best_model_path': best_model_path,
                'timestamp': datetime.now().isoformat(),
                'training_metrics': training_metrics
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
    
    except KeyboardInterrupt:
        print("\n🛑 Regularized training interrupted. Progress saved.")
    except Exception as e:
        print(f"\n❌ Regularized training error: {e}")
    finally:
        # Load final best model
        if os.path.exists(best_model_path):
            model = RecurrentPPO.load(best_model_path)
            print(f"\n✅ Loaded final regularized model from: {best_model_path}")
        
        # Print final summary
        print(f"\n🏁 REGULARIZED TRAINING COMPLETE:")
        print(f"   Iterations completed: {training_metrics['iterations_completed']}/{total_iterations}")
        print(f"   Best validation score: {training_metrics['best_validation_score']*100:.2f}%")
        print(f"   Validation improvements: {training_metrics['validation_improvements']}")
        print(f"   Early stops triggered: {training_metrics['early_stops_triggered']}")
        print(f"   Total training time: {training_metrics['total_training_time']/3600:.1f} hours")
        print(f"   Avg iteration time: {training_metrics['total_training_time']/max(training_metrics['iterations_completed'], 1)/60:.1f} minutes")
    
    return model


def validate_regularization_implementation() -> Dict[str, Any]:
    """
    Validate that all critical overfitting fixes are properly implemented.
    
    This ensures we've addressed all 5 major issues from the analysis.
    """
    validation_results = {
        'data_splitting': False,
        'model_selection': False,
        'regularization': False,
        'architecture': False,
        'early_stopping': False,
        'overall_status': 'FAILED'
    }
    
    issues_found = []
    fixes_verified = []
    
    # 1. Check data splitting
    if (REGULARIZED_DATA_CONFIG['train_split'] == 0.7 and 
        REGULARIZED_DATA_CONFIG['validation_split'] == 0.2 and
        REGULARIZED_DATA_CONFIG['test_split'] == 0.1):
        validation_results['data_splitting'] = True
        fixes_verified.append("✅ Data splitting: 70/20/10 (improved from 90/10)")
    else:
        issues_found.append("❌ Data splitting still uses old ratios")
    
    # 2. Check model selection criterion
    if REGULARIZED_VALIDATION_CONFIG['selection_criterion'] == 'validation_return':
        validation_results['model_selection'] = True
        fixes_verified.append("✅ Model selection: Validation-only (no combined dataset bias)")
    else:
        issues_found.append("❌ Model selection still uses combined dataset")
      # 3. Check regularization
    regularization_checks = [
        REGULARIZED_MODEL_KWARGS['learning_rate'] <= 0.0005,
        REGULARIZED_MODEL_KWARGS['max_grad_norm'] <= 0.5,
        REGULARIZED_POLICY_KWARGS.get('optimizer_kwargs', {}).get('weight_decay', 0) >= 1e-3,
        REGULARIZED_MODEL_KWARGS['batch_size'] <= 64
    ]
    
    if all(regularization_checks):
        validation_results['regularization'] = True
        fixes_verified.append("✅ Regularization: Stronger constraints (LR≤0.0005, grad_clip≤0.5, weight_decay≥1e-3)")
    else:
        issues_found.append("❌ Regularization parameters insufficient")
    
    # 4. Check architecture reduction
    architecture_checks = [
        REGULARIZED_POLICY_KWARGS['lstm_hidden_size'] <= 256,
        REGULARIZED_POLICY_KWARGS['n_lstm_layers'] <= 2,
        all(size <= 128 for size in REGULARIZED_POLICY_KWARGS['net_arch']['pi']),
        all(size <= 128 for size in REGULARIZED_POLICY_KWARGS['net_arch']['vf'])
    ]
    
    if all(architecture_checks):
        validation_results['architecture'] = True
        fixes_verified.append("✅ Architecture: Reduced complexity (LSTM≤256, layers≤2, networks≤128)")
    else:
        issues_found.append("❌ Architecture still too complex")
    
    # 5. Check early stopping
    if (REGULARIZED_VALIDATION_CONFIG['early_stopping']['enabled'] and
        REGULARIZED_VALIDATION_CONFIG['early_stopping']['metric'] == 'validation_return'):
        validation_results['early_stopping'] = True
        fixes_verified.append("✅ Early stopping: Validation-based with patience=10")
    else:
        issues_found.append("❌ Early stopping not properly configured")
    
    # Overall status
    all_fixed = all(validation_results[key] for key in ['data_splitting', 'model_selection', 'regularization', 'architecture', 'early_stopping'])
    
    if all_fixed:
        validation_results['overall_status'] = 'PASSED'
        print(f"\n🎯 REGULARIZATION VALIDATION: ✅ PASSED")
        print(f"\nFixes verified:")
        for fix in fixes_verified:
            print(f"  {fix}")
    else:
        validation_results['overall_status'] = 'FAILED'
        print(f"\n🎯 REGULARIZATION VALIDATION: ❌ FAILED")
        print(f"\nIssues found:")
        for issue in issues_found:
            print(f"  {issue}")
        if fixes_verified:
            print(f"\nFixes verified:")
            for fix in fixes_verified:
                print(f"  {fix}")
    
    return validation_results

# Add RegularizedTrainingManager class after the existing classes
class RegularizedTrainingManager:
    """
    Training manager that wraps RegularizedEvalCallback for testing and validation.
    
    This class provides a unified interface for regularized training management
    and validation metrics testing.
    """
    
    def __init__(self, model, env, eval_env, results_dir="results"):
        """
        Initialize the training manager.
        
        Args:
            model: The model to evaluate
            env: Training environment
            eval_env: Evaluation environment  
            results_dir: Directory for saving results
        """
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.results_dir = results_dir
        
        # Create mock data for the evaluation callback
        import pandas as pd
        import numpy as np
        
        # Create minimal mock data for evaluation
        self.train_data = pd.DataFrame({
            'close': np.random.normal(1.0, 0.01, 100)
        })
        self.val_data = pd.DataFrame({
            'close': np.random.normal(1.0, 0.01, 50) 
        })
        
        # Create the regularized evaluation callback
        self.eval_callback = RegularizedEvalCallback(
            eval_env=eval_env,
            train_data=self.train_data,
            val_data=self.val_data,
            eval_freq=1000,
            best_model_save_path=results_dir,
            verbose=1
        )
        
        # Set the model on the callback
        self.eval_callback.model = model
    
    def _evaluate_on_validation(self) -> Dict[str, Any]:
        """
        Evaluate the model on validation data using the regularized approach.
        
        Returns:
            Dict containing validation metrics
        """
        return self.eval_callback._evaluate_on_validation()
    
    def train(self, total_timesteps: int = 10000):
        """
        Train the model using regularized approach.
        
        Args:
            total_timesteps: Number of timesteps to train
        """
        if hasattr(self.model, 'learn'):
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.eval_callback
            )
        else:
            print("Mock model provided - no actual training performed")
    
    def save_best_model(self, path: str):
        """Save the best model based on validation performance."""
        if hasattr(self.model, 'save'):
            self.model.save(path)
        else:
            print(f"Mock model - would save to {path}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics from the evaluation callback."""
        return {
            'best_validation_score': self.eval_callback.best_validation_score,
            'validation_history': self.eval_callback.validation_history,
            'no_improvement_count': self.eval_callback.no_improvement_count
        }
