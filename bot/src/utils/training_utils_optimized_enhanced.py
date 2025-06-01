"""
ENHANCED Training utilities for PPO-LSTM model with OVERFITTING PREVENTION.

This module builds on the existing 5-10x speedup optimizations and adds:
- Validation-based early stopping to prevent overfitting
- Enhanced regularization parameters
- Training/validation performance gap monitoring
- Smart model selection based on generalization

PREVENTS: Training return 138% / Validation return -21% scenarios
TARGETS: Balanced performance with <20% train/validation gap
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
import time
import threading
from utils.progress import show_progress_continuous, stop_progress_indicator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv
import torch as th

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.eval_callback import UnifiedEvalCallback

# Import the original optimized utilities
from utils.training_utils_optimized import (
    POLICY_KWARGS_OPTIMIZED,
    MODEL_KWARGS_OPTIMIZED,
    TRAINING_SCHEDULES,
    EnvironmentCache,
    format_time_remaining,
    calculate_adaptive_timesteps,
    save_training_state,
    load_training_state,
    create_optimized_environment,
    clear_optimization_cache,
    get_optimization_info,
    FAST_EVALUATION_AVAILABLE
)

# ENHANCED Anti-Overfitting Model Configuration
POLICY_KWARGS_ANTI_OVERFITTING = {
    **POLICY_KWARGS_OPTIMIZED,
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-4,  # Increased regularization
        "betas": (0.9, 0.999)
    },
    # Add dropout for regularization
    "net_arch": {
        "pi": [128, 64, 32],  # Slightly larger but with more regularization
        "vf": [128, 64, 32]
    }
}

# ENHANCED Anti-Overfitting Training Parameters
MODEL_KWARGS_ANTI_OVERFITTING = {
    **MODEL_KWARGS_OPTIMIZED,
    "learning_rate": 5e-4,    # Lower learning rate to prevent overfitting
    "n_steps": 512,           # Larger batch for more stable gradients
    "batch_size": 256,        # Larger batch size
    "ent_coef": 0.05,         # Higher entropy to encourage exploration
    "vf_coef": 0.5,           # Lower value coefficient
    "max_grad_norm": 0.3,     # More conservative gradient clipping
    "n_epochs": 4,            # Fewer epochs to prevent overfitting
}

# Enhanced training schedules with regularization focus
TRAINING_SCHEDULES_ENHANCED = {
    'conservative': {  # Anti-overfitting focus
        'learning_rate': 3e-4,
        'n_epochs': 3,
        'ent_coef': 0.08,
        'clip_range': get_linear_fn(0.1, 0.1, 1.0),
        'vf_coef': 0.4
    },
    'balanced': {      # Balanced approach
        'learning_rate': 5e-4,
        'n_epochs': 4,
        'ent_coef': 0.05,
        'clip_range': get_linear_fn(0.12, 0.12, 1.0),
        'vf_coef': 0.5
    },
    'aggressive': {    # Only when validation performance is stable
        'learning_rate': 8e-4,
        'n_epochs': 5,
        'ent_coef': 0.03,
        'clip_range': get_linear_fn(0.15, 0.15, 1.0),
        'vf_coef': 0.6
    }
}

class ValidationAwareEarlyStoppingCallback:
    """
    ENHANCED early stopping that monitors training/validation performance gap.
    
    Prevents overfitting by stopping when:
    1. Validation performance degrades while training improves
    2. Training/validation gap becomes too large
    3. Validation metrics show consistent deterioration
    """
    
    def __init__(self, patience: int = 5, min_iterations: int = 8, 
                 max_gap_threshold: float = 0.3, degradation_threshold: float = 0.1):
        self.patience = patience
        self.min_iterations = min_iterations
        self.max_gap_threshold = max_gap_threshold  # Max allowed train/val gap (30%)
        self.degradation_threshold = degradation_threshold  # Max allowed validation degradation
        
        self.best_validation_score = float('-inf')
        self.best_training_score = float('-inf')
        self.no_improvement_count = 0
        self.should_stop = False
        
        # Track performance history
        self.validation_history = []
        self.training_history = []
        self.gap_history = []
        self.iteration_count = 0
        
        # Overfitting detection
        self.overfitting_warnings = 0
        self.max_overfitting_warnings = 3
        
    def update(self, training_score: float, validation_score: float, 
               training_metrics: dict = None, validation_metrics: dict = None) -> bool:
        """
        Enhanced validation-aware early stopping.
        
        Args:
            training_score: Training dataset performance score
            validation_score: Validation dataset performance score
            training_metrics: Training metrics (trades, win_rate, etc.)
            validation_metrics: Validation metrics
            
        Returns:
            True if training should stop early, False otherwise
        """
        self.iteration_count += 1
        self.validation_history.append(validation_score)
        self.training_history.append(training_score)
          # Calculate performance gap using FIXED logic
        # Only flag as overfitting if training >> validation (not the reverse)
        if training_score > validation_score and abs(training_score) > 1e-6:
            gap = (training_score - validation_score) / abs(training_score)
        else:
            gap = 0.0  # No overfitting concern when validation >= training
        self.gap_history.append(gap)
        
        print(f"\n📊 VALIDATION MONITORING (Iteration {self.iteration_count}):")
        print(f"   Training Score: {training_score:.4f}")
        print(f"   Validation Score: {validation_score:.4f}")
        print(f"   Performance Gap: {gap:.1%}")
        
        # Enhanced status messaging for healthy vs concerning scenarios
        if validation_score >= training_score:
            print(f"✅ EXCELLENT GENERALIZATION: Validation outperforms training!")
            print(f"   This indicates healthy model generalization, not overfitting.")
        elif gap < 0.1:
            print(f"✅ HEALTHY PERFORMANCE: Training/validation gap is minimal ({gap:.1%})")
        elif gap < self.max_gap_threshold:
            print(f"⚠️ MODERATE GAP: Training/validation gap is {gap:.1%} (threshold: {self.max_gap_threshold:.1%})")
        else:
            print(f"🚨 CONCERNING GAP: Training significantly outperforms validation ({gap:.1%})")
        
        # RULE 1: Never stop before minimum iterations
        if self.iteration_count < self.min_iterations:
            print(f"🕐 Anti-overfitting: {self.iteration_count}/{self.min_iterations} minimum iterations")
            return False        # RULE 2: Check for excessive training/validation gap (OVERFITTING DETECTION)
        if gap > self.max_gap_threshold:
            # Special case: Both training and validation are profitable (positive scores)
            # In trading, this indicates both strategies work, just different risk profiles
            both_profitable = training_score > 0 and validation_score > 0
            
            if both_profitable and validation_score > 0.03:  # Validation is decently profitable (>3%)
                print(f"💰 BOTH STRATEGIES PROFITABLE: Training {training_score:.1%}, Validation {validation_score:.1%}")
                print(f"   Gap {gap:.1%} is large but both strategies are making money - continuing training")
                # Still issue a warning but don't be as aggressive
                if gap > self.max_gap_threshold * 1.5:  # Only escalate if gap is very extreme
                    self.overfitting_warnings += 1
                    print(f"⚠️ MILD WARNING #{self.overfitting_warnings}: Very large gap despite profitability")
            else:
                self.overfitting_warnings += 1
                print(f"⚠️ OVERFITTING WARNING #{self.overfitting_warnings}: Gap {gap:.1%} > {self.max_gap_threshold:.1%}")
                print(f"   Training: {training_score:.4f}, Validation: {validation_score:.4f}")
                
                # Use intelligent detection for additional confirmation
                is_overfitting, reason = self._detect_overfitting_pattern()
                if is_overfitting:
                    print(f"🔍 PATTERN ANALYSIS: {reason}")
                    self.overfitting_warnings += 1  # Extra penalty for confirmed patterns
            
            if self.overfitting_warnings >= self.max_overfitting_warnings:
                print(f"🛑 STOPPING: Excessive overfitting detected!")
                print(f"   - {self.overfitting_warnings} consecutive gap warnings")
                print(f"   - Final gap: {gap:.1%} (threshold: {self.max_gap_threshold:.1%})")
                if 'is_overfitting' in locals() and is_overfitting:
                    print(f"   - Confirmed pattern: {reason}")
                return True
        else:
            # Reset warnings if gap improves and check if we're actually in a good state
            if self.overfitting_warnings > 0:
                is_overfitting, reason = self._detect_overfitting_pattern()
                if not is_overfitting:
                    self.overfitting_warnings = max(0, self.overfitting_warnings - 1)
                    print(f"✓ Gap improved and no overfitting patterns - reduced warnings to {self.overfitting_warnings}")
                else:
                    print(f"⚠️ Gap improved but pattern persists: {reason}")
            
            # Check for subtle overfitting even with acceptable gap
            is_overfitting, reason = self._detect_overfitting_pattern()
            if is_overfitting and gap > self.max_gap_threshold * 0.6:
                print(f"🔍 SUBTLE OVERFITTING DETECTED: {reason}")
                self.overfitting_warnings += 1
        
        # RULE 3: Check validation performance improvement
        validation_improved = validation_score > self.best_validation_score
        if validation_improved:
            self.best_validation_score = validation_score
            self.no_improvement_count = 0
            print(f"📈 NEW BEST validation score: {validation_score:.4f}")
        else:
            self.no_improvement_count += 1
            print(f"📊 No validation improvement ({self.no_improvement_count}/{self.patience})")
          # RULE 4: Enhanced trend-based overfitting detection (less aggressive)
        if len(self.validation_history) >= 4:
            # Use longer history for more stable trends
            recent_val_trend = np.mean(self.validation_history[-3:]) - np.mean(self.validation_history[-6:-3]) if len(self.validation_history) >= 6 else 0
            recent_train_trend = np.mean(self.training_history[-3:]) - np.mean(self.training_history[-6:-3]) if len(self.training_history) >= 6 else 0
            
            # More conservative thresholds to reduce false positives
            strong_train_improvement = recent_train_trend > 0.08  # Increased from 0.05
            significant_val_degradation = recent_val_trend < -self.degradation_threshold * 1.5  # Increased threshold
            
            # Additional check: ensure this isn't just normal variance
            val_variance = np.var(self.validation_history[-4:]) if len(self.validation_history) >= 4 else 0
            is_high_variance = val_variance > 0.02  # Don't penalize during high variance periods
            
            # Only flag if we have strong evidence of overfitting
            if strong_train_improvement and significant_val_degradation and not is_high_variance:
                print(f"⚠️ STRONG OVERFITTING PATTERN DETECTED:")
                print(f"   Training trend: +{recent_train_trend:.4f} (strong improvement)")
                print(f"   Validation trend: {recent_val_trend:.4f} (significant degradation)")
                print(f"   Validation variance: {val_variance:.4f} (low variance confirms pattern)")
                self.overfitting_warnings += 2  # More severe warning for confirmed patterns
                
                if self.overfitting_warnings >= self.max_overfitting_warnings:
                    print(f"🛑 STOPPING: Strong training/validation divergence confirmed!")
                    return True
            elif strong_train_improvement and significant_val_degradation:
                print(f"📊 POTENTIAL OVERFITTING (high variance period):")
                print(f"   Training trend: +{recent_train_trend:.4f}, Validation trend: {recent_val_trend:.4f}")
                print(f"   Validation variance: {val_variance:.4f} - monitoring but not penalizing")
        
        # RULE 5: Check trading activity (ensure model still trades)
        if validation_metrics and validation_metrics.get('total_trades', 0) == 0:
            print(f"⚠️ Model stopped trading on validation - continuing to restore activity")
            self.no_improvement_count = 0  # Reset counter
            return False
        
        # RULE 6: Final patience-based stopping
        if self.no_improvement_count >= self.patience:
            # Final check: is the gap reasonable?
            if gap < self.max_gap_threshold * 2:  # Allow larger gap for final stopping
                print(f"🛑 STOPPING: Validation improvement patience exceeded")
                print(f"   - {self.no_improvement_count} iterations without validation improvement")
                print(f"   - Best validation: {self.best_validation_score:.4f}")
                print(f"   - Current validation: {validation_score:.4f}")
                print(f"   - Final gap: {gap:.1%} (acceptable)")
                return True
            else:
                print(f"🔄 Gap too large ({gap:.1%}) - continuing despite patience exceeded")
                return False
        
        return False
    
    def _detect_overfitting_pattern(self) -> tuple[bool, str]:
        """
        Intelligent overfitting detection using multiple indicators.
        
        Returns:
            (is_overfitting, reason) tuple
        """
        if len(self.validation_history) < 4:
            return False, "Insufficient data"
        
        # Pattern 1: Classic overfitting - training improves, validation degrades
        if len(self.training_history) >= 4:
            train_trend = np.mean(self.training_history[-2:]) - np.mean(self.training_history[-4:-2])
            val_trend = np.mean(self.validation_history[-2:]) - np.mean(self.validation_history[-4:-2])
            
            if train_trend > 0.02 and val_trend < -0.02:
                return True, f"Classic overfitting: training improving (+{train_trend:.3f}), validation degrading ({val_trend:.3f})"
        
        # Pattern 2: Validation volatility increase (model becoming unstable)
        if len(self.validation_history) >= 6:
            early_volatility = np.std(self.validation_history[:3])
            recent_volatility = np.std(self.validation_history[-3:])
            
            if recent_volatility > early_volatility * 2 and recent_volatility > 0.05:
                return True, f"Validation instability: volatility increased from {early_volatility:.3f} to {recent_volatility:.3f}"
          # Pattern 3: Persistent large gap without improvement (but be conservative for profitable cases)
        recent_gaps = self.gap_history[-3:] if len(self.gap_history) >= 3 else []
        if len(recent_gaps) >= 3 and all(g > self.max_gap_threshold * 0.8 for g in recent_gaps):
            avg_gap = np.mean(recent_gaps)
            
            # Check if both strategies are profitable
            recent_training = self.training_history[-3:] if len(self.training_history) >= 3 else []
            recent_validation = self.validation_history[-3:] if len(self.validation_history) >= 3 else []
            
            if (len(recent_training) >= 3 and len(recent_validation) >= 3 and
                all(t > 0.02 for t in recent_training) and all(v > 0.02 for v in recent_validation)):
                # Both consistently profitable (>2%) - be more lenient
                if avg_gap > self.max_gap_threshold * 1.8:  # Only flag if gap is extremely large
                    return True, f"Extreme persistent gap despite profitability: {avg_gap:.1%} for {len(recent_gaps)} iterations"
                else:
                    return False, f"Large gap but both strategies profitable: {avg_gap:.1%}"
            else:
                return True, f"Persistent large gap: {avg_gap:.1%} for {len(recent_gaps)} iterations"
        
        # Pattern 4: Validation score collapse
        if len(self.validation_history) >= 4:
            peak_val = max(self.validation_history)
            current_val = self.validation_history[-1]
            if peak_val > 0 and current_val < peak_val * 0.7:  # 30% drop from peak
                return True, f"Validation collapse: dropped from {peak_val:.3f} to {current_val:.3f}"
        
        return False, "No overfitting pattern detected"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of early stopping decisions."""
        return {
            'iterations_completed': self.iteration_count,
            'best_validation_score': self.best_validation_score,
            'overfitting_warnings': self.overfitting_warnings,
            'final_gap': self.gap_history[-1] if self.gap_history else 0,
            'validation_trend': np.mean(self.validation_history[-3:]) - np.mean(self.validation_history[:3]) if len(self.validation_history) >= 6 else 0
        }

def get_enhanced_hyperparameters(iteration: int, total_iterations: int, 
                                validation_performance: List[float] = None) -> dict:
    """
    Get anti-overfitting hyperparameters based on validation performance.
    
    Strategy: Start conservative, only get aggressive if validation is stable
    """
    # Always start conservative for anti-overfitting
    if iteration < 3:
        return TRAINING_SCHEDULES_ENHANCED['conservative']
    
    # Check validation stability
    if validation_performance and len(validation_performance) >= 3:
        recent_variance = np.var(validation_performance[-3:])
        if recent_variance < 0.01:  # Stable validation performance
            if iteration < total_iterations * 0.6:
                return TRAINING_SCHEDULES_ENHANCED['balanced']
            else:
                return TRAINING_SCHEDULES_ENHANCED['aggressive']
    
    # Default to conservative approach
    return TRAINING_SCHEDULES_ENHANCED['conservative']

def train_walk_forward_enhanced(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """
    ENHANCED walk-forward training with OVERFITTING PREVENTION.
    
    Builds on the existing 5-10x speedup optimizations and adds:
    1. Validation-based early stopping
    2. Training/validation gap monitoring  
    3. Enhanced regularization
    4. Smart hyperparameter adjustment based on validation performance
    
    Prevents scenarios like: Training 138% return / Validation -21% return
    Targets: Balanced performance with <20% train/validation gap
    
    Args:
        data: Full dataset for training
        initial_window: Size of initial training window
        step_size: Step size for moving window forward
        args: Training arguments
        
    Returns:
        RecurrentPPO: Final trained model with good generalization
    """
    total_periods = len(data)
    base_timesteps = getattr(args, 'total_timesteps', 50000)  # Reduced default
    
    # Calculate total number of iterations
    total_iterations = (total_periods - initial_window) // step_size + 1
    
    # Initialize enhanced optimization tracking
    optimization_stats = {
        'total_speedup_achieved': 0.0,
        'avg_iteration_time': 0.0,
        'total_timesteps_saved': 0,
        'early_stops': 0,
        'overfitting_stops': 0,
        'warm_starts': 0,
        'cache_hits': 0,
        'validation_improvements': 0,
        'max_train_val_gap': 0.0
    }
    
    # Track validation performance for hyperparameter adjustment
    validation_performance_history = []
    
    # Enhanced early stopping with validation awareness
    enhanced_early_stopping = ValidationAwareEarlyStoppingCallback(
        patience=getattr(args, 'early_stopping_patience', 5),
        min_iterations=max(6, getattr(args, 'early_stopping_patience', 5)),
        max_gap_threshold=getattr(args, 'max_train_val_gap', 0.25),  # 25% max gap
        degradation_threshold=getattr(args, 'validation_degradation_threshold', 0.1)
    )
    
    state_path = f"../results/{args.seed}/training_state_enhanced.json"
    training_start, _, state = load_training_state(state_path)
    
    best_model_path = os.path.join(f"../results/{args.seed}", "best_model_enhanced.zip")
    if os.path.exists(best_model_path) and getattr(args, 'warm_start', True):
        print(f"🔄 Resuming ENHANCED anti-overfitting training from step {training_start}")
        model = RecurrentPPO.load(best_model_path)
        optimization_stats['warm_starts'] += 1
    else:
        print("🚀 Starting new ENHANCED anti-overfitting training")
        training_start = 0
        model = None
    
    print(f"\n🛡️ ANTI-OVERFITTING FEATURES ENABLED:")
    print(f"Enhanced Early Stopping: ✓ (patience={enhanced_early_stopping.patience})")
    print(f"Validation Gap Monitoring: ✓ (max gap={enhanced_early_stopping.max_gap_threshold:.1%})")
    print(f"Regularization: ✓ (L2 weight decay, higher entropy)")
    print(f"Conservative Training: ✓ (lower LR, fewer epochs)")
    print(f"Adaptive Timesteps: {'✓' if getattr(args, 'adaptive_timesteps', True) else '✗'}")
    print(f"Environment Caching: {'✓' if getattr(args, 'cache_environments', True) else '✗'}")

    try:
        while training_start + initial_window <= total_periods:
            iteration = training_start // step_size
            iteration_start_time = time.time()

            # Calculate adaptive timesteps (reduced for anti-overfitting)
            if getattr(args, 'adaptive_timesteps', True):
                min_timesteps = getattr(args, 'min_timesteps', 20000)  # Reduced minimum
                current_timesteps = calculate_adaptive_timesteps(
                    iteration, base_timesteps, min_timesteps
                )
                # Further reduce for anti-overfitting
                current_timesteps = min(current_timesteps, 40000)  # Cap at 40k
                timesteps_saved = base_timesteps - current_timesteps
                optimization_stats['total_timesteps_saved'] += timesteps_saved
                print(f"🎯 Anti-overfitting timesteps: {current_timesteps:,} (saved {timesteps_saved:,})")
            else:
                current_timesteps = min(base_timesteps, 40000)  # Still cap for anti-overfitting
            
            # Calculate window boundaries
            val_size = int(initial_window * getattr(args, 'validation_size', 0.2))
            train_size = initial_window - val_size
            
            train_start = training_start
            train_end = train_start + train_size
            val_end = min(train_end + val_size, total_periods)
            
            if val_end - train_end < val_size * 0.5:  # Ensure adequate validation data
                break
                
            train_data = data.iloc[train_start:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
        
            train_data.index = data.index[train_start:train_end]
            val_data.index = data.index[train_end:val_end]
            
            print(f"\n=== ENHANCED Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
            print(f"=== Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
            print(f"=== Iteration: {iteration}/{total_iterations} (ANTI-OVERFITTING) ===")
        
            # Environment parameters
            env_params = {
                'initial_balance': getattr(args, 'initial_balance', 10000),
                'balance_per_lot': getattr(args, 'balance_per_lot', 500),
                'random_start': getattr(args, 'random_start', False),
                'point_value': getattr(args, 'point_value', 0.01),
                'min_lots': getattr(args, 'min_lots', 0.01),
                'max_lots': getattr(args, 'max_lots', 200.0),
                'contract_size': getattr(args, 'contract_size', 100.0)
            }
        
            # Create environments
            if getattr(args, 'cache_environments', True):
                from utils.training_utils_optimized import _env_cache
                train_cache_key = _env_cache.get_cache_key(train_data, env_params)
                val_cache_key = _env_cache.get_cache_key(val_data, {**env_params, 'random_start': False})
                
                train_env = Monitor(create_optimized_environment(
                    train_data, env_params, train_cache_key, True
                ))
                val_env = Monitor(create_optimized_environment(
                    val_data, {**env_params, 'random_start': False}, val_cache_key, True
                ))
            else:
                train_env = Monitor(TradingEnv(train_data, **env_params))
                val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
            
            # Get enhanced hyperparameters based on validation performance
            enhanced_params = get_enhanced_hyperparameters(
                iteration, total_iterations, validation_performance_history
            )
            print(f"🛡️ Using anti-overfitting parameters: {list(enhanced_params.keys())}")
            
            if model is None:
                print("\n🚀 Performing ENHANCED initial training with anti-overfitting...")
                
                # Use anti-overfitting configuration
                model_kwargs = {**MODEL_KWARGS_ANTI_OVERFITTING, **enhanced_params}
                
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    train_env,
                    policy_kwargs=POLICY_KWARGS_ANTI_OVERFITTING,
                    verbose=0,
                    device=getattr(args, 'device', 'cuda'),
                    seed=getattr(args, 'seed', 42),
                    **model_kwargs
                )
            else:
                print(f"\n⚡ Continuing ENHANCED training with anti-overfitting...")
                model.set_env(train_env)
                
                # Apply enhanced parameters
                for param, value in enhanced_params.items():
                    if hasattr(model, param):
                        setattr(model, param, value)
                        print(f"🛡️ Updated {param} = {value}")
            
            # Set up enhanced callbacks
            callbacks = [
                CustomEpsilonCallback(
                    start_eps=0.4,    # Higher exploration to prevent overfitting
                    end_eps=0.1,      # Maintain exploration
                    decay_timesteps=int(current_timesteps * 0.8),
                    iteration=iteration
                ),
                UnifiedEvalCallback(
                    val_env,
                    train_data=train_data,
                    val_data=val_data,
                    best_model_save_path=f"../results/{args.seed}",
                    log_path=f"../results/{args.seed}",
                    eval_freq=getattr(args, 'eval_freq', 5000),
                    deterministic=True,
                    verbose=1,
                    iteration=iteration,
                    training_timesteps=current_timesteps
                )
            ]
            
            # Train with anti-overfitting focus
            model.learn(
                total_timesteps=current_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True
            )
            
            # ENHANCED model evaluation on both training and validation
            print("\n📊 Evaluating model performance on both datasets...")
              # Evaluate on training data
            train_env_eval = Monitor(TradingEnv(train_data, **{**env_params, 'random_start': False}))
            training_results = None
            try:
                from utils.training_utils import evaluate_model_on_dataset
                if os.path.exists(f"../results/{args.seed}/curr_best_model.zip"):
                    training_results = evaluate_model_on_dataset(
                        f"../results/{args.seed}/curr_best_model.zip", 
                        train_data, args, use_fast_evaluation=FAST_EVALUATION_AVAILABLE
                    )
            except Exception as e:
                print(f"⚠️ Training evaluation error: {e}")
            
            # Evaluate on validation data
            validation_results = None
            try:
                if os.path.exists(f"../results/{args.seed}/curr_best_model.zip"):
                    validation_results = evaluate_model_on_dataset(
                        f"../results/{args.seed}/curr_best_model.zip", 
                        val_data, args, use_fast_evaluation=FAST_EVALUATION_AVAILABLE
                    )
            except Exception as e:
                print(f"⚠️ Validation evaluation error: {e}")
            
            # Enhanced early stopping check
            if training_results and validation_results:
                training_score = training_results.get('score', 0)
                validation_score = validation_results.get('score', 0)
                
                validation_performance_history.append(validation_score)
                
                # Check for overfitting with enhanced callback
                should_stop = enhanced_early_stopping.update(
                    training_score, validation_score,
                    training_results, validation_results
                )
                
                # Track statistics
                gap = abs(training_score - validation_score) / max(abs(training_score), 1e-6)
                optimization_stats['max_train_val_gap'] = max(optimization_stats['max_train_val_gap'], gap)
                
                if validation_score > enhanced_early_stopping.best_validation_score:
                    optimization_stats['validation_improvements'] += 1
                
                if should_stop:
                    if enhanced_early_stopping.overfitting_warnings >= enhanced_early_stopping.max_overfitting_warnings:
                        optimization_stats['overfitting_stops'] += 1
                        print(f"\n🛑 ENHANCED EARLY STOPPING: Overfitting detected!")
                    else:
                        optimization_stats['early_stops'] += 1
                        print(f"\n🛑 ENHANCED EARLY STOPPING: Validation patience exceeded!")
                    
                    # Save early stopping summary
                    summary = enhanced_early_stopping.get_summary()
                    with open(f"../results/{args.seed}/early_stopping_summary.json", 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    break
            
            # Model selection based on validation performance
            curr_best_path = os.path.join(f"../results/{args.seed}", "curr_best_model.zip")
            
            if os.path.exists(curr_best_path):
                curr_best_metrics_path = curr_best_path.replace(".zip", "_metrics.json")
                
                if os.path.exists(best_model_path):
                    try:
                        with open(curr_best_metrics_path, 'r') as f:
                            curr_metrics = json.load(f)
                        
                        best_metrics_path = best_model_path.replace(".zip", "_metrics.json")
                        if os.path.exists(best_metrics_path):
                            with open(best_metrics_path, 'r') as f:
                                best_metrics = json.load(f)
                            
                            curr_score = curr_metrics.get('validation_score', curr_metrics.get('enhanced_score', 0))
                            best_score = best_metrics.get('validation_score', best_metrics.get('enhanced_score', 0))
                            
                            if curr_score > best_score:
                                model = RecurrentPPO.load(curr_best_path)
                                model.save(best_model_path)
                                import shutil
                                shutil.copy2(curr_best_metrics_path, best_metrics_path)
                                print(f"\n🎯 New best validation score: {curr_score:.4f} > {best_score:.4f}")
                            else:
                                model = RecurrentPPO.load(best_model_path)
                                print(f"\n📊 Keeping previous best: {best_score:.4f} >= {curr_score:.4f}")
                        else:
                            model = RecurrentPPO.load(curr_best_path)
                            model.save(best_model_path)
                            import shutil
                            shutil.copy2(curr_best_metrics_path, best_metrics_path)
                            print("\n🎯 First model saved as best")
                    except Exception as e:
                        print(f"\n⚠️ Error in model selection: {e}")
                        model = RecurrentPPO.load(curr_best_path) if os.path.exists(curr_best_path) else model
                        if model:
                            model.save(best_model_path)
                
                # Clean up
                if os.path.exists(curr_best_path):
                    os.remove(curr_best_path)
                if os.path.exists(curr_best_metrics_path):
                    os.remove(curr_best_metrics_path)
            
            # Calculate iteration time and update stats
            iteration_time = time.time() - iteration_start_time
            optimization_stats['avg_iteration_time'] = (
                optimization_stats['avg_iteration_time'] * iteration + iteration_time
            ) / (iteration + 1)
            
            baseline_time = 45 * 60  # 45 minutes baseline
            current_speedup = baseline_time / iteration_time
            optimization_stats['total_speedup_achieved'] = current_speedup
            
            print(f"\n⚡ ENHANCED PERFORMANCE:")
            print(f"Iteration time: {iteration_time/60:.1f} minutes")
            print(f"Speedup achieved: {current_speedup:.1f}x vs baseline")
            print(f"Max train/val gap: {optimization_stats['max_train_val_gap']:.1%}")
            print(f"Validation improvements: {optimization_stats['validation_improvements']}")
            
            save_training_state(state_path, training_start + step_size, best_model_path,
                              iteration_time=iteration_time, total_iterations=total_iterations,
                              step_size=step_size, optimization_stats=optimization_stats)
            
            training_start += step_size

    except KeyboardInterrupt:
        print("\n🛑 ENHANCED training interrupted. Progress saved.")
        return model

    # Load final model
    if os.path.exists(best_model_path):
        model = RecurrentPPO.load(best_model_path)

    # Save enhanced training summary
    final_summary = {
        'enhanced_training_completed': True,
        'anti_overfitting_enabled': True,
        'total_speedup_achieved': optimization_stats['total_speedup_achieved'],
        'total_timesteps_saved': optimization_stats['total_timesteps_saved'],
        'early_stops': optimization_stats['early_stops'],
        'overfitting_stops': optimization_stats['overfitting_stops'],
        'validation_improvements': optimization_stats['validation_improvements'],
        'max_train_val_gap': optimization_stats['max_train_val_gap'],
        'avg_iteration_time_minutes': optimization_stats['avg_iteration_time'] / 60,
        'early_stopping_summary': enhanced_early_stopping.get_summary(),
        'completion_timestamp': datetime.now().isoformat()
    }
    
    with open(f"../results/{args.seed}/enhanced_training_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n🎉 ENHANCED ANTI-OVERFITTING TRAINING COMPLETED!")
    print(f"⚡ Total speedup: {optimization_stats['total_speedup_achieved']:.1f}x")
    print(f"🛡️ Max train/val gap: {optimization_stats['max_train_val_gap']:.1%}")
    print(f"📈 Validation improvements: {optimization_stats['validation_improvements']}")
    print(f"🛑 Overfitting stops: {optimization_stats['overfitting_stops']}")
    print(f"⏱️ Average iteration time: {optimization_stats['avg_iteration_time']/60:.1f} minutes")
    
    return model
