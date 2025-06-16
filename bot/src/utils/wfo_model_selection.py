"""
Walk-Forward Optimization Model Selection Utilities

This module provides improved model selection strategies for WFO that address
the fundamental issue of comparing models validated on different time periods.

Key improvements:
1. Cross-validation on standardized validation sets
2. Ensemble scoring across multiple periods
3. Risk-adjusted performance metrics
4. Market regime-aware selection
"""

import os
import json
import numpy as np
import pandas as pd
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import shutil
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv


class WFOModelSelector:
    """
    Improved model selection for Walk-Forward Optimization.
    
    Strategies available:
    1. Rolling validation on consistent periods
    2. Ensemble scoring across multiple validation windows
    3. Risk-adjusted metrics (Sharpe, Calmar ratios)
    4. Market regime detection and adaptive thresholds
    """
    
    def __init__(self, results_path: str, strategy: str = "rolling_validation"):
        """
        Initialize the WFO Model Selector.
        
        Args:
            results_path: Path to store results and models
            strategy: Selection strategy ('rolling_validation', 'random_sequential_validation', 
                     'risk_adjusted', 'market_regime_adaptive')
        """
        self.results_path = results_path
        self.strategy = strategy
        self.selection_history_path = os.path.join(results_path, "model_selection_history.json")
        self.selection_history = self._load_selection_history()
        
        # Configuration for different strategies
        self.config = {
            'rolling_validation': {
                'validation_window_size': 2000,  # Fixed validation window size
                'num_rolling_windows': 5,  # Number of rolling windows to test
                'consistency_weight': 0.3,  # Weight for performance consistency
                'weights': {'return': 0.4, 'sharpe': 0.3, 'max_dd': 0.3}  # Ensemble scoring weights
            },
            'random_sequential_validation': {
                'validation_window_size': 2000,  # Size of each random sequential block
                'num_validation_periods': 5,  # Number of random blocks to select
                'min_gap_from_current': 2000,  # Don't pick blocks too close to current position
                'min_block_spacing': 1000,  # Minimum gap between selected blocks
                'max_lookback_periods': 50000,  # Maximum lookback (avoid very old data)
                'temporal_weighting': True,  # Weight more recent blocks higher
                'random_seed_base': 42,  # Base seed for reproducibility
                'weights': {'return': 0.4, 'sharpe': 0.3, 'max_dd': 0.3}  # Ensemble scoring weights
            },
            'risk_adjusted': {
                'sharpe_threshold': 0.5,   # Minimum Sharpe ratio
                'max_drawdown_threshold': 0.15,  # Maximum allowable drawdown
                'calmar_weight': 0.4,     # Weight for Calmar ratio
                'return_weight': 0.6      # Weight for raw returns
            },
            'market_regime_adaptive': {
                'volatility_lookback': 30,  # Days to calculate volatility regime
                'trend_lookback': 20,       # Days to calculate trend regime
                'regime_thresholds': {
                    'low_vol': 0.01,       # Daily vol < 1%
                    'high_vol': 0.03,      # Daily vol > 3%
                    'strong_trend': 0.02   # Daily price change > 2%
                }
            }
        }
    
    def _load_selection_history(self) -> List[Dict[str, Any]]:
        """Load model selection history."""
        if os.path.exists(self.selection_history_path):
            with open(self.selection_history_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_selection_history(self):
        """Save model selection history."""
        with open(self.selection_history_path, 'w') as f:
            json.dump(self.selection_history, f, indent=2)
    
    def _create_standardized_validation_sets(self, data: pd.DataFrame, 
                                           current_end_idx: int) -> List[pd.DataFrame]:
        """
        Create standardized validation sets for fair comparison.
        
        Args:
            data: Full dataset
            current_end_idx: Current position in the dataset
            
        Returns:
            List of validation DataFrames
        """
        validation_sets = []
        config = self.config[self.strategy]
        
        if self.strategy == "rolling_validation":
            # Create rolling windows of fixed size
            window_size = config['validation_window_size']
            num_windows = config['num_rolling_windows']
            
            for i in range(num_windows):
                start_idx = max(0, current_end_idx - (i + 1) * window_size)
                end_idx = max(window_size, current_end_idx - i * window_size)
                
                if end_idx - start_idx >= window_size * 0.8:
                    val_set = data.iloc[start_idx:end_idx].copy()
                    validation_sets.append(val_set)
        
        elif self.strategy == "random_sequential_validation":
            # Create random sequential blocks for diverse validation
            validation_sets = self._create_random_sequential_validation_sets(data, current_end_idx, config)
        
        return validation_sets
    
    def _create_random_sequential_validation_sets(self, data: pd.DataFrame, 
                                                current_end_idx: int, 
                                                config: Dict[str, Any]) -> List[pd.DataFrame]:
        """
        Create random sequential validation blocks for diverse market regime testing.
        
        Args:
            data: Full dataset
            current_end_idx: Current position in dataset
            config: Configuration for random sequential validation
            
        Returns:
            List of random sequential validation DataFrames
        """
        validation_sets = []
        
        window_size = config['validation_window_size']
        num_periods = config['num_validation_periods']
        min_gap = config['min_gap_from_current']
        min_spacing = config['min_block_spacing']
        max_lookback = config['max_lookback_periods']
        
        # Define available range for random selection
        # Don't go too far back (avoid very old data) or too close to current
        earliest_start = max(0, current_end_idx - max_lookback)
        latest_start = max(0, current_end_idx - min_gap - window_size)
        
        if latest_start <= earliest_start:
            print(f"   ⚠️ Insufficient data range for random sequential validation")
            return validation_sets
        
        # Set reproducible seed based on iteration and base seed
        # This ensures same blocks are selected for both current and best model comparison
        seed = config['random_seed_base'] + hash(str(current_end_idx)) % 1000
        random.seed(seed)
        
        # Generate random start positions with minimum spacing
        available_positions = list(range(earliest_start, latest_start))
        selected_starts = []
        
        for _ in range(num_periods):
            if not available_positions:
                break
                
            # Select random position
            start_pos = random.choice(available_positions)
            selected_starts.append(start_pos)
            
            # Remove positions too close to selected one (enforce spacing)
            available_positions = [
                pos for pos in available_positions 
                if abs(pos - start_pos) >= min_spacing
            ]
        
        # Sort starts for easier debugging and logging
        selected_starts.sort()
        
        # Create validation sets from selected positions
        temporal_positions = []
        for i, start_idx in enumerate(selected_starts):
            end_idx = min(start_idx + window_size, len(data))
            
            if end_idx - start_idx >= window_size * 0.8:  # At least 80% of desired size
                val_set = data.iloc[start_idx:end_idx].copy()
                validation_sets.append(val_set)
                
                # Calculate temporal position for logging (how far back from current)
                days_back = (current_end_idx - start_idx) // 96  # Assuming 96 periods per day
                temporal_positions.append(days_back)
                
                print(f"   🎲 Random block {i+1}: samples {start_idx}-{end_idx} "
                      f"({len(val_set)} samples, ~{days_back} days back)")
        
        if config.get('temporal_weighting', False) and validation_sets:
            # Store temporal positions for potential weighting in scoring
            # More recent blocks could get higher weights
            self._temporal_positions = temporal_positions
        
        print(f"   📊 Created {len(validation_sets)} random sequential validation blocks")
        print(f"   🌍 Market regime diversity: ~{max(temporal_positions) - min(temporal_positions)} days span")
        
        return validation_sets
    
    def _evaluate_model_on_validation_set(self, model: RecurrentPPO,
                                        validation_data: pd.DataFrame,
                                        env_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model on a single validation set.
        
        Args:
            model: Model to evaluate
            validation_data: Validation dataset
            env_params: Environment parameters
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Create validation environment
            val_env = Monitor(TradingEnv(validation_data, **{**env_params, 'random_start': False}))
            
            # Run evaluation
            obs, _ = val_env.reset()
            lstm_states = None
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done and step_count < len(validation_data):
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
                obs, reward, terminated, truncated, info = val_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
            
            # Extract metrics from environment
            if hasattr(val_env, 'env'):
                env_metrics = val_env.env.metrics.get_performance_summary()
            else:
                env_metrics = val_env.metrics.get_performance_summary()
            
            # Normalize metrics
            metrics = {
                'return': env_metrics.get('return_pct', 0.0) / 100.0,
                'sharpe_ratio': env_metrics.get('sharpe_ratio', 0.0),
                'max_drawdown': env_metrics.get('max_drawdown_pct', 0.0) / 100.0,
                'profit_factor': env_metrics.get('profit_factor', 0.0),
                'win_rate': env_metrics.get('win_rate', 0.0) / 100.0,
                'total_trades': env_metrics.get('total_trades', 0),
                'episode_reward': episode_reward,
                'steps': step_count
            }
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Validation evaluation error: {e}")
            return {
                'return': -1.0,
                'sharpe_ratio': -1.0,
                'max_drawdown': 1.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'episode_reward': -1000,
                'steps': 0
            }
    
    def _calculate_ensemble_score(self, validation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate ensemble score across multiple validation periods.
        
        Args:
            validation_results: List of validation results from different periods
            
        Returns:
            Dictionary with ensemble score and detailed metrics
        """
        if not validation_results:
            return {'score': -float('inf'), 'details': {}}
        
        # Get weights from appropriate config (all strategies that use ensemble scoring have weights)
        if self.strategy in ['rolling_validation', 'random_sequential_validation']:
            config = self.config[self.strategy]
            weights = config['weights']
        else:
            # Fallback weights
            weights = {'return': 0.4, 'sharpe': 0.3, 'max_dd': 0.3}
        
        # Calculate average metrics
        avg_metrics = {}
        for key in validation_results[0].keys():
            values = [result[key] for result in validation_results if result[key] is not None]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
                avg_metrics[f'{key}_consistency'] = 1.0 - (np.std(values) / (np.mean(np.abs(values)) + 1e-8))
            else:
                avg_metrics[key] = 0.0
                avg_metrics[f'{key}_std'] = 0.0
                avg_metrics[f'{key}_consistency'] = 0.0
        
        # Calculate ensemble score
        return_score = avg_metrics.get('return', 0.0)
        sharpe_score = max(0, avg_metrics.get('sharpe_ratio', 0.0)) / 3.0  # Normalize Sharpe
        drawdown_penalty = max(0, avg_metrics.get('max_drawdown', 0.0))
        
        ensemble_score = (
            weights['return'] * return_score + 
            weights['sharpe'] * sharpe_score - 
            weights['max_dd'] * drawdown_penalty
        )
        
        # Apply consistency bonus
        consistency_bonus = (
            avg_metrics.get('return_consistency', 0.0) + 
            avg_metrics.get('sharpe_ratio_consistency', 0.0)
        ) / 2.0 * 0.1  # Up to 10% bonus for consistency
        
        final_score = ensemble_score + consistency_bonus
        
        return {
            'score': final_score,
            'ensemble_score': ensemble_score,
            'consistency_bonus': consistency_bonus,
            'avg_metrics': avg_metrics,
            'individual_results': validation_results,
            'details': {
                'return_score': return_score,
                'sharpe_score': sharpe_score,
                'drawdown_penalty': drawdown_penalty,
                'num_validation_periods': len(validation_results)
            }
        }
    
    def _calculate_risk_adjusted_score(self, validation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate risk-adjusted score focusing on risk metrics."""
        if not validation_results:
            return {'score': -float('inf'), 'details': {}}
        
        config = self.config['risk_adjusted']
        
        # Average across validation periods
        avg_return = np.mean([r['return'] for r in validation_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in validation_results])
        avg_max_dd = np.mean([r['max_drawdown'] for r in validation_results])
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = avg_return / max(avg_max_dd, 0.01)  # Avoid division by zero
        
        # Risk-adjusted score
        risk_score = (
            config['return_weight'] * avg_return +
            config['calmar_weight'] * min(calmar_ratio, 5.0) / 5.0  # Normalize Calmar
        )
        
        # Apply penalties for excessive risk
        if avg_sharpe < config['sharpe_threshold']:
            risk_score *= 0.5  # 50% penalty for low Sharpe
        
        if avg_max_dd > config['max_drawdown_threshold']:
            risk_score *= 0.3  # 70% penalty for high drawdown
        
        return {
            'score': risk_score,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_max_dd': avg_max_dd,
            'calmar_ratio': calmar_ratio,
            'details': {
                'sharpe_penalty': avg_sharpe < config['sharpe_threshold'],
                'drawdown_penalty': avg_max_dd > config['max_drawdown_threshold']
            }
        }
    
    def select_best_model(self, current_model_path: str, best_model_path: str,
                         data: pd.DataFrame, current_end_idx: int,
                         env_params: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Select the best model using the configured strategy.
        
        Args:
            current_model_path: Path to current iteration's model
            best_model_path: Path to current best model
            data: Full dataset
            current_end_idx: Current position in dataset
            env_params: Environment parameters
            iteration: Current iteration number
            
        Returns:
            Dictionary with selection decision and details
        """
        if not os.path.exists(current_model_path):
            return {
                'decision': 'no_current_model',
                'reasoning': 'Current model file not found',
                'score_current': None,
                'score_best': None
            }
        
        print(f"\n💾 Advanced Model Selection ({self.strategy})...")
        
        # Load current model
        current_model = RecurrentPPO.load(current_model_path)
        
        # Create standardized validation sets
        validation_sets = self._create_standardized_validation_sets(data, current_end_idx)
        
        if not validation_sets:
            return {
                'decision': 'insufficient_validation_data',
                'reasoning': 'Not enough data for standardized validation',
                'score_current': None,
                'score_best': None
            }
        
        print(f"   📊 Evaluating on {len(validation_sets)} standardized validation periods")
        
        # Evaluate current model on all validation sets
        current_results = []
        for i, val_set in enumerate(validation_sets):
            print(f"   🧪 Validation period {i+1}: {len(val_set)} samples")
            result = self._evaluate_model_on_validation_set(current_model, val_set, env_params)
            current_results.append(result)
        
        # Calculate current model score based on strategy
        if self.strategy in ['rolling_validation', 'random_sequential_validation']:
            current_score_info = self._calculate_ensemble_score(current_results)
        elif self.strategy == 'risk_adjusted':
            current_score_info = self._calculate_risk_adjusted_score(current_results)
        else:
            # Fallback to simple average
            avg_return = np.mean([r['return'] for r in current_results])
            current_score_info = {'score': avg_return, 'details': {'avg_return': avg_return}}
        
        current_score = current_score_info['score']
        
        # Handle best model comparison
        if not os.path.exists(best_model_path):
            # No previous best model
            decision_info = {
                'decision': 'save_as_first_best',
                'reasoning': f'First model with score: {current_score:.4f}',
                'score_current': current_score,
                'score_best': None,
                'current_details': current_score_info,
                'improvement': float('inf')
            }
        else:
            # Load and evaluate best model
            best_model = RecurrentPPO.load(best_model_path)
            
            best_results = []
            for val_set in validation_sets:
                result = self._evaluate_model_on_validation_set(best_model, val_set, env_params)
                best_results.append(result)
            
            # Calculate best model score
            if self.strategy in ['rolling_validation', 'random_sequential_validation']:
                best_score_info = self._calculate_ensemble_score(best_results)
            elif self.strategy == 'risk_adjusted':
                best_score_info = self._calculate_risk_adjusted_score(best_results)
            else:
                avg_return = np.mean([r['return'] for r in best_results])
                best_score_info = {'score': avg_return, 'details': {'avg_return': avg_return}}
            
            best_score = best_score_info['score']
            
            # Make decision
            improvement = current_score - best_score
            should_update = current_score > best_score
            
            decision_info = {
                'decision': 'update_best' if should_update else 'keep_previous',
                'reasoning': (
                    f'Current: {current_score:.4f} vs Best: {best_score:.4f} '
                    f'({"+" if improvement > 0 else ""}{improvement:.4f})'
                ),
                'score_current': current_score,
                'score_best': best_score,
                'current_details': current_score_info,
                'best_details': best_score_info,
                'improvement': improvement
            }
        
        # Log decision
        self.selection_history.append({
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy,
            'decision': decision_info['decision'],
            'score_current': current_score,
            'score_best': decision_info.get('score_best'),
            'improvement': decision_info.get('improvement'),
            'num_validation_periods': len(validation_sets),
            'validation_period_sizes': [len(vs) for vs in validation_sets]
        })
        
        self._save_selection_history()
        
        return decision_info


def apply_improved_model_selection(results_path: str, current_model_path: str, 
                                 best_model_path: str, data: pd.DataFrame,
                                 current_end_idx: int, env_params: Dict[str, Any],
                                 iteration: int, strategy: str = "rolling_validation") -> bool:
    """
    Apply improved model selection and return whether model was updated.
    
    Args:
        results_path: Path to results directory
        current_model_path: Path to current model
        best_model_path: Path to best model
        data: Full dataset
        current_end_idx: Current position in dataset
        env_params: Environment parameters
        iteration: Current iteration
        strategy: Selection strategy to use
        
    Returns:
        True if best model was updated, False otherwise
    """
    selector = WFOModelSelector(results_path, strategy)
    
    decision = selector.select_best_model(
        current_model_path, best_model_path, data, current_end_idx, env_params, iteration
    )
    
    # Print results
    print(f"   📋 Decision: {decision['decision']}")
    print(f"   💭 Reasoning: {decision['reasoning']}")
    
    if decision['score_current'] is not None:
        print(f"   📊 Current Score: {decision['score_current']:.4f}")
    if decision['score_best'] is not None:
        print(f"   🏆 Best Score: {decision['score_best']:.4f}")
    
    # Execute decision
    should_update = decision['decision'] in ['save_as_first_best', 'update_best']
    
    if should_update:
        # Copy current model as new best
        shutil.copy2(current_model_path, best_model_path)
        
        # Save detailed metrics
        metrics_path = best_model_path.replace('.zip', '_detailed_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'selection_decision': decision,
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'strategy': strategy
            }, f, indent=2)
        
        if decision['decision'] == 'save_as_first_best':
            print(f"   ✅ Saved as first best model")
        else:
            print(f"   🎯 NEW BEST MODEL: {decision['score_current']:.4f} > {decision['score_best']:.4f}")
    else:
        print(f"   📊 Keeping previous best: {decision['score_best']:.4f} >= {decision['score_current']:.4f}")
    
    return should_update
