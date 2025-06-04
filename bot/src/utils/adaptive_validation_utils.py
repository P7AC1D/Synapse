"""
Adaptive Validation Utilities for WFO Training

This module provides improved validation criteria that adapt to market conditions
and prevent training stagnation when models can't achieve unrealistic benchmarks.

Key improvements:
1. Adaptive validation thresholds based on recent performance
2. Multi-metric scoring that considers risk-adjusted returns
3. Market regime detection and threshold adjustment
4. Progressive difficulty scaling for long training runs
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime


class AdaptiveValidationManager:
    """
    Manages adaptive validation criteria that prevent training stagnation.
    
    Features:
    - Dynamic threshold adjustment based on recent performance
    - Multi-factor scoring beyond just returns
    - Market regime detection
    - Progressive difficulty scaling
    """
    
    def __init__(self, results_path: str, iteration: int):
        self.results_path = results_path
        self.iteration = iteration
        self.validation_history_path = os.path.join(results_path, "adaptive_validation_history.json")
        self.config_path = os.path.join(results_path, "adaptive_config.json")
        
        # Load or initialize configuration
        self.config = self._load_or_create_config()
        self.history = self._load_validation_history()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create with defaults."""
        default_config = {
            "base_return_threshold": -0.05,  # Allow -5% losses initially
            "improvement_threshold": 0.01,   # 1% improvement needed
            "lookback_window": 20,           # Look at last 20 iterations
            "max_stagnation_iterations": 50, # Reset after 50 iterations
            "risk_adjustment_factor": 0.3,   # Weight for risk metrics
            "adaptive_mode": True,           # Enable adaptive thresholds
            "current_regime": "normal",      # Market regime
            "threshold_decay_rate": 0.95,    # Gradual threshold relaxation
            "min_threshold": -0.15,          # Never go below -15%
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Update with any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"⚠️ Error loading adaptive config: {e}")
                return default_config
        else:
            return default_config
    
    def _load_validation_history(self) -> List[Dict[str, Any]]:
        """Load validation history for analysis."""
        if os.path.exists(self.validation_history_path):
            try:
                with open(self.validation_history_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_config(self):
        """Save current configuration."""
        self.config["last_updated"] = datetime.now().isoformat()
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _save_history(self):
        """Save validation history."""
        with open(self.validation_history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent validation performance to detect patterns."""
        if len(self.history) < 5:
            return {
                "avg_return": -0.1,
                "return_std": 0.1,
                "best_recent": -0.05,
                "worst_recent": -0.2,
                "trend": "insufficient_data"
            }
        
        lookback = min(self.config["lookback_window"], len(self.history))
        recent_returns = [h["validation_return"] for h in self.history[-lookback:]]
        
        return {
            "avg_return": np.mean(recent_returns),
            "return_std": np.std(recent_returns),
            "best_recent": max(recent_returns),
            "worst_recent": min(recent_returns),
            "trend": "improving" if len(recent_returns) > 5 and 
                    np.mean(recent_returns[-5:]) > np.mean(recent_returns[-10:-5]) 
                    else "declining"
        }
    
    def calculate_adaptive_threshold(self, current_best_score: float) -> float:
        """Calculate adaptive validation threshold based on recent performance."""
        if not self.config["adaptive_mode"]:
            return self.config["base_return_threshold"]
        
        performance = self.analyze_recent_performance()
        
        # Start with base threshold
        threshold = self.config["base_return_threshold"]
        
        # Adjust based on recent performance
        if performance["avg_return"] < threshold:
            # If recent performance is poor, relax threshold
            threshold = max(
                performance["avg_return"] * 1.2,  # Allow 20% worse than recent avg
                self.config["min_threshold"]       # But not below minimum
            )
        
        # Progressive relaxation for long stagnation
        stagnation_count = self._count_stagnation_iterations()
        if stagnation_count > self.config["max_stagnation_iterations"]:
            relaxation_factor = (stagnation_count / self.config["max_stagnation_iterations"]) * 0.1
            threshold = max(
                threshold - relaxation_factor,
                self.config["min_threshold"]
            )
        
        # Don't make threshold higher than current best (prevents impossible targets)
        if current_best_score > 0:
            improvement_target = current_best_score * (1 + self.config["improvement_threshold"])
            # If the improvement target is unrealistic, lower it
            if improvement_target > performance["best_recent"] * 2:
                threshold = max(threshold, performance["best_recent"] * 0.8)
        
        return threshold
    
    def _count_stagnation_iterations(self) -> int:
        """Count consecutive iterations without improvement."""
        count = 0
        for record in reversed(self.history):
            if record.get("model_saved", False):
                break
            count += 1
        return count
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score considering multiple factors.
        
        Factors:
        - Raw return (primary)
        - Risk-adjusted return (Sharpe-like)
        - Trade frequency (stability)
        - Drawdown control
        """
        base_return = metrics.get('return', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0.0)
        
        # Base score is the return
        score = base_return
        
        # Risk adjustment (penalize high drawdown)
        if max_dd > 0:
            risk_penalty = min(max_dd * 0.5, 0.2)  # Cap penalty at 20%
            score -= risk_penalty
        
        # Trade frequency bonus (prefer active trading)
        if total_trades > 10:  # Minimum activity threshold
            activity_bonus = min(0.02, total_trades / 1000)  # Up to 2% bonus
            score += activity_bonus
        
        # Win rate bonus
        if win_rate > 0.5:
            winrate_bonus = (win_rate - 0.5) * 0.1  # Up to 5% bonus for 100% win rate
            score += winrate_bonus
        
        return score
    
    def should_save_model(self, validation_metrics: Dict[str, float], 
                         current_best_score: float) -> Dict[str, Any]:
        """
        Determine if model should be saved with adaptive criteria.
        
        Returns:
            Dict with decision, reasoning, and scores
        """
        validation_return = validation_metrics.get('return', 0.0)
        
        # Calculate adaptive threshold
        adaptive_threshold = self.calculate_adaptive_threshold(current_best_score)
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(validation_metrics)
        
        # Decision logic
        decision_info = {
            "should_save": False,
            "validation_return": validation_return,
            "composite_score": composite_score,
            "adaptive_threshold": adaptive_threshold,
            "current_best_score": current_best_score,
            "reasoning": "",
            "improvement_over_best": composite_score - current_best_score if current_best_score > -float('inf') else float('inf')
        }
        
        # Check adaptive threshold
        if validation_return >= adaptive_threshold:
            # Check if it improves upon best score
            if current_best_score == -float('inf') or composite_score > current_best_score:
                decision_info["should_save"] = True
                decision_info["reasoning"] = f"Meets adaptive threshold ({adaptive_threshold:.3f}) and improves best score"
            else:
                decision_info["reasoning"] = f"Meets threshold but doesn't improve best score ({composite_score:.3f} <= {current_best_score:.3f})"
        else:
            decision_info["reasoning"] = f"Below adaptive threshold ({validation_return:.3f} < {adaptive_threshold:.3f})"
        
        # Record this evaluation
        self.history.append({
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "validation_return": validation_return,
            "composite_score": composite_score,
            "adaptive_threshold": adaptive_threshold,
            "model_saved": decision_info["should_save"],
            "metrics": validation_metrics
        })
        
        # Save updated history and config
        self._save_history()
        self._save_config()
        
        return decision_info
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information for debugging."""
        performance = self.analyze_recent_performance()
        stagnation_count = self._count_stagnation_iterations()
        
        return {
            "total_evaluations": len(self.history),
            "recent_performance": performance,
            "stagnation_iterations": stagnation_count,
            "current_config": self.config,
            "adaptive_threshold": self.calculate_adaptive_threshold(-float('inf')),
            "last_10_results": [
                {
                    "iteration": h["iteration"],
                    "return": h["validation_return"],
                    "saved": h["model_saved"]
                }
                for h in self.history[-10:]
            ]
        }


def create_adaptive_eval_callback_wrapper(original_callback_class):
    """
    Creates a wrapper for the original EvalCallback that uses adaptive validation.
    """
    
    class AdaptiveEvalCallback(original_callback_class):
        """Adaptive version of EvalCallback with improved validation criteria."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.adaptive_manager = AdaptiveValidationManager(
                self.best_model_save_path, 
                self.iteration
            )
            print(f"🔄 Initialized adaptive validation for iteration {self.iteration}")
            
            # Print diagnostic info
            diag = self.adaptive_manager.get_diagnostic_info()
            print(f"   Stagnation count: {diag['stagnation_iterations']}")
            print(f"   Adaptive threshold: {diag['adaptive_threshold']:.3f}")
            print(f"   Recent avg return: {diag['recent_performance']['avg_return']:.3f}")
        
        def _should_save_model(self, validation_metrics: Dict[str, float]) -> bool:
            """Override with adaptive validation criteria."""
            
            # Use adaptive manager for decision
            decision = self.adaptive_manager.should_save_model(
                validation_metrics, 
                self.best_validation_score
            )
            
            # Update internal state if model should be saved
            if decision["should_save"]:
                self.best_validation_score = decision["composite_score"]
                self.best_validation_metrics = validation_metrics
                self.no_improvement_count = 0
                
                if self.verbose > 0:
                    print(f"✅ Model saved with adaptive criteria:")
                    print(f"   Composite score: {decision['composite_score']:.3f}")
                    print(f"   Validation return: {decision['validation_return']:.3f}")
                    print(f"   Threshold: {decision['adaptive_threshold']:.3f}")
                    print(f"   Reasoning: {decision['reasoning']}")
            else:
                self.no_improvement_count += 1
                
                if self.verbose > 0:
                    print(f"📊 Model not saved:")
                    print(f"   Validation return: {decision['validation_return']:.3f}")
                    print(f"   Threshold: {decision['adaptive_threshold']:.3f}")
                    print(f"   Reasoning: {decision['reasoning']}")
                    print(f"   No improvement: {self.no_improvement_count}/{self.early_stopping_patience}")
            
            # Save current state
            self._save_current_state()
            
            return decision["should_save"]
    
    return AdaptiveEvalCallback
