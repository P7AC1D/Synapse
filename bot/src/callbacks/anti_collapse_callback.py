"""
Anti-Collapse Callback - Prevents policy collapse by maintaining aggressive exploration.

This callback specifically addresses the "policy collapse" problem where DRL trading models
learn profitable strategies but then completely abandon them in favor of HOLD-only behavior.

Key Features:
1. Entropy monitoring and enforcement
2. Trading activity tracking  
3. Dynamic exploration adjustment
4. Emergency intervention when collapse is detected
"""
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from gym.spaces import Discrete
from typing import Optional, Dict, Any
import os
import json

class AntiCollapseCallback(BaseCallback):
    """Prevents policy collapse by maintaining minimum exploration and trading activity."""
    
    def __init__(
        self, 
        min_entropy_threshold: float = -1.0,
        min_trades_per_eval: int = 3,
        collapse_detection_window: int = 3,
        emergency_epsilon: float = 0.8,
        log_path: Optional[str] = None,
        iteration: int = 0,
        verbose: int = 1
    ):
        """
        Initialize the anti-collapse callback.
        
        Args:
            min_entropy_threshold: Minimum entropy threshold (e.g., -1.0)
            min_trades_per_eval: Minimum trades expected per evaluation
            collapse_detection_window: How many consecutive evaluations to check
            emergency_epsilon: Emergency exploration rate when collapse detected
            log_path: Path to save collapse detection logs
            iteration: Current iteration number
            verbose: Verbosity level
        """
        super().__init__(verbose=verbose)
        self.min_entropy_threshold = min_entropy_threshold
        self.min_trades_per_eval = min_trades_per_eval
        self.collapse_detection_window = collapse_detection_window
        self.emergency_epsilon = emergency_epsilon
        self.log_path = log_path
        self.iteration = iteration
        
        # State tracking
        self.entropy_history = []
        self.trading_activity_history = []
        self.collapse_detected = False
        self.interventions_count = 0
        self.original_forward = None
        self.emergency_mode = False
        
        # Load aggressive configuration for emergency intervention
        self.aggressive_config = self._load_aggressive_config()
        
        if self.verbose >= 1:
            print(f"🛡️ Anti-Collapse Callback initialized:")
            print(f"   - Min entropy threshold: {self.min_entropy_threshold}")
            print(f"   - Min trades per eval: {self.min_trades_per_eval}")
            print(f"   - Detection window: {self.collapse_detection_window}")
            print(f"   - Emergency epsilon: {self.emergency_epsilon}")
    
    def _load_aggressive_config(self) -> Dict[str, Any]:
        """Load aggressive exploration configuration for emergency intervention."""
        try:
            # Get path to aggressive config
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            config_path = os.path.join(src_dir, 'configs', 'aggressive_exploration_config.py')
            
            if os.path.exists(config_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("aggressive_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                if hasattr(config_module, 'get_aggressive_exploration_config'):
                    return config_module.get_aggressive_exploration_config()
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Could not load aggressive config: {e}")
        
        # Fallback configuration
        return {
            'epsilon_config': {
                'start_eps': 0.9,
                'end_eps': 0.6,
                'min_exploration_rate': 0.7
            },
            'model_kwargs': {
                'ent_coef': 1.0,
                'learning_rate': 1e-3
            }
        }
    
    def _setup_emergency_exploration(self):
        """Setup emergency exploration mode by hijacking the policy forward pass."""
        if not hasattr(self.model, 'policy') or self.original_forward is not None:
            return
            
        # Backup original forward method
        self.original_forward = self.model.policy.forward
        
        def emergency_forward_with_exploration(*args, **kwargs):
            """Emergency forward pass with forced high exploration."""
            # Get the original action distribution
            dist = self.original_forward(*args, **kwargs)
            
            # Force high exploration in emergency mode
            if self.emergency_mode and np.random.random() < self.emergency_epsilon:
                if isinstance(self.training_env.action_space, Discrete):
                    # Create uniform random distribution
                    random_logits = th.ones_like(dist.distribution.logits) * 0.1  # Small favoring
                    # Add some randomness to break deterministic patterns
                    random_logits += th.randn_like(random_logits) * 0.5
                    dist.distribution.logits = random_logits
                    
            return dist
        
        # Replace the forward method
        self.model.policy.forward = emergency_forward_with_exploration
        
        if self.verbose >= 1:
            print(f"🚨 Emergency exploration mode activated (ε={self.emergency_epsilon})")
    
    def _restore_original_forward(self):
        """Restore the original forward pass method."""
        if self.original_forward is not None and hasattr(self.model, 'policy'):
            self.model.policy.forward = self.original_forward
            self.original_forward = None
            if self.verbose >= 1:
                print("✅ Restored original policy forward pass")
    
    def _check_entropy_collapse(self) -> bool:
        """Check if entropy has collapsed below threshold."""
        if len(self.entropy_history) < self.collapse_detection_window:
            return False
            
        recent_entropy = self.entropy_history[-self.collapse_detection_window:]
        return all(entropy < self.min_entropy_threshold for entropy in recent_entropy)
    
    def _check_trading_collapse(self) -> bool:
        """Check if trading activity has collapsed."""
        if len(self.trading_activity_history) < self.collapse_detection_window:
            return False
            
        recent_activity = self.trading_activity_history[-self.collapse_detection_window:]
        return all(trades < self.min_trades_per_eval for trades in recent_activity)
    
    def _detect_policy_collapse(self) -> bool:
        """Detect if policy collapse has occurred."""
        entropy_collapsed = self._check_entropy_collapse()
        trading_collapsed = self._check_trading_collapse()
        
        # Collapse detected if both entropy and trading activity are low
        return entropy_collapsed and trading_collapsed
    
    def _apply_emergency_intervention(self):
        """Apply emergency intervention to break policy collapse."""
        if self.verbose >= 1:
            print(f"\n🚨 POLICY COLLAPSE DETECTED - APPLYING EMERGENCY INTERVENTION")
            print(f"   - Recent entropy: {self.entropy_history[-3:] if len(self.entropy_history) >= 3 else self.entropy_history}")
            print(f"   - Recent trading: {self.trading_activity_history[-3:] if len(self.trading_activity_history) >= 3 else self.trading_activity_history}")
        
        # 1. Activate emergency exploration mode
        self.emergency_mode = True
        self._setup_emergency_exploration()
        
        # 2. Increase model's entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            original_ent_coef = self.model.ent_coef
            emergency_ent_coef = self.aggressive_config['model_kwargs'].get('ent_coef', 1.0)
            self.model.ent_coef = emergency_ent_coef
            
            if self.verbose >= 1:
                print(f"   - Entropy coef: {original_ent_coef} → {emergency_ent_coef}")
        
        # 3. Increase learning rate for faster adaptation
        if hasattr(self.model, 'learning_rate'):
            original_lr = self.model.learning_rate
            emergency_lr = self.aggressive_config['model_kwargs'].get('learning_rate', 1e-3)
            self.model.learning_rate = emergency_lr
            
            if self.verbose >= 1:
                print(f"   - Learning rate: {original_lr} → {emergency_lr}")
        
        # 4. Log intervention
        self.interventions_count += 1
        self._log_intervention()
        
        if self.verbose >= 1:
            print(f"🛡️ Emergency intervention #{self.interventions_count} applied")
    
    def _log_intervention(self):
        """Log the intervention for analysis."""
        if self.log_path is None:
            return
            
        intervention_data = {
            'timestamp': str(np.datetime64('now')),
            'iteration': self.iteration,
            'intervention_count': self.interventions_count,
            'entropy_history': self.entropy_history[-5:],  # Last 5 values
            'trading_activity_history': self.trading_activity_history[-5:],
            'emergency_epsilon': self.emergency_epsilon,
            'min_entropy_threshold': self.min_entropy_threshold,
            'model_changes': {
                'ent_coef': getattr(self.model, 'ent_coef', None),
                'learning_rate': getattr(self.model, 'learning_rate', None)
            }
        }
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Load existing interventions
        interventions = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    interventions = json.load(f)
            except:
                pass
        
        # Add new intervention
        interventions.append(intervention_data)
        
        # Save updated interventions
        with open(self.log_path, 'w') as f:
            json.dump(interventions, f, indent=2)
    
    def update_training_metrics(self, entropy_loss: float, total_trades: int):
        """
        Update training metrics for collapse detection.
        
        Args:
            entropy_loss: Current entropy loss (should be negative)
            total_trades: Total number of trades in recent evaluation
        """
        # Store entropy (note: entropy_loss is negative, so lower values = less entropy)
        self.entropy_history.append(entropy_loss)
        self.trading_activity_history.append(total_trades)
        
        # Keep only recent history
        max_history = self.collapse_detection_window * 2
        if len(self.entropy_history) > max_history:
            self.entropy_history = self.entropy_history[-max_history:]
        if len(self.trading_activity_history) > max_history:
            self.trading_activity_history = self.trading_activity_history[-max_history:]
        
        # Check for collapse
        if self._detect_policy_collapse() and not self.collapse_detected:
            self.collapse_detected = True
            self._apply_emergency_intervention()
        
        # Log current state if verbose
        if self.verbose >= 2:
            print(f"   📊 Entropy: {entropy_loss:.3f}, Trades: {total_trades}, Collapse: {self.collapse_detected}")
    
    def reset_collapse_detection(self):
        """Reset collapse detection for new iteration."""
        self.collapse_detected = False
        self.emergency_mode = False
        self._restore_original_forward()
        
        if self.verbose >= 1 and self.interventions_count > 0:
            print(f"🔄 Collapse detection reset. Total interventions: {self.interventions_count}")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # The main logic is handled by update_training_metrics() 
        # which should be called from the evaluation callback
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        self._restore_original_forward()
        
        if self.verbose >= 1:
            print(f"\n🛡️ Anti-Collapse Callback Summary:")
            print(f"   - Total interventions: {self.interventions_count}")
            print(f"   - Final entropy: {self.entropy_history[-1] if self.entropy_history else 'N/A'}")
            print(f"   - Final trading activity: {self.trading_activity_history[-1] if self.trading_activity_history else 'N/A'}")
            if self.log_path and os.path.exists(self.log_path):
                print(f"   - Intervention log: {self.log_path}")
