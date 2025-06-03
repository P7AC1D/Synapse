"""Custom epsilon decay callback for training."""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gym.spaces import Discrete
from typing import Optional
import torch as th

class CustomEpsilonCallback(BaseCallback):
    """Custom callback for epsilon-greedy exploration during training"""
    
    def __init__(self, start_eps=1.0, end_eps=0.05, decay_timesteps=40000, iteration=0):
        super().__init__()
        
        # Load exploration parameters using robust import mechanism with aggressive prioritization
        import os
        import sys
        
        # Default enhanced values (fallback)
        epsilon_config = {
            'start_eps': 0.9,
            'end_eps': 0.2,
            'decay_timesteps': 30000,
            'min_exploration_rate': 0.4
        }
        
        # 🚀 PRIORITIZE AGGRESSIVE EXPLORATION FOR EARLY ITERATIONS
        needs_aggressive_exploration = iteration < 5  # First 5 iterations get aggressive exploration
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)  # Go up to src directory
            
            if needs_aggressive_exploration:
                # Try to load aggressive exploration config first
                aggressive_configs_path = os.path.join(src_dir, 'configs', 'aggressive_exploration_config.py')
                if os.path.exists(aggressive_configs_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("aggressive_exploration_config", aggressive_configs_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    if hasattr(config_module, 'AGGRESSIVE_EPSILON_CONFIG'):
                        epsilon_config = config_module.AGGRESSIVE_EPSILON_CONFIG
                        print("🚀 Successfully loaded AGGRESSIVE epsilon configuration (MAXIMUM EXPLORATION)")
                        print(f"   - Start epsilon: {epsilon_config.get('start_eps', 'N/A')}")
                        print(f"   - End epsilon: {epsilon_config.get('end_eps', 'N/A')}")
                        print(f"   - Min exploration: {epsilon_config.get('min_exploration_rate', 'N/A')}")
                    else:
                        print("⚠️ Aggressive epsilon config exists but AGGRESSIVE_EPSILON_CONFIG not found")
                else:
                    print("⚠️ Aggressive epsilon config not found, falling back to enhanced")
            
            # Fallback to enhanced configuration if aggressive not available or not needed
            if not needs_aggressive_exploration or 'AGGRESSIVE_EPSILON_CONFIG' not in locals():
                enhanced_configs_path = os.path.join(src_dir, 'configs', 'enhanced_exploration_config.py')
                if os.path.exists(enhanced_configs_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("enhanced_exploration_config", enhanced_configs_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    if hasattr(config_module, 'ENHANCED_EPSILON_CONFIG'):
                        epsilon_config = config_module.ENHANCED_EPSILON_CONFIG
                        print("✅ Successfully loaded enhanced epsilon configuration")
                    
        except Exception as e:
            print(f"⚠️ Using default epsilon parameters: {e}")
        
        # Apply exploration parameters
        self.start_eps = start_eps if start_eps != 1.0 else epsilon_config.get('start_eps', 0.9)
        self.end_eps = end_eps if end_eps != 0.05 else epsilon_config.get('end_eps', 0.2)
        self.decay_timesteps = decay_timesteps
        self.min_exploration_rate = epsilon_config.get('min_exploration_rate', 0.4)
        
        self.iteration = iteration
        self.original_forward = None
        self.setup_done = False
        
    def _setup_exploration(self) -> None:
        """Setup exploration by modifying the policy's forward pass"""
        if not self.setup_done and hasattr(self.model, 'policy'):
            self.original_forward = self.model.policy.forward
            
            def forward_with_exploration(*args, **kwargs):
                # Get the original action distribution
                dist = self.original_forward(*args, **kwargs)
                
                # Calculate current epsilon with minimum exploration rate
                if self.iteration <= 1:
                    progress = min(1.0, (self.num_timesteps / self.decay_timesteps) ** 1.2)
                    current_eps = max(
                        self.start_eps + progress * (self.end_eps - self.start_eps),
                        self.min_exploration_rate
                    )
                else:
                    progress = min(1.0, self.num_timesteps / self.decay_timesteps)
                    current_eps = self.start_eps + progress * (self.end_eps - self.start_eps)
                
                # Random exploration
                if np.random.random() < current_eps:
                    # Force random action by modifying the distribution
                    if isinstance(self.training_env.action_space, Discrete):
                        random_logits = th.ones_like(dist.distribution.logits)
                        dist.distribution.logits = random_logits
                        
                return dist
                
            self.model.policy.forward = forward_with_exploration
            self.setup_done = True
    
    def _on_step(self) -> bool:
        if not self.setup_done:
            self._setup_exploration()
        return True
    
    def _on_training_end(self) -> None:
        """Restore original forward pass at end of training"""
        if self.original_forward is not None and hasattr(self.model, 'policy'):
            self.model.policy.forward = self.original_forward
