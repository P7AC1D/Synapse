"""Custom epsilon decay callback for training."""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from gym.spaces import Discrete
from typing import Optional
import torch as th
import os
from configs.config import ENHANCED_EPSILON_CONFIG

class CustomEpsilonCallback(BaseCallback):
    """Custom callback for epsilon-greedy exploration during training"""
    
    def __init__(self, start_eps=1.0, end_eps=0.05, decay_timesteps=40000, iteration=0):
        super().__init__()
        
        # Load exploration parameters from enhanced config
        epsilon_config = ENHANCED_EPSILON_CONFIG.copy()  # Create a copy to avoid modifying original
        
        # Apply exploration parameters with overrides if explicitly provided
        self.start_eps = start_eps if start_eps != 1.0 else epsilon_config.get('start_eps', 0.9)
        self.end_eps = end_eps if end_eps != 0.05 else epsilon_config.get('end_eps', 0.2)
        self.decay_timesteps = decay_timesteps
        self.min_exploration_rate = epsilon_config.get('min_exploration_rate', 0.4)
        
        # Print active configuration
        print("✅ Using enhanced exploration parameters:")
        print(f"   - Start epsilon: {self.start_eps}")
        print(f"   - End epsilon: {self.end_eps}")
        print(f"   - Min exploration rate: {self.min_exploration_rate}")
        
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
