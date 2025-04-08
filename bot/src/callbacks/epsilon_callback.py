"""Custom epsilon decay callback for training."""
from stable_baselines3.common.callbacks import BaseCallback

class CustomEpsilonCallback(BaseCallback):
    """Custom callback for epsilon decay during training"""
    def __init__(self, start_eps=0.5, end_eps=0.02, decay_timesteps=60000, iteration=0):  # Enhanced exploration
        super().__init__()
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_timesteps = min(decay_timesteps, 60000)  # Extended cap for longer exploration
        self.iteration = iteration  # Track training iteration
        
    def _on_step(self) -> bool:
        # Use slower decay in early iterations
        if self.iteration <= 1:
            progress = min(1.0, (self.num_timesteps / self.decay_timesteps) ** 1.5)  # Slower decay curve
        else:
            progress = min(1.0, self.num_timesteps / self.decay_timesteps)
            
        current_eps = self.start_eps + progress * (self.end_eps - self.start_eps)
        
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'exploration_rate'):
            self.model.policy.exploration_rate = current_eps
            
        return True
