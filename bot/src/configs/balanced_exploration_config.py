"""
Balanced exploration configuration to fix both the "no trades" and "unstable trading" problems.

This configuration provides sufficient exploration to discover profitable strategies while 
maintaining stability to prevent the model from forgetting learned behaviors.
"""
import torch as th

# 🎯 BALANCED EXPLORATION MODEL CONFIGURATION
POLICY_KWARGS_BALANCED_EXPLORATION = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 256,              # Moderate capacity (was 512, too large)
    "n_lstm_layers": 2,                   # Moderate depth (was 4, too deep)
    "shared_lstm": False,                 # Separate LSTM architectures
    "enable_critic_lstm": True,           # Enable LSTM for value estimation
    "net_arch": {
        "pi": [256, 128, 64],             # Balanced policy network
        "vf": [256, 128, 64]              # Balanced value network
    },
    "activation_fn": th.nn.ReLU,          # Stable activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-4              # Regularization
    }
}

# 🎯 BALANCED EXPLORATION TRAINING PARAMETERS
MODEL_KWARGS_BALANCED_EXPLORATION = {
    'learning_rate': 3e-4,               # MODERATE: Balanced learning rate for stability
    'n_steps': 1024,                     # Good sequence length
    'batch_size': 512,                   # Stable batch size
    'n_epochs': 10,                      # Adequate training epochs
    'gamma': 0.995,                      # High gamma for long-term rewards
    'gae_lambda': 0.95,                  # REDUCED: More conservative advantage estimation
    'clip_range': lambda progress: 0.15 + 0.05 * (1 - progress),  # Adaptive clipping (0.20 → 0.15)
    'clip_range_vf': lambda progress: 0.15 + 0.05 * (1 - progress),  # Adaptive value clipping
    'ent_coef': 0.25,                    # BALANCED: Moderate exploration (not too high/low)
    'vf_coef': 0.5,                      # Balanced value learning
    'max_grad_norm': 0.5,                # Conservative gradient clipping
    'verbose': 0
}

# 🎯 BALANCED REWARD SYSTEM PARAMETERS
BALANCED_REWARD_CONFIG = {
    # Core trading rewards - Balanced incentives
    'PROFITABLE_TRADE_REWARD': 6.0,      # Good reward for winning trades
    'LOSING_TRADE_PENALTY': -2.0,        # Moderate penalty to encourage learning
    'MARKET_ENGAGEMENT_BONUS': 1.5,      # Moderate bonus for taking positions
    
    # Position management - Encouraging holding profitable positions
    'PROFIT_HOLD_REWARD': 1.5,           # Reward for holding profitable positions
    'LOSS_HOLD_PENALTY': -0.1,           # Small penalty for holding losing positions
    'PROFIT_PROTECTION_BONUS': 0.8,      # Bonus for protecting unrealized profits
    'SIGNIFICANT_PROFIT_THRESHOLD': 0.005, # 0.5% profit threshold for bonus rewards
    'SIGNIFICANT_PROFIT_BONUS': 1.2,     # Extra bonus for significantly profitable positions
    
    # Activity incentives - Balanced approach
    'HOLD_COST': -0.01,                  # MODERATE: Cost for inaction (not too high)
    'EXCESSIVE_HOLD_PENALTY': -0.03,     # Penalty for excessive holding
    'INACTIVITY_THRESHOLD': 40,          # BALANCED: Moderate threshold
    
    # Risk management
    'INVALID_ACTION_PENALTY': -2.0,      # Penalty for invalid actions
    'OVERTRADING_PENALTY': -0.8,         # Moderate penalty for too frequent trading
    'MIN_POSITION_HOLD': 3,              # Minimum bars to hold position
    
    # Performance bonuses
    'NEW_HIGH_BONUS': 2.5,               # Moderate bonus for new equity highs
    'CONSISTENCY_BONUS': 0.4,            # Bonus for consistent performance
    'RISK_ADJUSTED_MULTIPLIER': 1.8,     # Moderate multiplier for risk-adjusted returns
}

# 🎯 BALANCED EXPLORATION CALLBACK SETTINGS
BALANCED_EPSILON_CONFIG = {
    'start_eps': 0.6,                    # MODERATE: Start with reasonable exploration
    'end_eps': 0.1,                      # MODERATE: End with some exploration maintained
    'decay_timesteps_ratio': 0.8,       # Moderate decay - explore for 80% of training
    'min_exploration_rate': 0.25,       # MODERATE: Reasonable minimum exploration
}

def get_balanced_exploration_config():
    """Get the complete balanced exploration configuration."""
    return {
        'policy_kwargs': POLICY_KWARGS_BALANCED_EXPLORATION,
        'model_kwargs': MODEL_KWARGS_BALANCED_EXPLORATION,
        'reward_config': BALANCED_REWARD_CONFIG,
        'epsilon_config': BALANCED_EPSILON_CONFIG
    }

def apply_balanced_exploration_to_model(model):
    """Apply balanced exploration parameters to an existing model."""
    # Update exploration coefficient
    model.ent_coef = MODEL_KWARGS_BALANCED_EXPLORATION['ent_coef']
    model.learning_rate = MODEL_KWARGS_BALANCED_EXPLORATION['learning_rate']
    
    print(f"✅ Applied balanced exploration:")
    print(f"   - Entropy coefficient: {model.ent_coef}")
    print(f"   - Learning rate: {model.learning_rate}")
    print(f"   - Architecture: Moderate complexity for stability")
    
    return model

if __name__ == "__main__":
    config = get_balanced_exploration_config()
    print("🎯 Balanced Exploration Configuration:")
    print(f"   - Entropy coefficient: {config['model_kwargs']['ent_coef']}")
    print(f"   - Learning rate: {config['model_kwargs']['learning_rate']}")
    print(f"   - Start epsilon: {config['epsilon_config']['start_eps']}")
    print(f"   - Market engagement bonus: {config['reward_config']['MARKET_ENGAGEMENT_BONUS']}")
    print(f"   - Hold cost: {config['reward_config']['HOLD_COST']}")
    print("   - Focus: Balanced exploration + stability")
