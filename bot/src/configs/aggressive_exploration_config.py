"""
AGGRESSIVE EXPLORATION CONFIGURATION - Maximum exploration to force trading behavior.

This configuration uses extreme exploration parameters to break the model out of 
HOLD-only mode and force it to discover trading strategies.
"""
import torch as th

# 🚀 AGGRESSIVE EXPLORATION MODEL CONFIGURATION  
POLICY_KWARGS_AGGRESSIVE_EXPLORATION = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 256,              # REDUCED: Smaller model for faster learning
    "n_lstm_layers": 2,                   # REDUCED: Simpler architecture
    "shared_lstm": False,                 # Separate LSTM architectures
    "enable_critic_lstm": True,           # Enable LSTM for value estimation
    "net_arch": {
        "pi": [256, 128],                 # SIMPLIFIED: Smaller policy network for faster exploration
        "vf": [256, 128]                  # SIMPLIFIED: Smaller value network
    },
    "activation_fn": th.nn.ReLU,          # Standard activation for stability
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-5              # REDUCED: Minimal regularization for maximum exploration
    }
}

# 🚀 AGGRESSIVE EXPLORATION TRAINING PARAMETERS
MODEL_KWARGS_AGGRESSIVE_EXPLORATION = {
    'learning_rate': 1e-3,               # INCREASED: High LR for rapid policy changes
    'n_steps': 512,                      # REDUCED: Shorter sequences for faster updates
    'batch_size': 256,                   # REDUCED: Smaller batches for more frequent updates
    'n_epochs': 15,                      # INCREASED: More epochs for aggressive learning
    'gamma': 0.99,                       # REDUCED: Focus on immediate rewards
    'gae_lambda': 0.90,                  # REDUCED: Less complex advantage estimation
    'clip_range': lambda progress: 0.25 + 0.15 * (1 - progress),  # AGGRESSIVE: High clipping (starts 0.40, ends 0.25)
    'clip_range_vf': lambda progress: 0.25 + 0.15 * (1 - progress),  # AGGRESSIVE: High value clipping
    'ent_coef': 1.0,                     # MAXIMUM: Extremely high exploration (10x normal)
    'vf_coef': 0.25,                     # REDUCED: Less emphasis on value function
    'max_grad_norm': 1.0,                # INCREASED: Allow larger gradient steps
    'verbose': 0
}

# 🎯 AGGRESSIVE REWARD SYSTEM PARAMETERS  
AGGRESSIVE_REWARD_CONFIG = {
    # Core trading rewards - EXTREME bonuses for ANY trading activity
    'PROFITABLE_TRADE_REWARD': 15.0,     # MASSIVE: Huge reward for winning trades
    'LOSING_TRADE_PENALTY': -0.5,        # MINIMAL: Tiny penalty to encourage trying
    'MARKET_ENGAGEMENT_BONUS': 5.0,      # MASSIVE: Huge bonus just for taking positions
    
    # Position management - Encourage any position taking
    'PROFIT_HOLD_REWARD': 3.0,           # MASSIVE: Strong reward for holding profitable positions
    'LOSS_HOLD_PENALTY': -0.01,          # TINY: Almost no penalty for holding losing positions
    'PROFIT_PROTECTION_BONUS': 2.0,      # Strong bonus for protecting profits
    'SIGNIFICANT_PROFIT_THRESHOLD': 0.001, # LOWERED: Easier to achieve profit threshold
    'SIGNIFICANT_PROFIT_BONUS': 3.0,     # MASSIVE: Extra bonus for any profit
    
    # Activity incentives - FORCE trading behavior
    'HOLD_COST': -0.1,                   # MASSIVE: Huge cost for inaction
    'EXCESSIVE_HOLD_PENALTY': -0.2,      # MASSIVE: Severe penalty for excessive holding
    'INACTIVITY_THRESHOLD': 10,          # AGGRESSIVE: Very short before penalty kicks in
    
    # Risk management - Minimal penalties to encourage exploration
    'INVALID_ACTION_PENALTY': -0.1,      # MINIMAL: Small penalty for invalid actions
    'OVERTRADING_PENALTY': 0.0,          # DISABLED: No overtrading penalty during exploration
    'MIN_POSITION_HOLD': 1,              # MINIMAL: Can close positions immediately
    
    # Performance bonuses - Massive rewards for any success
    'NEW_HIGH_BONUS': 10.0,              # MASSIVE: Huge bonus for new equity highs
    'CONSISTENCY_BONUS': 2.0,            # Strong bonus for consistent performance
    'RISK_ADJUSTED_MULTIPLIER': 3.0,     # MASSIVE: Multiplier for risk-adjusted returns
}

# 🎲 AGGRESSIVE EXPLORATION CALLBACK SETTINGS
AGGRESSIVE_EPSILON_CONFIG = {
    'start_eps': 0.95,                   # MAXIMUM: Start with 95% random actions
    'end_eps': 0.5,                      # HIGH: Maintain 50% exploration even at end
    'decay_timesteps_ratio': 0.95,      # SLOW: Explore for 95% of training
    'min_exploration_rate': 0.7,        # MAXIMUM: 70% minimum exploration throughout
}

def get_aggressive_exploration_config():
    """Get the complete aggressive exploration configuration."""
    return {
        'policy_kwargs': POLICY_KWARGS_AGGRESSIVE_EXPLORATION,
        'model_kwargs': MODEL_KWARGS_AGGRESSIVE_EXPLORATION,
        'reward_config': AGGRESSIVE_REWARD_CONFIG,
        'epsilon_config': AGGRESSIVE_EPSILON_CONFIG
    }

def apply_aggressive_exploration_to_model(model):
    """Apply aggressive exploration parameters to an existing model."""
    # Update exploration coefficient
    model.ent_coef = MODEL_KWARGS_AGGRESSIVE_EXPLORATION['ent_coef']
    model.learning_rate = MODEL_KWARGS_AGGRESSIVE_EXPLORATION['learning_rate']
    model.clip_range = MODEL_KWARGS_AGGRESSIVE_EXPLORATION['clip_range']
    model.clip_range_vf = MODEL_KWARGS_AGGRESSIVE_EXPLORATION['clip_range_vf']
    
    print(f"🚀 Applied AGGRESSIVE exploration:")
    print(f"   - Entropy coefficient: {model.ent_coef} (MAXIMUM)")
    print(f"   - Learning rate: {model.learning_rate}")
    print(f"   - Clip range: {model.clip_range}")
    
    return model

if __name__ == "__main__":
    config = get_aggressive_exploration_config()
    print("🚀 AGGRESSIVE Exploration Configuration:")
    print(f"   - Entropy coefficient: {config['model_kwargs']['ent_coef']} (MAXIMUM)")
    print(f"   - Learning rate: {config['model_kwargs']['learning_rate']}")
    print(f"   - Start epsilon: {config['epsilon_config']['start_eps']}")
    print(f"   - Market engagement bonus: {config['reward_config']['MARKET_ENGAGEMENT_BONUS']}")
    print(f"   - Hold cost: {config['reward_config']['HOLD_COST']} (AGGRESSIVE)")
