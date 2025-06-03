"""
Enhanced exploration configuration to fix the "no trades" problem.

This configuration significantly increases exploration to encourage the model 
to discover profitable trading strategies instead of staying in HOLD-only mode.
"""
import torch as th

# 🚀 PHASE 2: ENHANCED EXPLORATION MODEL CONFIGURATION
POLICY_KWARGS_ENHANCED_EXPLORATION = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 512,              # Phase 2: 4x increase (128→512)
    "n_lstm_layers": 4,                   # Phase 2: 4x increase (1→4)
    "shared_lstm": False,                 # Separate LSTM architectures
    "enable_critic_lstm": True,           # Enable LSTM for value estimation
    "net_arch": {
        "pi": [512, 256, 128],            # Phase 2: Enhanced policy network
        "vf": [512, 256, 128]             # Phase 2: Enhanced value network
    },
    "activation_fn": th.nn.Mish,          # Better activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-4              # Enhanced regularization
    }
}

# 🚀 ENHANCED EXPLORATION TRAINING PARAMETERS
MODEL_KWARGS_ENHANCED_EXPLORATION = {
    'learning_rate': 5e-4,               # INCREASED: Higher LR for faster exploration
    'n_steps': 1024,                     # Longer sequences for 4-layer LSTM
    'batch_size': 512,                   # Larger batch for stability
    'n_epochs': 10,                      # Optimal for larger model
    'gamma': 0.995,                      # Higher gamma for complex patterns
    'gae_lambda': 0.98,                  # Higher lambda for advantage estimation
    'clip_range': lambda progress: 0.15 + 0.05 * (1 - progress),  # FIXED: Callable clip range (starts 0.20, ends 0.15)
    'clip_range_vf': lambda progress: 0.15 + 0.05 * (1 - progress),  # FIXED: Callable clip range for value function
    'ent_coef': 0.35,                    # REDUCED: More moderate exploration (was 0.50, too high)
    'vf_coef': 0.5,                      # Balanced value learning
    'max_grad_norm': 0.5,                # Conservative gradient clipping
    'verbose': 0
}

# 🎯 ENHANCED REWARD SYSTEM PARAMETERS
ENHANCED_REWARD_CONFIG = {
    # Core trading rewards
    'PROFITABLE_TRADE_REWARD': 8.0,      # INCREASED: Stronger reward for winning trades
    'LOSING_TRADE_PENALTY': -1.5,        # REDUCED: Less harsh penalty to encourage trying
    'MARKET_ENGAGEMENT_BONUS': 2.0,      # INCREASED: Strong bonus for taking positions
    
    # Position management  
    'PROFIT_HOLD_REWARD': 2.0,           # INCREASED: Strong reward for holding profitable positions
    'LOSS_HOLD_PENALTY': -0.05,          # Tiny penalty for holding losing positions
    'PROFIT_PROTECTION_BONUS': 1.0,      # Bonus for protecting unrealized profits
    'SIGNIFICANT_PROFIT_THRESHOLD': 0.003, # 0.3% profit threshold for bonus rewards
    'SIGNIFICANT_PROFIT_BONUS': 1.5,     # Extra bonus for significantly profitable positions
    
    # Activity incentives
    'HOLD_COST': -0.02,                  # INCREASED: Higher cost for inaction (was -0.005)
    'EXCESSIVE_HOLD_PENALTY': -0.05,     # Penalty for excessive holding
    'INACTIVITY_THRESHOLD': 30,          # REDUCED: Steps before inactivity penalty kicks in
    
    # Risk management
    'INVALID_ACTION_PENALTY': -2.0,      # Penalty for invalid actions
    'OVERTRADING_PENALTY': -1.0,         # Penalty for too frequent trading
    'MIN_POSITION_HOLD': 3,              # Minimum bars to hold position
    
    # Performance bonuses
    'NEW_HIGH_BONUS': 3.0,               # INCREASED: Bonus for new equity highs
    'CONSISTENCY_BONUS': 0.5,            # Bonus for consistent performance
    'RISK_ADJUSTED_MULTIPLIER': 2.0,     # INCREASED: Multiplier for risk-adjusted returns
}

# 🎲 ENHANCED EXPLORATION CALLBACK SETTINGS
ENHANCED_EPSILON_CONFIG = {
    'start_eps': 0.9,                    # INCREASED: Start with very high exploration (was 0.25)
    'end_eps': 0.2,                      # INCREASED: Maintain significant exploration (was 0.05)
    'decay_timesteps_ratio': 0.9,       # Slow decay - explore for 90% of training
    'min_exploration_rate': 0.4,        # INCREASED: High minimum exploration (was 0.3)
}

def get_enhanced_exploration_config():
    """Get the complete enhanced exploration configuration."""
    return {
        'policy_kwargs': POLICY_KWARGS_ENHANCED_EXPLORATION,
        'model_kwargs': MODEL_KWARGS_ENHANCED_EXPLORATION,
        'reward_config': ENHANCED_REWARD_CONFIG,
        'epsilon_config': ENHANCED_EPSILON_CONFIG
    }

def apply_enhanced_exploration_to_model(model):
    """Apply enhanced exploration parameters to an existing model."""
    # Update exploration coefficient
    model.ent_coef = MODEL_KWARGS_ENHANCED_EXPLORATION['ent_coef']
    model.learning_rate = MODEL_KWARGS_ENHANCED_EXPLORATION['learning_rate']
    model.clip_range = MODEL_KWARGS_ENHANCED_EXPLORATION['clip_range']
    model.clip_range_vf = MODEL_KWARGS_ENHANCED_EXPLORATION['clip_range_vf']
    
    print(f"✅ Applied enhanced exploration:")
    print(f"   - Entropy coefficient: {model.ent_coef}")
    print(f"   - Learning rate: {model.learning_rate}")
    print(f"   - Clip range: {model.clip_range}")
    
    return model

if __name__ == "__main__":
    config = get_enhanced_exploration_config()
    print("🚀 Enhanced Exploration Configuration:")
    print(f"   - Entropy coefficient: {config['model_kwargs']['ent_coef']}")
    print(f"   - Learning rate: {config['model_kwargs']['learning_rate']}")
    print(f"   - Start epsilon: {config['epsilon_config']['start_eps']}")
    print(f"   - Market engagement bonus: {config['reward_config']['MARKET_ENGAGEMENT_BONUS']}")
    print(f"   - Hold cost: {config['reward_config']['HOLD_COST']}")
