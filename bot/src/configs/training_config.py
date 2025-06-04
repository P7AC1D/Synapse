"""
Regularized Training Configuration

This configuration implements critical fixes for overfitting issues:
1. Stronger regularization through model architecture changes
2. Validation-based model selection
3. Improved data splitting (70/20/10)
4. Gradient clipping and reduced learning rates
5. Early stopping based on validation performance
"""

import torch as th
from torch import nn

# Core training configuration
TRAINING_CONFIG = {
    'total_timesteps': 150000,        # Total training steps
    'eval_freq': 10000,               # More frequent evaluation
    'learning_starts': 1000,         # Initial learning delay
    'train_split': 0.7,             # Training data proportion
    'validation_split': 0.2,         # Validation data proportion
    'test_split': 0.1,              # Test data proportion
    'random_start': False,           # Deterministic start for validation consistency
}

# Model policy configuration with regularization
POLICY_KWARGS = {
    'optimizer_class': th.optim.AdamW,  # Use AdamW for better regularization
    'lstm_hidden_size': 256,            # Reduced from original larger size
    'n_lstm_layers': 2,                # Reduced from 4 to prevent overfitting
    'enable_critic_lstm': True,         # Use LSTM for value function
    'shared_lstm': False,
    'net_arch': {                       # Reduced network sizes
        'pi': [128, 128],              # Policy network
        'vf': [128, 128]               # Value network
    },
    'activation_fn': nn.ReLU,           # Changed from Tanh for better gradients
    'ortho_init': True,                 # Orthogonal initialization for stability
    'optimizer_kwargs': {
        'eps': 1e-5,
        'weight_decay': 1e-3            # L2 regularization
    }
}

# Model training parameters with conservative settings
MODEL_KWARGS = {
    'learning_rate': 5e-4,             # Reduced for stability
    'n_steps': 1024,                   # Reasonable sequence length
    'batch_size': 64,                  # Smaller batches for regularization
    'n_epochs': 4,                     # Reduced from 10 to prevent overtraining
    'gamma': 0.99,                     # Discount factor
    'gae_lambda': 0.9,                 # Reduced GAE lambda
    'clip_range': 0.1,                 # Reduced from 0.2 for stability
    'clip_range_vf': 0.1,              # Value function clipping
    'normalize_advantage': True,        # Normalize advantages
    'ent_coef': 0.01,                  # Increased entropy coefficient
    'vf_coef': 0.5,                    # Value function coefficient
    'max_grad_norm': 0.5,              # Gradient clipping
    'use_sde': False,                  # No state-dependent exploration
    'sde_sample_freq': -1,             # SDE sampling frequency
    'target_kl': None,                 # Target KL divergence
    'verbose': 0                       # Verbosity level
}

# Enhanced validation configuration
VALIDATION_CONFIG = {
    'early_stopping': {
        'enabled': True,                # Enable early stopping
        'metric': 'validation_return',  # Monitor validation returns
        'patience': 15,                 # Increased patience for adaptive mode
        'min_improvement': 0.005,       # Reduced minimum improvement (0.5%)
        'mode': 'maximize',             # Maximize validation returns
        'restore_best_weights': True    # Restore best model weights
    },
    'selection_criterion': 'validation_return',  # Use validation instead of combined
    'save_best_only': True,
    'save_frequency': 1,                # Save every iteration
    'validation_frequency': 1,          # Validate every iteration
    'detailed_validation_logging': True,
      # Adaptive validation settings
    'adaptive': {
        'enabled': True,                # Enable adaptive validation
        'base_return_threshold': -0.05,  # Start with -5% threshold
        'min_threshold': -0.15,         # Never go below -15%
        'max_stagnation_iterations': 50, # Reset after 50 iterations
        'lookback_window': 20,          # Analyze last 20 iterations
        'threshold_decay_rate': 0.98,   # Gradual relaxation
        'risk_adjustment_factor': 0.3,  # Weight for risk metrics
        'activity_bonus_factor': 0.02,  # Bonus for trading activity
        'winrate_bonus_factor': 0.1,    # Bonus for win rate
    },
    
    # Walk-Forward Model Selection settings
    'wfo_model_selection': {
        'enabled': True,                # Enable improved WFO model selection
        'strategy': 'ensemble_validation',  # Strategy: ensemble_validation, rolling_validation, risk_adjusted
        'fallback_to_legacy': True,     # Fallback to legacy comparison if improved fails
        'warn_about_legacy': True       # Show warnings when using legacy comparison
    }
}

# Enhanced epsilon configuration for exploration
ENHANCED_EPSILON_CONFIG = {
    'start_eps': 0.9,                  # Starting epsilon value
    'end_eps': 0.2,                    # Final epsilon value after decay
    'min_exploration_rate': 0.4        # Minimum exploration rate to maintain
}

# Environment settings
ENVIRONMENT_CONFIG = {
    'initial_balance': 10000,
    'balance_per_lot': 500,
    'random_start': False,             # Deterministic start for validation consistency
    'point_value': 0.01,
    'min_lots': 0.01,
    'max_lots': 1.0,                   # Conservative position sizing
    'contract_size': 100000
}

def get_training_args():
    """Get complete training arguments with regularization."""
    return {
        # Data splits
        'train_split': TRAINING_CONFIG['train_split'],
        'validation_split': TRAINING_CONFIG['validation_split'],
        'test_split': TRAINING_CONFIG['test_split'],

        # Model selection
        'model_selection_metric': VALIDATION_CONFIG['selection_criterion'],
        'save_best_validation_only': VALIDATION_CONFIG['save_best_only'],

        # Regularization
        'learning_rate': MODEL_KWARGS['learning_rate'],
        'max_grad_norm': MODEL_KWARGS['max_grad_norm'],
        'weight_decay': POLICY_KWARGS['optimizer_kwargs']['weight_decay'],

        # Early stopping
        'early_stopping_patience': VALIDATION_CONFIG['early_stopping']['patience'],
        'early_stopping_min_improvement': VALIDATION_CONFIG['early_stopping']['min_improvement'],
        'early_stopping_metric': VALIDATION_CONFIG['early_stopping']['metric'],

        # Training constraints
        'max_timesteps_per_iteration': TRAINING_CONFIG['total_timesteps'],
        'max_iterations': 25,

        # Validation
        'validation_frequency': VALIDATION_CONFIG['validation_frequency'],
        'detailed_validation': VALIDATION_CONFIG['detailed_validation_logging']
    }

if __name__ == "__main__":
    print("Regularized Training Configuration")
    print("=" * 50)
    print(f"Learning Rate: {MODEL_KWARGS['learning_rate']}")
    print(f"Epochs: {MODEL_KWARGS['n_epochs']}")
    print(f"Clip Range: {MODEL_KWARGS['clip_range']}")
    print(f"Network Architecture: {POLICY_KWARGS['net_arch']}")
    print(f"Data Splits: {TRAINING_CONFIG['train_split']}/{TRAINING_CONFIG['validation_split']}/{TRAINING_CONFIG['test_split']}")
