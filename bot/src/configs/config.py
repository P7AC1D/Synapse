"""
Main Training Configuration

This configuration implements optimized network architecture for price ratio features.

Key optimizations:
1. Price ratio-based feature processing (8 total features: 6 ratios + 2 runtime)
2. Right-sized model architecture (~20K parameters vs 79K)
3. Smaller LSTM (64 units) and dense layers [32,16] for simple features
4. Time-series window alignment
5. Normalized feature representation
6. Regularized training parameters
"""

import torch as th
from torch import nn
from ..trading.features import FeatureProcessor

# Core training configuration
TRAINING_CONFIG = {
    'total_timesteps': 150000,        # Total training steps
    'eval_freq': 10000,               # Evaluation frequency
    'learning_starts': 1000,          # Initial learning delay
    'train_split': 0.7,              # Training data proportion
    'validation_split': 0.2,          # Validation data proportion
    'test_split': 0.1,               # Test data proportion
    'random_start': False,           # Deterministic start for validation consistency
    'feature_processor': FeatureProcessor,  # Use paper's feature processing
    'window_size': 10                # Time window for feature calculation
}

# Model architecture configured for price ratio features
POLICY_KWARGS = {
    'optimizer_class': th.optim.AdamW,  # AdamW for better regularization
    'lstm_hidden_size': 64,             # Reduced size optimized for 8 simple features
    'n_lstm_layers': 2,                 # Two-layer LSTM for temporal processing
    'enable_critic_lstm': True,         # Use LSTM for value function
    'shared_lstm': False,               # Separate policy and value networks
    'net_arch': {                       
        'pi': [32, 16],                # Smaller policy network for price ratios
        'vf': [32, 16]                 # Smaller value network for price ratios
    },
    'activation_fn': nn.ReLU,           # ReLU activation
    'ortho_init': True,                 # Orthogonal initialization
    'optimizer_kwargs': {
        'eps': 1e-5,
        'weight_decay': 1e-3            # L2 regularization
    }
}

# Training parameters optimized for smaller network with price ratio features
MODEL_KWARGS = {
    'learning_rate': 4e-4,             # Slightly higher LR for smaller network
    'n_steps': 512,                    # Reduced sequence length
    'batch_size': 32,                  # Small batches for better generalization
    'n_epochs': 4,                     # Reduced epochs
    'gamma': 0.99,                     # Discount factor
    'gae_lambda': 0.9,                 # GAE lambda
    'clip_range': 0.1,                 # Conservative clipping
    'clip_range_vf': 0.1,              # Value function clipping
    'normalize_advantage': True,        # Normalize advantages
    'ent_coef': 0.01,                  # Entropy coefficient
    'vf_coef': 0.5,                    # Value function coefficient
    'max_grad_norm': 0.5,              # Gradient clipping
    'use_sde': False,                  # No state-dependent exploration
    'sde_sample_freq': -1,             # SDE sampling frequency
    'target_kl': None,                 # Target KL divergence
    'verbose': 0                       # Verbosity level
}

# Enhanced epsilon configuration for exploration
ENHANCED_EPSILON_CONFIG = {
    'start_eps': 0.9,                  # Starting epsilon value
    'end_eps': 0.2,                    # Final epsilon value after decay
    'min_exploration_rate': 0.4        # Minimum exploration rate to maintain
}

# Validation configuration
VALIDATION_CONFIG = {
    'early_stopping': {
        'enabled': True,                
        'metric': 'validation_return',  
        'patience': 10,                 # Reduced patience for faster adaptation
        'min_improvement': 0.005,       
        'mode': 'maximize',            
        'restore_best_weights': True    
    },
    'selection_criterion': 'validation_return',
    'save_best_only': True,
    'save_frequency': 1,
    'validation_frequency': 1,
    'detailed_validation_logging': True,
    
    # Adaptive validation settings
    'adaptive': {
        'enabled': True,
        'base_return_threshold': -0.05,
        'min_threshold': -0.15,
        'max_stagnation_iterations': 40,  # Reduced for faster adaptation
        'lookback_window': 15,           # Shorter lookback for price ratio features
        'threshold_decay_rate': 0.98,
        'risk_adjustment_factor': 0.3,
        'activity_bonus_factor': 0.02,
        'winrate_bonus_factor': 0.1
    }
}

# Environment settings optimized for price ratio features
ENVIRONMENT_CONFIG = {
    'initial_balance': 10000,
    'balance_per_lot': 500,
    'random_start': False,
    'point_value': 0.01,
    'min_lots': 0.01,
    'max_lots': 1.0,
    'contract_size': 100000
}

def get_training_args():
    """Get complete training arguments optimized for price ratio features."""
    return {
        # Data splits
        'train_split': TRAINING_CONFIG['train_split'],
        'validation_split': TRAINING_CONFIG['validation_split'],
        'test_split': TRAINING_CONFIG['test_split'],

        # Feature processing
        'feature_processor': TRAINING_CONFIG['feature_processor'],
        'window_size': TRAINING_CONFIG['window_size'],

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
