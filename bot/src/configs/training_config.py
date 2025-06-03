"""
Training Configuration Settings

This module defines the configuration parameters for training the DRL trading bot,
including model architecture, training hyperparameters, and validation settings.
"""

from torch import nn

# Core training configuration
TRAINING_CONFIG = {
    'total_timesteps': 40000,        # Total training steps
    'eval_freq': 10000,               # Evaluation frequency
    'learning_starts': 1000,         # Initial learning delay
    'train_split': 0.7,             # Training data proportion
    'validation_split': 0.2,         # Validation data proportion
    'test_split': 0.1               # Test data proportion
}

# Model policy configuration
POLICY_KWARGS = {
    'lstm_hidden_size': 256,         # LSTM hidden layer size
    'n_lstm_layers': 2,             # Number of LSTM layers
    'enable_critic_lstm': True,      # Use LSTM for value function
    'net_arch': {                    # Network architecture
        'pi': [128, 128],           # Policy network
        'vf': [128, 128]            # Value network
    },
    'activation_fn': nn.Tanh,        # Activation function
    'optimizer_kwargs': {            # Optimizer settings
        'weight_decay': 1e-3         # L2 regularization
    }
}

# Model training parameters
MODEL_KWARGS = {
    'learning_rate': 5e-4,          # Learning rate
    'n_steps': 1024,                # Steps per update
    'batch_size': 64,               # Mini-batch size
    'n_epochs': 10,                 # Training epochs
    'gamma': 0.99,                  # Discount factor
    'gae_lambda': 0.95,             # GAE parameter
    'clip_range': 0.2,              # PPO clip range
    'clip_range_vf': None,          # Value function clip
    'normalize_advantage': True,    # Normalize advantages
    'ent_coef': 0.01,               # Entropy coefficient
    'vf_coef': 0.5,                 # Value function coefficient
    'max_grad_norm': 0.5,           # Gradient clipping
    'use_sde': False,               # State-dependent exploration
    'sde_sample_freq': -1,          # SDE sampling frequency
    'target_kl': None,              # Target KL divergence
    'verbose': 0                    # Verbosity level
}

# Validation configuration
VALIDATION_CONFIG = {
    'early_stopping': {
        'enabled': True,            # Enable early stopping
        'metric': 'validation_return',  # Metric to monitor
        'patience': 10,            # Patience counter
        'min_improvement': 0.001,  # Minimum improvement threshold
        'mode': 'maximize'         # Optimization mode
    }
}
