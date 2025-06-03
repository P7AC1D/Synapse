"""
Regularized Training Configuration - Addresses Overfitting Issues

This configuration implements the critical fixes identified in the generalization analysis:
1. Stronger regularization through model architecture changes
2. Validation-based model selection (instead of combined dataset selection)
3. Improved data splitting (70/20/10 instead of 90/10/0)
4. Gradient clipping and reduced learning rates
5. Early stopping based on validation performance

Based on analysis showing:
- 1,169% performance gap between training and validation
- Current 90/10 data split insufficient for robust validation
- Need for validation-based selection to prevent overfitting memorization
"""

import torch as th
from typing import Dict, Any

# ====== REGULARIZED MODEL CONFIGURATION ======

REGULARIZED_POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 256,              # Reduced from 512 to prevent overfitting
    "n_lstm_layers": 2,                   # Reduced from 4 to prevent overfitting
    "shared_lstm": False,
    "enable_critic_lstm": True,
    "net_arch": {
        "pi": [128, 128],                 # Reduced from [256, 256] 
        "vf": [128, 128]                  # Reduced from [256, 256]
    },
    "activation_fn": th.nn.ReLU,          # Stable activation
    "ortho_init": True,                   # Orthogonal initialization for stability
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-3,             # Increased weight decay for regularization
    }
}

REGULARIZED_MODEL_KWARGS = {
    # Reduced learning parameters to prevent overfitting
    'learning_rate': 0.0005,             # Reduced from 0.001 (50% reduction)
    'n_steps': 1024,                     # Reasonable sequence length
    'batch_size': 64,                    # Smaller batches for regularization
    'n_epochs': 4,                       # Fewer epochs to prevent overtraining
    'gamma': 0.99,                       # Standard discount factor
    'gae_lambda': 0.9,                   # Reduced GAE lambda
      # Clipping for regularization
    'clip_range': 0.1,                   # Reduced from 0.2 (conservative clipping)
    'clip_range_vf': 0.1,                # Value function clipping
    'max_grad_norm': 0.5,                # Gradient clipping for regularization
      # Optimizer settings with weight decay
    'optimizer_kwargs': {
        'eps': 1e-5,
        'weight_decay': 1e-3,             # Increased weight decay for regularization
    },
    
    # Exploration parameters
    'ent_coef': 0.01,                    # Increased entropy coefficient
    'vf_coef': 0.5,                      # Balanced value function coefficient
}

# ====== DATA SPLITTING CONFIGURATION ======

REGULARIZED_DATA_CONFIG = {
    # Improved data splits for better validation
    'train_split': 0.7,                  # 70% training (reduced from ~90%)
    'validation_split': 0.2,             # 20% validation (increased from ~10%)
    'test_split': 0.1,                   # 10% test set (new)
    
    # Temporal integrity
    'temporal_split': True,              # Maintain chronological order
    'shuffle_training': False,           # Don't shuffle to preserve time series structure
    
    # Data augmentation for regularization
    'add_noise_during_training': True,
    'noise_std': 0.01,                   # 1% noise injection
}

# ====== VALIDATION-BASED MODEL SELECTION ======

REGULARIZED_VALIDATION_CONFIG = {
    # Model selection criteria (CRITICAL CHANGE)
    'selection_criterion': 'validation_return',  # Use validation instead of combined
    'save_best_only': True,
    'save_frequency': 1,                 # Save every iteration
    
    # Early stopping based on validation performance
    'early_stopping': {
        'enabled': True,
        'patience': 10,                  # Stop if no improvement for 10 iterations
        'min_improvement': 0.01,         # Minimum 1% improvement required
        'metric': 'validation_return',   # Monitor validation returns
        'mode': 'max',                   # Maximize validation returns
        'restore_best_weights': True,
    },
    
    # Validation frequency
    'validation_frequency': 1,           # Validate every iteration
    'detailed_validation_logging': True,
}

# ====== REGULARIZED TRAINING PIPELINE CONFIG ======

REGULARIZED_TRAINING_CONFIG = {
    # Core model configuration
    'policy_kwargs': REGULARIZED_POLICY_KWARGS,
    'model_kwargs': REGULARIZED_MODEL_KWARGS,
    
    # Data handling
    'data_config': REGULARIZED_DATA_CONFIG,
    
    # Validation and model selection
    'validation_config': REGULARIZED_VALIDATION_CONFIG,
    
    # Training parameters
    'total_timesteps': 40000,            # Reduced from 50000 to prevent overtraining
    'eval_freq': 3000,                   # More frequent evaluation
    
    # Environment settings
    'initial_balance': 10000,
    'balance_per_lot': 500,
    'random_start': False,               # Deterministic start for validation consistency
    'point_value': 0.01,
    'min_lots': 0.01,
    'max_lots': 1.0,                     # Conservative position sizing
    'contract_size': 100000,
}

# ====== REGULARIZED TRAINING ARGUMENTS ======

def get_regularized_training_args():
    """Get training arguments with regularization fixes."""
    
    return {
        # Data splits - CRITICAL CHANGE
        'train_split': 0.7,               # Reduced from ~0.9
        'validation_split': 0.2,          # Increased from ~0.1
        'test_split': 0.1,                # New test set
        
        # Model selection - CRITICAL CHANGE
        'model_selection_metric': 'validation_return',  # Use validation instead of combined
        'save_best_validation_only': True,
        
        # Regularization
        'learning_rate': 0.0005,          # Reduced learning rate
        'max_grad_norm': 0.5,             # Gradient clipping
        'weight_decay': 1e-3,             # L2 regularization
        
        # Early stopping
        'early_stopping_patience': 10,
        'early_stopping_min_improvement': 0.01,
        'early_stopping_metric': 'validation_return',
        
        # Training constraints
        'max_timesteps_per_iteration': 40000,  # Reduced to prevent overtraining
        'max_iterations': 25,                  # Limit total iterations
        
        # Validation
        'validation_frequency': 1,
        'detailed_validation': True,
    }

def create_regularized_namespace(base_args=None):
    """Create namespace object with regularized training arguments."""
    
    import argparse
    
    # Start with base args if provided
    if base_args is not None:
        namespace = argparse.Namespace(**vars(base_args))
    else:
        namespace = argparse.Namespace()
    
    # Apply regularized training arguments
    reg_args = get_regularized_training_args()
    for key, value in reg_args.items():
        setattr(namespace, key, value)
    
    # Ensure compatibility with existing training system
    namespace.adaptive_timesteps = True
    namespace.warm_start = True
    namespace.cache_environments = True
    namespace.use_fast_evaluation = True
    
    return namespace

# ====== VALIDATION UTILITIES ======

def get_data_splits(data, config=None):
    """Split data according to regularized configuration."""
    
    if config is None:
        config = REGULARIZED_DATA_CONFIG
    
    total_length = len(data)
    
    # Calculate split points
    train_end = int(total_length * config['train_split'])
    val_end = int(total_length * (config['train_split'] + config['validation_split']))
    
    # Split data maintaining temporal order
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"📊 Regularized Data Splits:")
    print(f"   Training: {len(train_data):,} samples ({len(train_data)/total_length:.1%})")
    print(f"   Validation: {len(val_data):,} samples ({len(val_data)/total_length:.1%})")
    print(f"   Test: {len(test_data):,} samples ({len(test_data)/total_length:.1%})")
    
    return train_data, val_data, test_data

def validate_regularization_config():
    """Validate that regularization configuration addresses identified issues."""
    
    print("🔍 Validating Regularization Configuration:")
    print("=" * 60)
    
    issues_addressed = []
    
    # Check data splitting improvement
    train_split = REGULARIZED_DATA_CONFIG['train_split']
    val_split = REGULARIZED_DATA_CONFIG['validation_split']
    
    if train_split <= 0.8 and val_split >= 0.15:
        issues_addressed.append("✅ Data splitting: Improved to 70/20/10 from 90/10/0")
    else:
        print("❌ Data splitting: Still problematic")
    
    # Check model selection change
    selection_criterion = REGULARIZED_VALIDATION_CONFIG['selection_criterion']
    if selection_criterion == 'validation_return':
        issues_addressed.append("✅ Model selection: Changed to validation-based from combined")
    else:
        print("❌ Model selection: Still using combined dataset")
    
    # Check regularization strength
    learning_rate = REGULARIZED_MODEL_KWARGS['learning_rate']
    max_grad_norm = REGULARIZED_MODEL_KWARGS['max_grad_norm']
    weight_decay = REGULARIZED_POLICY_KWARGS['optimizer_kwargs']['weight_decay']
    
    if learning_rate <= 0.0005 and max_grad_norm <= 0.5 and weight_decay >= 1e-3:
        issues_addressed.append("✅ Regularization: Stronger constraints applied")
    else:
        print("❌ Regularization: Insufficient constraints")
    
    # Check architecture reduction
    net_arch = REGULARIZED_POLICY_KWARGS['net_arch']
    if max(max(net_arch['pi']), max(net_arch['vf'])) <= 128:
        issues_addressed.append("✅ Architecture: Reduced complexity to prevent overfitting")
    else:
        print("❌ Architecture: Still too complex")
    
    # Check early stopping
    early_stopping = REGULARIZED_VALIDATION_CONFIG['early_stopping']
    if early_stopping['enabled'] and early_stopping['metric'] == 'validation_return':
        issues_addressed.append("✅ Early stopping: Validation-based stopping implemented")
    else:
        print("❌ Early stopping: Not properly configured")
    
    print(f"\n📈 Issues Addressed: {len(issues_addressed)}/5")
    for issue in issues_addressed:
        print(f"   {issue}")
    
    if len(issues_addressed) == 5:
        print("\n🎯 Configuration successfully addresses all overfitting issues!")
        return True
    else:
        print(f"\n⚠️  Configuration addresses {len(issues_addressed)}/5 issues - review needed")
        return False

if __name__ == "__main__":
    print("🔧 Regularized Training Configuration")
    print("=" * 50)
    
    # Validate configuration
    config_valid = validate_regularization_config()
    
    print(f"\n📋 Configuration Summary:")
    print(f"   Policy Architecture: {REGULARIZED_POLICY_KWARGS['net_arch']}")
    print(f"   Learning Rate: {REGULARIZED_MODEL_KWARGS['learning_rate']}")
    print(f"   Data Splits: {REGULARIZED_DATA_CONFIG['train_split']}/{REGULARIZED_DATA_CONFIG['validation_split']}/{REGULARIZED_DATA_CONFIG['test_split']}")
    print(f"   Selection Criterion: {REGULARIZED_VALIDATION_CONFIG['selection_criterion']}")
    print(f"   Early Stopping: {REGULARIZED_VALIDATION_CONFIG['early_stopping']['enabled']}")
    
    if config_valid:
        print("\n✅ Ready to integrate with training pipeline!")
    else:
        print("\n❌ Configuration needs review before integration")
