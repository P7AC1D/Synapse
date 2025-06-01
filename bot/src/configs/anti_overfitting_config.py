"""
Anti-Overfitting Configuration for Trading Bot Training

This configuration is specifically designed to prevent the overfitting issues
identified in the training analysis:
- Training: 138% return, 59% win rate
- Validation: -21% return, 43% win rate

Target: Balanced performance with <20% train/validation gap
"""

# Anti-Overfitting Training Arguments
ANTI_OVERFITTING_ARGS = {
    # Reduced training intensity to prevent overfitting
    'total_timesteps': 30000,      # Reduced from 100K
    'min_timesteps': 15000,        # Conservative minimum
    
    # Enhanced early stopping
    'early_stopping_patience': 5,  # Stop after 5 iterations without validation improvement
    'max_train_val_gap': 0.25,     # Maximum 25% gap between training and validation
    'validation_degradation_threshold': 0.1,  # Stop if validation degrades by 10%
    
    # Conservative hyperparameters
    'learning_rate': 3e-4,         # Lower learning rate
    'validation_size': 0.25,       # Larger validation set (25% vs 20%)
    'eval_freq': 3000,             # More frequent evaluation
    
    # Regularization settings
    'adaptive_timesteps': True,    # Reduce timesteps as training progresses
    'warm_start': True,            # Continue from previous best model
    'cache_environments': True,    # Use caching for speed
    
    # Environment settings (unchanged)
    'initial_balance': 10000,
    'balance_per_lot': 500,
    'random_start': False,
    'point_value': 0.01,
    'min_lots': 0.01,
    'max_lots': 200.0,
    'contract_size': 100.0,
    'device': 'cuda'
}

# Quick start configuration with maximum anti-overfitting
CONSERVATIVE_ARGS = {
    **ANTI_OVERFITTING_ARGS,
    'total_timesteps': 25000,      # Slightly more training (was 20k)
    'min_timesteps': 12000,        # Lower minimum (was 10k)
    'early_stopping_patience': 4,  # Moderate patience (was 3)
    'max_train_val_gap': 0.15,     # Very strict gap limit (15% was 20%)
    'validation_size': 0.3,        # Even larger validation set
}

# Ultra-conservative configuration for severe overfitting cases
ULTRA_CONSERVATIVE_ARGS = {
    **ANTI_OVERFITTING_ARGS,
    'total_timesteps': 15000,      # Very conservative training
    'min_timesteps': 8000,         # Lower minimum
    'early_stopping_patience': 3,  # Reasonable patience (was 2 - too aggressive)
    'max_train_val_gap': 0.25,     # Reasonable gap limit (25% - was 10% too strict)
    'validation_size': 0.35,       # Maximum validation set
    'learning_rate': 1e-4,         # Even lower learning rate
}

# Experimental configuration with moderate anti-overfitting
BALANCED_ARGS = {
    **ANTI_OVERFITTING_ARGS,
    'total_timesteps': 40000,      # Slightly higher timesteps
    'min_timesteps': 20000,        # Higher minimum
    'early_stopping_patience': 8,  # More patience
    'max_train_val_gap': 0.3,      # Allow larger gap (30%)
    'validation_size': 0.2,        # Standard validation size
}

def get_anti_overfitting_args(profile: str = 'default') -> dict:
    """
    Get anti-overfitting configuration based on profile.
    
    Args:
        profile: 'ultra_conservative', 'conservative', 'default', or 'balanced'
        
    Returns:
        Dictionary of training arguments optimized for preventing overfitting
    """
    if profile == 'ultra_conservative':
        return ULTRA_CONSERVATIVE_ARGS.copy()
    elif profile == 'conservative':
        return CONSERVATIVE_ARGS.copy()
    elif profile == 'balanced':
        return BALANCED_ARGS.copy()
    else:
        return ANTI_OVERFITTING_ARGS.copy()

def create_anti_overfitting_namespace(profile: str = 'default'):
    """
    Create a namespace object with anti-overfitting arguments.
    
    This can be used directly with the enhanced training functions.
    """
    class Args:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = get_anti_overfitting_args(profile)
    return Args(config)

# Expected performance improvements with anti-overfitting
EXPECTED_IMPROVEMENTS = {
    'training_validation_gap': {
        'current': '160% gap (138% training vs -21% validation)',
        'target': '<25% gap (both datasets positive)',
        'metric': 'Performance consistency'
    },
    'validation_return': {
        'current': '-21.4% (losing money)',
        'target': '5-15% (profitable)',
        'metric': 'Generalization ability'
    },
    'validation_win_rate': {
        'current': '42.9% (poor)',
        'target': '50-55% (acceptable)',
        'metric': 'Trading consistency'
    },
    'validation_profit_factor': {
        'current': '0.67 (unprofitable)',
        'target': '1.1-1.3 (profitable)',
        'metric': 'Risk-adjusted returns'
    },
    'training_time': {
        'current': '5-10 minutes (optimized)',
        'target': '5-10 minutes (maintained)',
        'metric': 'Development speed'
    }
}

# Usage examples for different scenarios
USAGE_EXAMPLES = {
    'quick_test': {
        'description': 'Fast test with maximum anti-overfitting protection',
        'profile': 'conservative',
        'expected_time': '3-5 minutes per iteration',
        'use_case': 'Testing new features or quick validation'
    },
    'production_training': {
        'description': 'Balanced approach for production models',
        'profile': 'default',
        'expected_time': '5-8 minutes per iteration',
        'use_case': 'Standard model training with overfitting prevention'
    },
    'experimental': {
        'description': 'Moderate anti-overfitting for research',
        'profile': 'balanced',
        'expected_time': '8-12 minutes per iteration',
        'use_case': 'Experimenting with new architectures or features'
    }
}

def print_configuration_summary(profile: str = 'default'):
    """Print a summary of the anti-overfitting configuration."""
    config = get_anti_overfitting_args(profile)
    usage = USAGE_EXAMPLES.get(profile, USAGE_EXAMPLES['production_training'])
    
    print(f"\n🛡️ ANTI-OVERFITTING CONFIGURATION: {profile.upper()}")
    print(f"Description: {usage['description']}")
    print(f"Expected time: {usage['expected_time']}")
    print(f"Use case: {usage['use_case']}")
    
    print(f"\n📊 KEY SETTINGS:")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Minimum timesteps: {config['min_timesteps']:,}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print(f"Max train/val gap: {config['max_train_val_gap']:.1%}")
    print(f"Validation size: {config['validation_size']:.1%}")
    print(f"Evaluation frequency: {config['eval_freq']:,}")
    
    print(f"\n🎯 EXPECTED IMPROVEMENTS:")
    for metric, info in EXPECTED_IMPROVEMENTS.items():
        print(f"{metric}: {info['current']} → {info['target']}")

if __name__ == "__main__":
    # Example usage
    print("Available anti-overfitting profiles:")
    for profile in ['ultra_conservative', 'conservative', 'default', 'balanced']:
        print_configuration_summary(profile)
        print("-" * 60)
