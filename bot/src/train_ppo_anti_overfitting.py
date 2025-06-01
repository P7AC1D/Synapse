#!/usr/bin/env python3
"""
Anti-Overfitting PPO Training Script

This script implements enhanced training with overfitting prevention specifically
designed to address the training/validation performance gap issue:

CURRENT ISSUE:
- Training: 138% return, 59% win rate  
- Validation: -21% return, 43% win rate
- Gap: 160% performance difference

TARGET SOLUTION:
- Balanced performance with <25% train/validation gap
- Both training and validation datasets profitable
- Maintained 5-10x training speedup

USAGE:
    python train_ppo_anti_overfitting.py --data_path ../data/XAUUSDm_15min.csv --profile ultra_conservative
    python train_ppo_anti_overfitting.py --data_path ../data/XAUUSDm_15min.csv --profile conservative
    python train_ppo_anti_overfitting.py --data_path ../data/XAUUSDm_15min.csv --profile default
    python train_ppo_anti_overfitting.py --data_path ../data/XAUUSDm_15min.csv --profile balanced

PROFILES:
    - ultra_conservative: Maximum anti-overfitting (15K timesteps, 2 patience, 10% max gap)
    - conservative: Strong anti-overfitting (25K timesteps, 4 patience, 15% max gap)
    - default: Balanced anti-overfitting (30K timesteps, 5 patience, 25% max gap)  
    - balanced: Moderate anti-overfitting (40K timesteps, 8 patience, 30% max gap)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch as th

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced training utilities
from utils.training_utils_optimized_enhanced import train_walk_forward_enhanced
from configs.anti_overfitting_config import (
    get_anti_overfitting_args, 
    create_anti_overfitting_namespace,
    print_configuration_summary
)

def setup_cuda():
    """Setup CUDA if available."""
    if th.cuda.is_available():
        print(f"🚀 CUDA available: {th.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return 'cuda'
    else:
        print("⚠️ CUDA not available, using CPU")
        return 'cpu'

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """Load and validate trading data."""
    print(f"📊 Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        print(f"✓ Data loaded: {len(data):,} rows, {len(data.columns)} columns")
        
        # Basic validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
          # Convert timestamp/time column if present
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
            print(f"✓ Timestamp index set: {data.index[0]} to {data.index[-1]}")
        elif 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            data = data.set_index('time')
            print(f"✓ Time index set: {data.index[0]} to {data.index[-1]}")
        
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} NaN values - forward filling")
            data = data.fillna(method='ffill')
        
        print(f"📈 Data range: {data.index[0]} to {data.index[-1]}")
        print(f"📊 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def create_results_directory(seed: int) -> str:
    """Create results directory for this training run."""
    results_dir = f"../results/{seed}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/iterations", exist_ok=True)
    
    print(f"📁 Results directory: {results_dir}")
    return results_dir

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Anti-Overfitting PPO Training")
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the CSV data file')
      # Anti-overfitting profile
    parser.add_argument('--profile', type=str, default='default',
                       choices=['ultra_conservative', 'conservative', 'default', 'balanced'],
                       help='Anti-overfitting profile to use')
    
    # Optional overrides
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--initial_window', type=int, default=500,
                       help='Initial training window size')
    parser.add_argument('--step_size', type=int, default=100,
                       help='Step size for walk-forward optimization')
    
    # Trading environment parameters (for compatibility)
    parser.add_argument('--initial_balance', type=float, default=None,
                       help='Initial balance for trading (overrides profile setting)')
    parser.add_argument('--balance_per_lot', type=float, default=None,
                       help='Account balance required per 0.01 lot (overrides profile setting)')
    parser.add_argument('--point_value', type=float, default=None,
                       help='Value of one price point movement (overrides profile setting)')
    parser.add_argument('--min_lots', type=float, default=None,
                       help='Minimum lot size (overrides profile setting)')
    parser.add_argument('--max_lots', type=float, default=None,
                       help='Maximum lot size (overrides profile setting)')
    parser.add_argument('--contract_size', type=float, default=None,
                       help='Standard contract size (overrides profile setting)')
    
    # Performance options
    parser.add_argument('--show_config', action='store_true',
                       help='Show configuration and exit')
    parser.add_argument('--dry_run', action='store_true',
                       help='Validate setup without training')
    
    # Advanced overrides (optional)
    parser.add_argument('--total_timesteps', type=int,
                       help='Override total timesteps')
    parser.add_argument('--early_stopping_patience', type=int,
                       help='Override early stopping patience')
    parser.add_argument('--max_train_val_gap', type=float,
                       help='Override maximum train/validation gap')
    
    return parser.parse_args()

def main():
    """Main training function with anti-overfitting."""
    print("🛡️ ANTI-OVERFITTING PPO TRAINING SYSTEM")
    print("=" * 60)
    
    args = parse_arguments()
      # Show configuration if requested
    if args.show_config:
        print("Available configurations:")
        for profile in ['ultra_conservative', 'conservative', 'default', 'balanced']:
            print_configuration_summary(profile)
            print("-" * 60)
        return
    
    # Display current configuration
    print_configuration_summary(args.profile)
    
    # Setup device
    device = setup_cuda()
    
    # Load anti-overfitting configuration
    config_args = create_anti_overfitting_namespace(args.profile)
    
    # Override with command line arguments
    config_args.seed = args.seed
    config_args.device = device
    
    # Apply trading parameter overrides if provided
    if args.initial_balance is not None:
        config_args.initial_balance = args.initial_balance
        print(f"🔧 Override: initial_balance = {args.initial_balance}")
    
    if args.balance_per_lot is not None:
        config_args.balance_per_lot = args.balance_per_lot
        print(f"🔧 Override: balance_per_lot = {args.balance_per_lot}")
    
    if args.point_value is not None:
        config_args.point_value = args.point_value
        print(f"🔧 Override: point_value = {args.point_value}")
    
    if args.min_lots is not None:
        config_args.min_lots = args.min_lots
        print(f"🔧 Override: min_lots = {args.min_lots}")
    
    if args.max_lots is not None:
        config_args.max_lots = args.max_lots
        print(f"🔧 Override: max_lots = {args.max_lots}")
    
    if args.contract_size is not None:
        config_args.contract_size = args.contract_size
        print(f"🔧 Override: contract_size = {args.contract_size}")
    
    # Apply any manual overrides
    if args.total_timesteps:
        config_args.total_timesteps = args.total_timesteps
        print(f"🔧 Override: total_timesteps = {args.total_timesteps}")
    
    if args.early_stopping_patience:
        config_args.early_stopping_patience = args.early_stopping_patience
        print(f"🔧 Override: early_stopping_patience = {args.early_stopping_patience}")
    
    if args.max_train_val_gap:
        config_args.max_train_val_gap = args.max_train_val_gap
        print(f"🔧 Override: max_train_val_gap = {args.max_train_val_gap:.1%}")
    
    # Load and validate data
    try:
        data = load_and_validate_data(args.data_path)
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return 1
    
    # Create results directory
    results_dir = create_results_directory(args.seed)
    
    # Validate training parameters
    print(f"\n📋 TRAINING SETUP VALIDATION:")
    print(f"Data points: {len(data):,}")
    print(f"Initial window: {args.initial_window:,}")
    print(f"Step size: {args.step_size:,}")
    print(f"Expected iterations: {(len(data) - args.initial_window) // args.step_size + 1}")
    
    validation_size = int(args.initial_window * config_args.validation_size)
    training_size = args.initial_window - validation_size
    print(f"Training size: {training_size:,} ({training_size/args.initial_window:.1%})")
    print(f"Validation size: {validation_size:,} ({config_args.validation_size:.1%})")
    
    if training_size < args.step_size * 2:
        print("❌ Training window too small relative to step size")
        return 1
      # Dry run check
    if args.dry_run:
        print("\n✅ DRY RUN COMPLETED - Configuration validated")
        print("Remove --dry_run flag to start actual training")
        return 0
        print("\n✅ DRY RUN COMPLETED - Configuration validated")
        print("Remove --dry_run flag to start actual training")
        return 0
    
    print(f"\n🚀 STARTING ANTI-OVERFITTING TRAINING")
    print(f"Profile: {args.profile}")
    print(f"Expected improvements:")
    print(f"  • Training/Validation gap: 160% → <{config_args.max_train_val_gap:.0%}")
    print(f"  • Validation return: -21% → 5-15%")
    print(f"  • Validation win rate: 43% → 50-55%")
    
    # Save training configuration
    config_summary = {
        'profile': args.profile,
        'anti_overfitting_enabled': True,
        'training_start': datetime.now().isoformat(),
        'data_path': args.data_path,
        'data_points': len(data),
        'seed': args.seed,
        'device': device,
        'configuration': vars(config_args),
        'expected_improvements': {
            'train_val_gap_target': f"<{config_args.max_train_val_gap:.0%}",
            'validation_return_target': "5-15%",
            'validation_win_rate_target': "50-55%"
        }
    }
    
    import json
    with open(f"{results_dir}/anti_overfitting_config.json", 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    try:
        # Start enhanced training with anti-overfitting
        print(f"\n⚡ Training with enhanced walk-forward optimization...")
        model = train_walk_forward_enhanced(
            data=data,
            initial_window=args.initial_window,
            step_size=args.step_size,
            args=config_args
        )
        
        if model is not None:
            print(f"\n🎉 ANTI-OVERFITTING TRAINING COMPLETED SUCCESSFULLY!")
            
            # Save final model
            final_model_path = f"{results_dir}/final_anti_overfitting_model.zip"
            model.save(final_model_path)
            print(f"💾 Final model saved: {final_model_path}")
            
            # Load and display final summary
            summary_path = f"{results_dir}/enhanced_training_summary.json"
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
                print(f"Total speedup achieved: {summary.get('total_speedup_achieved', 'N/A'):.1f}x")
                print(f"Max train/val gap: {summary.get('max_train_val_gap', 'N/A'):.1%}")
                print(f"Validation improvements: {summary.get('validation_improvements', 'N/A')}")
                print(f"Overfitting stops: {summary.get('overfitting_stops', 'N/A')}")
                print(f"Average iteration time: {summary.get('avg_iteration_time_minutes', 'N/A'):.1f} minutes")
                
                # Check if targets were met
                max_gap = summary.get('max_train_val_gap', 1.0)
                target_gap = config_args.max_train_val_gap
                
                if max_gap <= target_gap:
                    print(f"\n✅ SUCCESS: Training/validation gap target MET!")
                    print(f"   Achieved: {max_gap:.1%} ≤ Target: {target_gap:.1%}")
                else:
                    print(f"\n⚠️ GAP TARGET MISSED:")
                    print(f"   Achieved: {max_gap:.1%} > Target: {target_gap:.1%}")
                    print(f"   Consider using 'conservative' profile for stricter control")
            
            print(f"\n🎯 Next steps:")
            print(f"1. Check validation results in: {results_dir}/iterations/")
            print(f"2. Compare with previous training using regular train_ppo_optimized.py")
            print(f"3. If results are good, use this model for production")
            
            return 0
            
        else:
            print(f"\n❌ Training failed - no model returned")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        print(f"Progress saved - use same command to resume")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
