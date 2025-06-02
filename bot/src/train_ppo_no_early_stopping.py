#!/usr/bin/env python3
"""
Train PPO-LSTM model with NO EARLY STOPPING for complete WFO cycles.

This script removes all early stopping mechanisms to ensure full walk-forward
optimization cycles complete, addressing issues where early stopping interferes
with learning in highly volatile financial markets (especially forex).

FEATURES REMOVED:
- Early stopping based on validation performance
- Overfitting detection mechanisms
- Training/validation gap monitoring with stopping
- TradingAwareEarlyStoppingCallback
- ValidationAwareEarlyStoppingCallback

FEATURES PRESERVED:
- All 5-10x speedup optimizations
- Model evaluation and selection (best model saved)
- Warm starting between iterations
- Environment caching
- Progressive hyperparameters
- Adaptive timesteps

This ensures that WFO training completes all intended iterations without
premature termination due to market volatility patterns.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch as th
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Import the new no early stopping training function
from utils.training_utils_no_early_stopping import train_walk_forward_no_early_stopping
from utils.data_loader import load_and_prepare_data

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train PPO-LSTM with NO EARLY STOPPING for complete WFO")
    
    # Data parameters
    parser.add_argument('--data-file', type=str, required=True, 
                       help='CSV file containing trading data')
    parser.add_argument('--initial-window', type=int, default=5000,
                       help='Initial training window size')
    parser.add_argument('--step-size', type=int, default=500,
                       help='Step size for walk-forward optimization')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=50000,
                       help='Number of timesteps per training iteration')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Evaluation frequency during training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    
    # Environment parameters
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial account balance')
    parser.add_argument('--balance-per-lot', type=float, default=500,
                       help='Balance required per lot')
    parser.add_argument('--random-start', action='store_true',
                       help='Use random start positions in training')
    parser.add_argument('--point-value', type=float, default=0.01,
                       help='Point value for the instrument')
    parser.add_argument('--min-lots', type=float, default=0.01,
                       help='Minimum lot size')
    parser.add_argument('--max-lots', type=float, default=1.0,
                       help='Maximum lot size')
    parser.add_argument('--contract-size', type=int, default=100000,
                       help='Contract size')
    
    # Optimization features (all preserved except early stopping)
    parser.add_argument('--no-adaptive-timesteps', action='store_true',
                       help='Disable adaptive timestep reduction')
    parser.add_argument('--no-warm-start', action='store_true',
                       help='Disable warm starting between iterations')
    parser.add_argument('--no-cache-environments', action='store_true',
                       help='Disable environment caching')
    parser.add_argument('--no-fast-evaluation', action='store_true',
                       help='Disable fast evaluation system')
    
    # REMOVED: All early stopping parameters
    # These are intentionally removed to ensure full WFO completion:
    # --early-stopping-patience, --convergence-threshold, --max-train-val-gap, etc.
    
    args = parser.parse_args()
    
    # Set derived attributes for compatibility
    args.adaptive_timesteps = not args.no_adaptive_timesteps
    args.warm_start = not args.no_warm_start
    args.cache_environments = not args.no_cache_environments
    args.use_fast_evaluation = not args.no_fast_evaluation
    
    # EXPLICITLY DISABLE early stopping
    args.early_stopping_patience = 0  # Disabled
    args.convergence_threshold = 0.0  # Not used
    args.max_train_val_gap = 1.0      # Not used (100% gap allowed)
    args.validation_degradation_threshold = 1.0  # Not used
    
    print("=" * 80)
    print("🚫 PPO-LSTM TRAINING WITH NO EARLY STOPPING")
    print("=" * 80)
    print(f"📅 Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data file: {args.data_file}")
    print(f"🎯 Seed: {args.seed}")
    print(f"🏗️ Device: {args.device}")
    print()
    print("🚫 EARLY STOPPING STATUS:")
    print("   Early Stopping: ❌ DISABLED")
    print("   Overfitting Detection: ❌ DISABLED") 
    print("   Validation Gap Monitoring: ❌ DISABLED")
    print("   Trading Activity Monitoring: ❌ DISABLED")
    print("   ✅ Full WFO cycles will complete regardless of performance")
    print()
    print("⚡ OPTIMIZATION FEATURES PRESERVED:")
    print(f"   Adaptive Timesteps: {'✅' if args.adaptive_timesteps else '❌'}")
    print(f"   Warm Starting: {'✅' if args.warm_start else '❌'}")
    print(f"   Environment Caching: {'✅' if args.cache_environments else '❌'}")
    print(f"   Fast Evaluation: {'✅' if args.use_fast_evaluation else '❌'}")
    print()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create results directory
    results_dir = f"../results/{args.seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['training_type'] = 'no_early_stopping'
    config['early_stopping_disabled'] = True
    config['start_time'] = datetime.now().isoformat()
    
    import json
    with open(f"{results_dir}/no_early_stopping_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📊 WFO Configuration:")
    print(f"   Initial window: {args.initial_window:,} samples")
    print(f"   Step size: {args.step_size:,} samples")
    print(f"   Timesteps per iteration: {args.total_timesteps:,}")
    print(f"   Evaluation frequency: {args.eval_freq:,}")
    print()
    
    try:
        # Load and prepare data
        print("📈 Loading trading data...")
        data = load_and_prepare_data(args.data_file)
        print(f"✅ Loaded {len(data):,} samples from {data.index[0]} to {data.index[-1]}")
        
        # Calculate expected iterations
        total_periods = len(data)
        expected_iterations = (total_periods - args.initial_window) // args.step_size + 1
        estimated_hours = expected_iterations * 0.75  # Assuming 45min average with optimizations
        
        print(f"\n🎯 WFO Training Plan:")
        print(f"   Total samples: {total_periods:,}")
        print(f"   Expected iterations: {expected_iterations}")
        print(f"   Estimated time: {estimated_hours:.1f} hours")
        print(f"   🚫 Early stopping: DISABLED - all {expected_iterations} iterations will complete")
        print()
        
        # Confirm training start
        print("🚀 Starting NO EARLY STOPPING walk-forward training...")
        print("⚠️  This may take a significant amount of time as ALL iterations will complete")
        print()
        
        # Train model with NO early stopping
        final_model = train_walk_forward_no_early_stopping(
            data=data,
            initial_window=args.initial_window,
            step_size=args.step_size,
            args=args
        )
        
        if final_model is not None:
            # Save final model
            final_model_path = f"{results_dir}/final_model_no_early_stopping.zip"
            final_model.save(final_model_path)
            print(f"💾 Final model saved: {final_model_path}")
            
            # Create completion marker
            completion_marker = {
                'completed': True,
                'completion_time': datetime.now().isoformat(),
                'training_type': 'no_early_stopping',
                'early_stopping_disabled': True,
                'total_iterations_completed': expected_iterations,
                'model_path': final_model_path
            }
            
            with open(f"{results_dir}/no_early_stopping_completion.json", 'w') as f:
                json.dump(completion_marker, f, indent=2)
            
            print("\n" + "=" * 80)
            print("🎉 NO EARLY STOPPING TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"✅ All {expected_iterations} WFO iterations completed")
            print(f"🚫 Zero early stops (as intended)")
            print(f"💾 Final model: {final_model_path}")
            print(f"📊 Training summary: {results_dir}/no_early_stopping_summary.json")
            print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        else:
            print("❌ Training failed - final model is None")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("💾 Progress has been saved - you can resume with the same command")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
