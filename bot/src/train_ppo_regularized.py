#!/usr/bin/env python3
"""
Regularized PPO Training Script - Addresses Overfitting Issues

This script implements the regularized training pipeline to address the critical
overfitting issues identified in the generalization analysis:

- 1,169% performance gap (Training: +1,146%, Validation: -23.7%)
- Improper data splitting (90/10 → 70/20/10)
- Combined dataset model selection bias → Validation-only selection
- Insufficient regularization → Stronger constraints
- Oversized architecture → Reduced complexity
- No early stopping → Validation-based early stopping

Usage:
    python train_ppo_regularized.py --seed 1007 --total-timesteps 40000

Key Changes:
- Uses regularized_training_utils instead of training_utils_no_early_stopping
- Implements validation-only model selection
- 70/20/10 data splits instead of 90/10
- Reduced architecture complexity
- Early stopping on validation performance
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.regularized_training_utils import (
    train_regularized_walk_forward,
    validate_regularization_implementation
)

def load_and_prepare_data(args):
    """Load and prepare data for regularized training."""
    print("📊 Loading and preparing data for regularized training...")
    
    # Load data
    data_path = getattr(args, 'data_path', '../data/XAUUSD_M15_enriched_features.csv')
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    data = pd.read_csv(data_path)
    
    # Basic data preparation
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
    
    print(f"✅ Data loaded: {len(data):,} samples")
    print(f"   Period: {data.index[0]} to {data.index[-1]}")
    print(f"   Features: {data.shape[1]} columns")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Regularized PPO Training for DRL Trading Bot")
      # Core training parameters (regularized defaults)
    parser.add_argument('--seed', type=int, default=1007, help='Random seed for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=40000, dest='total_timesteps', help='Reduced timesteps for regularization')
    parser.add_argument('--initial-window', type=int, default=5000, dest='initial_window', help='Initial training window size')
    parser.add_argument('--step-size', type=int, default=500, dest='step_size', help='Walk-forward step size')
    
    # Environment parameters
    parser.add_argument('--initial-balance', type=float, default=10000, dest='initial_balance', help='Initial account balance')
    parser.add_argument('--balance-per-lot', type=float, default=500, dest='balance_per_lot', help='Balance per lot sizing')
    parser.add_argument('--point-value', type=float, default=0.01, dest='point_value', help='Point value for XAUUSD')
    parser.add_argument('--min-lots', type=float, default=0.01, dest='min_lots', help='Minimum lot size')
    parser.add_argument('--max-lots', type=float, default=1.0, dest='max_lots', help='Maximum lot size')
    parser.add_argument('--contract-size', type=float, default=100000, dest='contract_size', help='Contract size')
    
    # Data and paths
    parser.add_argument('--data-path', type=str, default='../data/XAUUSD_M15_enriched_features.csv', 
                       dest='data_path', help='Path to training data')
    parser.add_argument('--device', type=str, default='auto', help='Device for training (auto/cpu/cuda)')
    
    # Regularization-specific parameters
    parser.add_argument('--eval-freq', type=int, default=3000, dest='eval_freq', help='Evaluation frequency (more frequent for regularization)')
    parser.add_argument('--validate-config', action='store_true', dest='validate_config', help='Validate regularization configuration before training')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🛡️ REGULARIZED PPO TRAINING - OVERFITTING FIXES")
    print(f"{'='*60}")
    print(f"Purpose: Address 1,169% performance gap (Training +1,146%, Validation -23.7%)")
    print(f"Seed: {args.seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Validate regularization configuration
    if args.validate_config:
        print(f"\n🔍 VALIDATING REGULARIZATION CONFIGURATION...")
        validation_results = validate_regularization_implementation()
        
        if validation_results['overall_status'] != 'PASSED':
            print(f"\n❌ Regularization validation failed. Please fix configuration issues.")
            return 1
        else:
            print(f"\n✅ Regularization validation passed. All fixes implemented correctly.")
    
    # Create results directory
    results_dir = f"../results/{args.seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_and_prepare_data(args)
    if data is None:
        return 1
    
    print(f"\n🚀 STARTING REGULARIZED TRAINING...")
    print(f"Key Features:")
    print(f"  • Validation-only model selection (NO combined dataset bias)")
    print(f"  • 70/20/10 data splits (improved from 90/10)")
    print(f"  • Early stopping on validation performance")
    print(f"  • Reduced architecture complexity")
    print(f"  • Stronger regularization constraints")
    print(f"  • Learning rate: 0.0005 (reduced from 0.001)")
    print(f"  • Gradient clipping: 0.5")
    print(f"  • Weight decay: 1e-3")
    
    # Run regularized training
    try:
        start_time = datetime.now()
        
        # Train with regularization
        final_model = train_regularized_walk_forward(
            data=data,
            initial_window=args.initial_window,
            step_size=args.step_size,
            args=args
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\n✅ REGULARIZED TRAINING COMPLETED!")
        print(f"Training Duration: {training_duration}")
        print(f"Results Directory: {results_dir}")
        
        # Save training summary
        summary = {
            'training_type': 'regularized_walk_forward',
            'purpose': 'Address overfitting issues (1,169% performance gap)',
            'key_fixes': [
                'Validation-only model selection',
                '70/20/10 data splits',
                'Early stopping on validation',
                'Reduced architecture complexity',
                'Stronger regularization'
            ],
            'seed': args.seed,
            'total_timesteps': args.total_timesteps,
            'initial_window': args.initial_window,
            'step_size': args.step_size,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': training_duration.total_seconds(),
            'data_samples': len(data),
            'results_directory': results_dir
        }
        
        summary_path = os.path.join(results_dir, 'regularized_training_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training Summary: {summary_path}")
        
        # Final validation check
        print(f"\n🔍 FINAL REGULARIZATION CHECK...")
        validation_results = validate_regularization_implementation()
        
        if validation_results['overall_status'] == 'PASSED':
            print(f"\n🎯 SUCCESS: All regularization fixes have been properly implemented!")
            print(f"The model should now show improved generalization with reduced overfitting.")
            print(f"\nNext Steps:")
            print(f"1. Test the regularized model on validation data")
            print(f"2. Compare performance gap with previous overfitted model")
            print(f"3. If validation performance is positive, proceed to production testing")
            return 0
        else:
            print(f"\n⚠️ WARNING: Some regularization issues detected during final check.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user.")
        print(f"Progress has been saved. Use same command to resume.")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
