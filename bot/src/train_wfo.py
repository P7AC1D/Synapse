#!/usr/bin/env python3
"""
Walk-Forward Optimization Training Script

This script implements walk-forward optimization for training a DRL trading bot.
The training process uses sliding windows to simulate real-world conditions
and ensure robust model performance.

Usage:
    python train_wfo.py --seed 1007 --total-timesteps 40000
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import json

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.training_utils import (
    train_walk_forward
)

def load_and_prepare_data(args):
    """Load and prepare data for training."""
    print("📊 Loading data...")
    
    data_path = getattr(args, 'data_path', '../data/XAUUSDm_15min.csv')
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    data = pd.read_csv(data_path)
    
    # Handle both 'datetime' and 'time' columns for compatibility
    datetime_col = None
    if 'datetime' in data.columns:
        datetime_col = 'datetime'
    elif 'time' in data.columns:
        datetime_col = 'time'
        
    if datetime_col:
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data.set_index(datetime_col, inplace=True)
    
    print(f"✅ Data loaded: {len(data):,} samples")
    print(f"   Period: {data.index[0]} to {data.index[-1]}")
    print(f"   Features: {data.shape[1]} columns")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Training for DRL Trading Bot")
    
    # Core training parameters
    parser.add_argument('--seed', type=int, default=1007, help='Random seed for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=40000, dest='total_timesteps', help='Total timesteps for training')
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
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 WALK-FORWARD OPTIMIZATION TRAINING")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Create results directory
    results_dir = f"../results/{args.seed}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_and_prepare_data(args)
    if data is None:
        return 1
    
    # Run training
    try:
        start_time = datetime.now()
        
        # Train model
        final_model = train_walk_forward(
            data=data,
            initial_window=args.initial_window,
            step_size=args.step_size,
            args=args
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\n✅ TRAINING COMPLETED!")
        print(f"Training Duration: {training_duration}")
        print(f"Results Directory: {results_dir}")
        
        # Save training summary
        summary = {
            'training_type': 'walk_forward',
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
        
        summary_path = os.path.join(results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training Summary: {summary_path}")
        return 0
            
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
