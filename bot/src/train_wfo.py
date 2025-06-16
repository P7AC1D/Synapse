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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.training_utils import (
    train_walk_forward
)
from utils.adaptive_validation_utils import AdaptiveValidationManager
from configs.config import WFO_CONFIG, TRAINING_CONFIG

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
    
    # Calculate defaults from WFO_CONFIG (6 months = 17,280 periods, 1.5 months = 4,320 periods)
    periods_per_day = 96  # 24 hours * 4 (15-min periods)
    default_initial_window = WFO_CONFIG['training_window_days'] * periods_per_day  # 17,280
    default_step_size = WFO_CONFIG['step_forward_days'] * periods_per_day  # 4,320
    default_timesteps = TRAINING_CONFIG['total_timesteps']  # 52,000
    
    # Core training parameters (now using WFO_CONFIG)
    parser.add_argument('--seed', type=int, default=1007, help='Random seed for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=default_timesteps, dest='total_timesteps', 
                       help=f'Total timesteps for training (default: {default_timesteps:,} - optimized for WFO)')
    parser.add_argument('--initial-window', type=int, default=default_initial_window, dest='initial_window', 
                       help=f'Initial training window size (default: {default_initial_window:,} - 6 months)')
    parser.add_argument('--step-size', type=int, default=default_step_size, dest='step_size', 
                       help=f'Walk-forward step size (default: {default_step_size:,} - 1.5 months)')
      # Environment parameters
    parser.add_argument('--initial-balance', type=float, default=10000, dest='initial_balance', help='Initial account balance')
    parser.add_argument('--balance-per-lot', type=float, default=500, dest='balance_per_lot', help='Balance per lot sizing')
    parser.add_argument('--point-value', type=float, default=0.01, dest='point_value', help='Point value for XAUUSD')
    parser.add_argument('--min-lots', type=float, default=0.01, dest='min_lots', help='Minimum lot size')
    parser.add_argument('--max-lots', type=float, default=1.0, dest='max_lots', help='Maximum lot size')
    parser.add_argument('--contract-size', type=float, default=100000, dest='contract_size', help='Contract size')
    
    # Warm-start training parameters
    parser.add_argument('--warm-start-model', type=str, dest='warm_start_model_path',
                       help='Path to existing model (.zip) to continue training from')
    parser.add_argument('--warm-start-lr', type=float, dest='warm_start_learning_rate',
                       help='Learning rate for warm-start training (recommended: 2e-4 for refinement)')
    
    # Data and paths
    parser.add_argument('--data-path', type=str, default='../data/XAUUSDm_15min.csv', 
                       dest='data_path', help='Path to training data')
    parser.add_argument('--device', type=str, default='auto', help='Device for training (auto/cpu/cuda)')
    
    # Model selection strategy arguments
    parser.add_argument('--model-selection', type=str, default='ensemble_validation',
                       choices=['ensemble_validation', 'rolling_validation', 'risk_adjusted', 'legacy'],
                       dest='model_selection', help='Model selection strategy for walk-forward optimization')
    parser.add_argument('--disable-improved-selection', action='store_true', dest='disable_improved_selection',
                       help='Disable improved model selection and use legacy comparison')
    parser.add_argument('--no-legacy-warnings', action='store_true', dest='no_legacy_warnings',
                       help='Suppress warnings when using legacy model selection')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 WALK-FORWARD OPTIMIZATION TRAINING")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Model Selection: {args.model_selection}")
    
    # Display warm-start information if enabled
    if hasattr(args, 'warm_start_model_path') and args.warm_start_model_path:
        print(f"🔥 Warm-Start Mode: ENABLED")
        print(f"   Source Model: {args.warm_start_model_path}")
        if hasattr(args, 'warm_start_learning_rate') and args.warm_start_learning_rate:
            print(f"   Learning Rate: {args.warm_start_learning_rate:.2e} (reduced for refinement)")
        else:
            print(f"   Learning Rate: Using default (6e-4)")
    else:
        print(f"🆕 Training Mode: Fresh model training")
    
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Display WFO configuration
    print(f"\n📊 WFO Configuration (from WFO_CONFIG):")
    print(f"   Training Window: {args.initial_window:,} periods ({args.initial_window/96:.0f} days)")
    print(f"   Step Forward: {args.step_size:,} periods ({args.step_size/96:.0f} days)")
    print(f"   Total Timesteps: {args.total_timesteps:,} per window")
    print(f"   Overlap Ratio: {WFO_CONFIG['knowledge_retention']['overlap_ratio']*100:.0f}% (prevents forgetting)")
    print(f"{'='*60}")
    
    # Update configuration based on command line arguments
    if hasattr(args, 'disable_improved_selection') and args.disable_improved_selection:
        print("⚠️  Improved model selection disabled - using legacy comparison")
    elif hasattr(args, 'model_selection') and args.model_selection == 'legacy':
        print("⚠️  Legacy model selection explicitly requested")
    else:
        print(f"✅ Using improved model selection: {getattr(args, 'model_selection', 'ensemble_validation')}")
    
    # Create results directory using proper cross-platform path handling
    results_dir = os.path.join("..", "results", str(args.seed))
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_and_prepare_data(args)
    if data is None:
        return 1
    
    # Calculate dynamic WFO metadata from actual data
    total_periods = len(data)
    expected_iterations = (total_periods - args.initial_window) // args.step_size + 1
    total_dataset_days = len(data) // 96  # 96 periods per day for 15-min data
    
    # Update WFO display with calculated values
    print(f"\n📊 Dynamic WFO Metadata (calculated from actual data):")
    print(f"   Total Data Periods: {total_periods:,}")
    print(f"   Total Dataset Days: {total_dataset_days:,}")
    print(f"   Expected Iterations: {expected_iterations} (calculated)")
    print(f"   Data Utilization: {((expected_iterations * args.step_size + args.initial_window) / total_periods * 100):.1f}%")
    print(f"{'='*60}")
    
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
            'results_directory': results_dir,
            'warm_start_enabled': hasattr(args, 'warm_start_model_path') and args.warm_start_model_path is not None,
            'warm_start_model_path': getattr(args, 'warm_start_model_path', None),
            'warm_start_learning_rate': getattr(args, 'warm_start_learning_rate', None)
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
