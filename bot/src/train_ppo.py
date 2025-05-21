"""
Recurrent PPO-GRU training script for trading model with walk-forward optimization.
Implements sequential learning with standard PPO exploration and GRU state management.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
import torch as th
from datetime import datetime

from utils.training_utils import (
    save_training_state, load_training_state, train_walk_forward, 
    calculate_timesteps, TRAINING_PASSES
)

def main():
    parser = argparse.ArgumentParser(description='Train a PPO-GRU model for trading with walk-forward optimization')    
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input dataset CSV file')
    
    # Hardware/system arguments
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to use for training (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--verbose', type=int, default=1,
                      help='Verbosity level (0: minimal, 1: normal, 2: debug)')
    
    # Window configuration
    parser.add_argument('--initial_window', type=int, default=1920,
                      help='Initial window size in bars (20 days of 15-min data)')
    parser.add_argument('--validation_size', type=float, default=0.25,
                      help='Fraction of window for validation (5 days)')
    parser.add_argument('--step_size', type=int, default=96,
                      help='Walk-forward step size in bars (1 trading day)')
    parser.add_argument('--eval_freq', type=int, default=10000,
                      help='Evaluation frequency in timesteps')
    
    # Trading environment configuration
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance')
    parser.add_argument('--balance_per_lot', type=float, default=500.0,
                      help='Account balance required per 0.01 lot')
    
    # Symbol-specific parameters
    parser.add_argument('--point_value', type=float, default=0.01,
                      help='Value of one price point movement')
    parser.add_argument('--min_lots', type=float, default=0.01,
                      help='Minimum trading lot size')
    parser.add_argument('--max_lots', type=float, default=200.0,
                      help='Maximum trading lot size')
    parser.add_argument('--contract_size', type=float, default=100.0,
                      help='Standard contract size')
    
    # Training options
    parser.add_argument('--warm_start', action='store_true',
                      help='Enable continuous learning and load initial model if specified')
    parser.add_argument('--initial_model', type=str,
                      help='Optional path to pre-trained model for warm start')
    
    args = parser.parse_args()
    
    os.makedirs(f"../results/{args.seed}", exist_ok=True)
    os.makedirs(f"../results/{args.seed}/checkpoints", exist_ok=True)
    
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if args.device == 'cuda':
        th.cuda.manual_seed(args.seed)
        print("Note: Training PPO-GRU on GPU requires sufficient memory for sequence processing.")
    
    data = pd.read_csv(args.data_path)
    data.set_index('time', inplace=True)
    print(f"Dataset shape: {data.shape}, from {data.index[0]} to {data.index[-1]}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Display window configuration (15-min bars)
    bars_per_day = 96  # 24 hours * 4 bars per hour for 15-min data
    val_size = int(args.initial_window * args.validation_size)
    train_size = args.initial_window - val_size
    
    print(f"\nWindow Configuration (15-min bars):")
    print(f"Training Window: {train_size} bars ({train_size/bars_per_day:.1f} days)")
    print(f"Validation Window: {val_size} bars ({val_size/bars_per_day:.1f} days)")
    print(f"Step Size: {args.step_size} bars ({args.step_size/bars_per_day:.1f} days)\n")

    try:
        model = train_walk_forward(
            data=data,
            initial_window=args.initial_window,
            step_size=args.step_size,
            args=args
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        return
    
    print("\nWalk-forward optimization completed.")
    final_evolved_model_path = f"../results/{args.seed}/final_evolved_model.zip"
    if model:
        model.save(final_evolved_model_path)
        print(f"Final evolved model saved to: {final_evolved_model_path}")
    else:
        print("No model was returned from training, possibly due to early interruption.")

if __name__ == "__main__":
    main()
