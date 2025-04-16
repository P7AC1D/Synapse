"""PPO training script for trading model."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
import torch as th
from datetime import datetime

from utils.training_utils import save_training_state, load_training_state, train_walk_forward

def main():
    parser = argparse.ArgumentParser(description='Train a PPO-LSTM model for trading')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last saved state')
    
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name for saving the trained model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input dataset CSV file')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                      help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial balance for trading')
    parser.add_argument('--initial_window', type=int, default=2500,
                      help='Initial training window size in bars')
    parser.add_argument('--validation_size', type=float, default=0.2,
                      help='Fraction of window to use for validation (default: 0.2)')
    parser.add_argument('--step_size', type=int, default=500,
                      help='Walk-forward step size in bars')
    parser.add_argument('--balance_per_lot', type=float, default=500.0,
                      help='Account balance required per 0.01 lot')
    parser.add_argument('--random_start', action='store_true',
                      help='Start training from random positions in the dataset')
    
    parser.add_argument('--total_timesteps', type=int, default=100000,
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=10000,
                      help='Evaluation frequency in timesteps')
    
    args = parser.parse_args()
    
    os.makedirs(f"../results/{args.seed}", exist_ok=True)
    os.makedirs(f"../results/{args.seed}/checkpoints", exist_ok=True)
    
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if args.device == 'cuda':
        th.cuda.manual_seed(args.seed)
    
    data = pd.read_csv(args.data_path)
    data.set_index('time', inplace=True)
    print(f"Dataset shape: {data.shape}, from {data.index[0]} to {data.index[-1]}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Calculate window sizes - using bars directly
    initial_window = args.initial_window
    
    # Calculate step size as 10% of window size if using default
    step_size = args.step_size
    if step_size == 672:  # If using default value
        step_size = max(int(initial_window * 0.1), step_size)
    
    # Display window configuration with approximate days
    bars_per_day = 24 * 4  # For displaying approximate days
    val_size = int(initial_window * args.validation_size)
    train_size = initial_window - val_size
    
    print(f"\nWindow Configuration:")
    print(f"Initial Window: {initial_window} bars (~{initial_window/bars_per_day:.1f} days)")
    print(f"Step Size: {step_size} bars (~{step_size/bars_per_day:.1f} days)")
    print(f"Training Window: {train_size} bars (~{train_size/bars_per_day:.1f} days)")
    print(f"Validation Window: {val_size} bars (~{val_size/bars_per_day:.1f} days)")
    print(f"Training/Validation Split: {(1-args.validation_size):.0%}/{args.validation_size:.0%}\n")
    
    if args.resume:
        state_path = f"../results/{args.seed}/training_state.json"
        if os.path.exists(state_path):
            print("\nResuming walk-forward optimization...")
        else:
            print("\nNo previous state found. Starting new training...")
    else:
        print("\nStarting new walk-forward optimization...")
    
    try:
        model = train_walk_forward(data, initial_window, step_size, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        return
    
    print("\nWalk-forward optimization completed.")
    print(f"Final model saved at: ../results/{args.seed}/model_final.zip")
    model.save(f"../results/{args.seed}/model_final.zip")

if __name__ == "__main__":
    main()
