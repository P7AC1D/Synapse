"""
Recurrent PPO-LSTM training script for trading model with walk-forward optimization.
Implements continuous sequential learning with LSTM state management and dynamic exploration.
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

from utils.training_utils import save_training_state, load_training_state, train_walk_forward

def main():
    parser = argparse.ArgumentParser(description='Train a Recurrent PPO-LSTM model for trading with walk-forward optimization')    
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input dataset CSV file')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to use for training (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Training window parameters
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial balance for trading')
    parser.add_argument('--initial_window', type=int, default=5000,
                      help='Initial training window size in bars')
    parser.add_argument('--test_window', type=int, default=5000,
                      help='Test window size in bars (default: same as training window)')
    parser.add_argument('--train_split', type=float, default=0.7,
                      help='Fraction of window to use for training vs validation (default: 0.7)')
    parser.add_argument('--step_size', type=int, default=1000,
                      help='Walk-forward step size in bars (should be large enough for feature stability)')
    parser.add_argument('--balance_per_lot', type=float, default=500.0,
                      help='Account balance required per 0.01 lot')
    parser.add_argument('--random_start', action='store_true',
                      help='Start training from random positions in the dataset')
    
    # Warm start parameters
    parser.add_argument('--warm_start', action='store_true',
                      help='Use best model from previous iteration as starting point')
    parser.add_argument('--initial_model', type=str,
                      help='Path to initial model for first iteration warm start')
    
    parser.add_argument('--total_timesteps', type=int, default=100000,
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=10000,
                      help='Evaluation frequency in timesteps')
    
    # Trading environment parameters
    parser.add_argument('--point_value', type=float, default=0.01,
                      help='Value of one price point movement (default: 0.01)')
    parser.add_argument('--min_lots', type=float, default=0.01,
                      help='Minimum lot size (default: 0.01)')
    parser.add_argument('--max_lots', type=float, default=200.0,
                      help='Maximum lot size (default: 200.0)')
    parser.add_argument('--contract_size', type=float, default=100.0,
                      help='Standard contract size (default: 100.0)')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of past timesteps for market features in observation (default: 50)')
    
    args = parser.parse_args()
    
    os.makedirs(f"../results/{args.seed}", exist_ok=True)
    os.makedirs(f"../results/{args.seed}/checkpoints", exist_ok=True)
    
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if args.device == 'cuda':
        th.cuda.manual_seed(args.seed)
        print("Note: Training PPO-LSTM on GPU requires sufficient memory for sequence processing.")
    
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
        
    try:
        model = train_walk_forward(data, initial_window, step_size, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        return
    
    print("\nWalk-forward optimization completed.")
    # The returned model is the continuously evolved one.
    # The overall best performing model on the full dataset is saved as best_model.zip by train_walk_forward.
    final_evolved_model_path = f"../results/{args.seed}/model_final_evolved.zip"
    if model: # model could be None if training was interrupted very early
        model.save(final_evolved_model_path)
        print(f"Final state of continuously evolved model saved at: {final_evolved_model_path}")
    else:
        print("No model was returned from training, possibly due to early interruption.")

if __name__ == "__main__":
    main()
