"""PPO training script for trading model."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch as th
from datetime import datetime
import pprint
import copy

from utils.training_utils import save_training_state, load_training_state, train_walk_forward
from utils.ppo_params import get_ppo_params
from utils.wfo_params import get_wfo_params

def log_parameters_to_file(args, ppo_hyperparams, wfo_params=None, file_path=None):
    """
    Log all training parameters to a file and print them to console.
    
    Args:
        args: Command line arguments
        ppo_hyperparams: PPO hyperparameters
        wfo_params: Walk-forward optimization parameters
        file_path: Path to save the parameters log (optional)
    """
    # Create a dictionary with all parameters
    all_params = {
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "dataset": os.path.basename(args.data_path),
            "device": args.device,
            "seed": args.seed,
            "initial_balance": args.initial_balance,
            "balance_per_lot": args.balance_per_lot,
            "point_value": args.point_value,
            "min_lots": args.min_lots,
            "max_lots": args.max_lots,
            "contract_size": args.contract_size,
            "random_start": args.random_start
        },
        "walk_forward_optimization": {
            "initial_window": args.initial_window,
            "step_size": args.step_size,
            "validation_size": args.validation_size,
            "use_research_params": args.use_research_params,
        },
        "ppo_configuration": {
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "eval_freq": args.eval_freq,
            "market_type": args.market_type,
            "financial_adjustments": args.financial_adjustments
        }
    }
    
    # Add detailed PPO hyperparameters
    ppo_detailed = copy.deepcopy(ppo_hyperparams)
    if "policy_kwargs" in ppo_detailed:
        # Convert network architecture to string representation
        if "net_arch" in ppo_detailed["policy_kwargs"]:
            net_arch = ppo_detailed["policy_kwargs"]["net_arch"]
            ppo_detailed["policy_kwargs"]["net_arch"] = str(net_arch)
        
        # Convert activation function to string representation
        if "activation_fn" in ppo_detailed["policy_kwargs"]:
            activation = ppo_detailed["policy_kwargs"]["activation_fn"]
            ppo_detailed["policy_kwargs"]["activation_fn"] = activation.__name__ if hasattr(activation, "__name__") else str(activation)
    
    if "policy" in ppo_detailed:
        ppo_detailed["policy"] = ppo_detailed["policy"].__name__ if hasattr(ppo_detailed["policy"], "__name__") else str(ppo_detailed["policy"])
    
    all_params["ppo_hyperparameters"] = ppo_detailed
    
    # Add WFO research parameters if used
    if wfo_params is not None:
        all_params["research_wfo_parameters"] = wfo_params
    
    # Print parameters to console
    print("\n" + "="*80)
    print("TRAINING PARAMETERS SUMMARY".center(80))
    print("="*80)
    
    print("\nEnvironment Configuration:")
    print("-" * 50)
    for key, value in all_params["environment"].items():
        print(f"{key:20s}: {value}")
    
    print("\nWalk-Forward Optimization:")
    print("-" * 50)
    for key, value in all_params["walk_forward_optimization"].items():
        print(f"{key:20s}: {value}")
    
    print("\nPPO Configuration:")
    print("-" * 50)
    for key, value in all_params["ppo_configuration"].items():
        print(f"{key:20s}: {value}")
    
    print("\nPPO Hyperparameters:")
    print("-" * 50)
    for key, value in all_params["ppo_hyperparameters"].items():
        if key not in ["policy", "policy_kwargs", "device", "verbose"]:
            print(f"{key:20s}: {value}")
    
    if "policy_kwargs" in all_params["ppo_hyperparameters"]:
        print("\nNetwork Architecture:")
        print("-" * 50)
        for key, value in all_params["ppo_hyperparameters"]["policy_kwargs"].items():
            print(f"{key:20s}: {value}")
    
    if wfo_params is not None:
        print("\nResearch-based WFO Parameters:")
        print("-" * 50)
        for key, value in all_params["research_wfo_parameters"].items():
            print(f"{key:20s}: {value}")
    
    print("\n" + "="*80 + "\n")
    
    # Save parameters to file if path is provided
    if file_path is None:
        # Create default path in results directory
        results_dir = f"../results/{args.seed}"
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, "training_parameters.json")
    
    with open(file_path, 'w') as f:
        json.dump(all_params, f, indent=4)
    
    print(f"Parameters saved to: {file_path}")
    
    return all_params

def main():
    parser = argparse.ArgumentParser(description='Train a PPO model for trading')    
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input dataset CSV file')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to use for training (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial balance for trading')
    parser.add_argument('--initial_window', type=int, default=10080,
                      help='Initial training window size in bars (default: 10080, ~21 weeks for 15min gold data)')
    parser.add_argument('--validation_size', type=float, default=0.2,
                      help='Fraction of window to use for validation (default: 0.2)')
    parser.add_argument('--step_size', type=int, default=2016,
                      help='Walk-forward step size in bars (default: 2016, ~3 weeks for 15min gold data)')
    parser.add_argument('--balance_per_lot', type=float, default=500.0,
                      help='Account balance required per 0.01 lot')
    parser.add_argument('--random_start', action='store_true',
                      help='Start training from random positions in the dataset')
    
    # PPO-specific hyperparameters
    parser.add_argument('--total_timesteps', type=int, default=500000,
                      help='Total timesteps for training (default: 500000)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=5e-5,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=50000,
                      help='Evaluation frequency in timesteps')
    
    # Market-specific adjustments
    parser.add_argument('--market_type', type=str, choices=['short_term', 'medium_term', 'long_term'], 
                      default='short_term', 
                      help='Market type for trading horizon (determines appropriate hyperparameters)')
    parser.add_argument('--financial_adjustments', action='store_true', 
                      help='Apply financial market specific adjustments to hyperparameters')
    
    # WFO-specific parameters
    parser.add_argument('--use_research_params', action='store_true',
                      help='Use research-based WFO parameters for window/step size and timesteps')
    
    # Logging parameters
    parser.add_argument('--params_log_path', type=str, default=None,
                      help='Path to save the parameters log (default: ../results/{seed}/training_parameters.json)')
    
    # Trading environment parameters
    parser.add_argument('--point_value', type=float, default=0.01,
                      help='Value of one price point movement (default: 0.01)')
    parser.add_argument('--min_lots', type=float, default=0.01,
                      help='Minimum lot size (default: 0.01)')
    parser.add_argument('--max_lots', type=float, default=200.0,
                      help='Maximum lot size (default: 200.0)')
    parser.add_argument('--contract_size', type=float, default=100.0,
                      help='Standard contract size (default: 100.0)')
    
    args = parser.parse_args()
    
    os.makedirs(f"../results/{args.seed}", exist_ok=True)
    os.makedirs(f"../results/{args.seed}/checkpoints", exist_ok=True)
    
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if args.device == 'cuda':
        th.cuda.manual_seed(args.seed)
        print("Note: Training PPO with MLP policy on GPU may be slower than CPU. Consider using CPU for this model type.")
    
    data = pd.read_csv(args.data_path)
    data.set_index('time', inplace=True)
    print(f"Dataset shape: {data.shape}, from {data.index[0]} to {data.index[-1]}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Use research-based WFO parameters if requested
    wfo_params = None
    if args.use_research_params:
        data_length = len(data)
        wfo_params = get_wfo_params(args, data_length)
        
        # Update parameters with research-based values
        args.initial_window = wfo_params['window_size']
        args.step_size = wfo_params['step_size']
        args.total_timesteps = wfo_params['total_timesteps']
        args.validation_size = wfo_params['validation_split']
        
        print("\n=== Using Research-Based WFO Parameters ===")
        print(f"Window Size: {args.initial_window} bars (~{wfo_params['window_days']:.1f} days)")
        print(f"Step Size: {args.step_size} bars (~{wfo_params['step_days']:.1f} days)")
        print(f"Total Timesteps: {args.total_timesteps}")
        print(f"Window Coverage: {wfo_params['data_coverage']}")
    
    # Calculate window sizes - using bars directly
    initial_window = args.initial_window
    step_size = args.step_size
    
    # Display window configuration with approximate days
    bars_per_day = 96  # 96 bars/day for 15min data (24 hours * 4 bars per hour)
    val_size = int(initial_window * args.validation_size)
    train_size = initial_window - val_size
    
    # Calculate window coverage of total dataset
    total_bars = len(data)
    window_percent = (initial_window / total_bars) * 100
    
    print(f"\nWindow Configuration:")
    print(f"Dataset size: {total_bars} bars")
    print(f"Initial Window: {initial_window} bars (~{initial_window/bars_per_day:.1f} days, {window_percent:.1f}% of dataset)")
    print(f"Step Size: {step_size} bars (~{step_size/bars_per_day:.1f} days)")
    print(f"Training Window: {train_size} bars (~{train_size/bars_per_day:.1f} days)")
    print(f"Validation Window: {val_size} bars (~{val_size/bars_per_day:.1f} days)")
    print(f"Training/Validation Split: {(1-args.validation_size):.0%}/{args.validation_size:.0%}\n")
    
    # Get recommended PPO hyperparameters
    ppo_hyperparams = get_ppo_params(args)
    
    # Log all parameters to file and print them
    log_parameters_to_file(args, ppo_hyperparams, wfo_params, args.params_log_path)
    
    try:
        model = train_walk_forward(data, initial_window, step_size, args, ppo_hyperparams)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        return
    
    print("\nWalk-forward optimization completed.")
    print(f"Final model saved at: ../results/{args.seed}/model_final.zip")
    model.save(f"../results/{args.seed}/model_final.zip")

if __name__ == "__main__":
    main()
