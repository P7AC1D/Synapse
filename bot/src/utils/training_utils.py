"""
Training utilities for PPO-GRU model with walk-forward optimization.

This module provides functions and configurations for training a PPO-GRU model
using walk-forward optimization. It includes:
- Model architecture and hyperparameter configurations 
- Training state management
- Model evaluation and comparison functions
- Walk-forward training loop implementation

The implementation is optimized for sparse reward scenarios in financial trading.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
import time
import threading
from utils.progress import show_progress_continuous, stop_progress_indicator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv, TradingConfig
import torch as th

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.eval_callback import ValidationCallback
from utils.model_evaluator import ModelEvaluator

# Training configuration
TRAINING_PASSES = 50    # Number of passes through each training window
WINDOW_SIZE = 30       # Number of past timesteps for market features

# Model architecture configuration with separate GRUs for policy and value
POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW,
    "gru_hidden_size": 512,          # Standard size for capturing intraday patterns
    "n_gru_layers": 2,               # Two layers for complex temporal dependencies
    "shared_gru": False,             # Separate GRU architectures
    "enable_critic_gru": True,       # Enable GRU for value estimation
    "net_arch": {
        "pi": [256, 128],           # Research-aligned policy network
        "vf": [256, 128]            # Matching value network
    },
    "activation_fn": th.nn.Mish,     # Modern activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-6         # Light regularization for stability
    }
}

# Standard model hyperparameters for consistent training
MODEL_KWARGS = {
    "learning_rate": 3e-4,           # Standard stable learning rate
    "n_steps": 256,                  # Keep faster training steps
    "batch_size": 128,               # Keep smaller batches
    "gamma": 0.99,                   # Standard discount
    "gae_lambda": 0.95,              # Standard GAE
    "clip_range": 0.2,               # Standard PPO clip
    "clip_range_vf": 0.2,            # Match policy clip
    "ent_coef": 0.15,                # Balanced exploration
    "vf_coef": 1.0,                  # Standard value weight
    "max_grad_norm": 0.5,            # Standard gradient clip
    "n_epochs": 4                    # Keep faster epochs
}

def calculate_timesteps(window_size: int, training_passes: int) -> int:
    """Calculate training timesteps for current window."""
    return window_size * training_passes

def format_time_remaining(seconds: float) -> str:
    """Convert seconds to days, hours, minutes format."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    
    return " ".join(parts)

def save_training_state(path: str, training_start: int, model_path: str,
                       iteration_time: float = None, total_iterations: int = None, 
                       step_size: int = None, is_completed: bool = True) -> None:
    """Save current training state to file."""
    try:
        with open(path, 'r') as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {
            'training_start': training_start,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'iteration_times': [],
            'avg_iteration_time': 0.0,
            'total_iterations': total_iterations
        }
    
    state['training_start'] = training_start
    state['model_path'] = model_path
    state['timestamp'] = datetime.now().isoformat()
    
    if iteration_time is not None:
        state['iteration_times'].append(iteration_time)
        # Keep only last 5 iterations for moving average
        state['iteration_times'] = state['iteration_times'][-5:]
        state['avg_iteration_time'] = sum(state['iteration_times']) / len(state['iteration_times'])
    
        if step_size is not None:
            current_iteration = training_start // step_size
            state['completed_iterations'] = current_iteration - 1 if current_iteration > 0 else 0
    
    if total_iterations is not None:
        state['total_iterations'] = total_iterations
    
    with open(path, 'w') as f:
        json.dump(state, f)

def save_interrupt_state(path: str, iteration: int, model_path: str) -> None:
    """Save state when training is interrupted."""
    try:
        with open(path, 'r') as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
        
    state['interrupted_iteration'] = iteration
    state['model_path'] = model_path
    state['timestamp'] = datetime.now().isoformat()
    state['completed_iterations'] = iteration - 1 if iteration > 0 else 0
    
    print(f"\nSaving interrupt state:")
    print(f"Last completed iteration: {state['completed_iterations']}")
    print(f"Interrupted at iteration: {iteration}")
    
    with open(path, 'w') as f:
        json.dump(state, f)

def load_training_state(path: str) -> Tuple[int, str, Dict[str, Any]]:
    """Load training state for resuming interrupted training."""
    if not os.path.exists(path):
        return 0, None, {}
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        return state['training_start'], state['model_path'], state
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, None, {}

def create_env_config(args) -> TradingConfig:
    """Create TradingConfig from args."""
    return TradingConfig(
        initial_balance=args.initial_balance,
        balance_per_lot=args.balance_per_lot,
        point_value=args.point_value,
        min_lots=args.min_lots,
        max_lots=args.max_lots,
        contract_size=args.contract_size,
        window_size=WINDOW_SIZE
    )

def evaluate_model_on_dataset(model_path: str, data: pd.DataFrame, args) -> Dict[str, Any]:
    """Evaluate a model on a given dataset."""
    if not os.path.exists(model_path):
        return None
        
    try:
        model = RecurrentPPO.load(model_path, device=args.device)
        
        env = TradingEnv(
            data=data,
            predict_mode=False,
            config=create_env_config(args)
        )

        progress_thread = threading.Thread(
            target=show_progress_continuous,
            args=("Evaluating model",)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            obs, _ = env.reset()
            done = False
            gru_states = None

            while not done:
                action, gru_states = model.predict(
                    obs,
                    state=gru_states,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            performance = env.metrics.get_performance_summary()
        finally:
            stop_progress_indicator()
        
        # Calculate metrics
        score = 0.0
        returns = performance['return_pct'] / 100
        max_dd = max(performance['max_drawdown_pct'], performance['max_equity_drawdown_pct']) / 100
        profit_factor = performance['profit_factor']
        win_rate = performance['win_rate'] / 100
        
        risk_adj_return = returns / (max_dd + 0.05)
        score += risk_adj_return * 0.40
        score += returns * 0.30
        
        if profit_factor > 1.0:
            score += min((profit_factor - 1.0) / 2.0, 1.0) * 0.20
        
        score += win_rate * 0.10
        
        return {
            'score': score,
            'returns': returns,
            'risk_adj_return': risk_adj_return,
            'drawdown': max_dd,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': performance['total_trades'],
            'metrics': performance
        }
        
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None

def train_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """Train a model using walk-forward optimization with continuous learning."""
    total_periods = len(data)
    training_passes = args.train_passes if hasattr(args, 'train_passes') else TRAINING_PASSES

    # Calculate training dimensions
    total_iterations = (total_periods - initial_window) // step_size + 1
    train_window_size = initial_window - int(initial_window * args.validation_size)
    timesteps_per_iteration = calculate_timesteps(train_window_size, training_passes)
    
    # Setup directories
    results_dir = f"../results/{args.seed}"
    state_path = os.path.join(results_dir, "training_state.json")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    plots_dir = os.path.join(results_dir, "plots")
    iterations_dir = os.path.join(results_dir, "iterations")
    
    for directory in [checkpoints_dir, plots_dir, iterations_dir]:
        os.makedirs(directory, exist_ok=True)

    # Load or initialize training state
    training_start, initial_model_path, state = load_training_state(state_path)
    model = None
    env_config = create_env_config(args)

    try:
        while training_start + initial_window <= total_periods:
            iteration = training_start // step_size
            iteration_start_time = time.time()
            
            # Calculate data windows
            train_start = training_start
            val_start = train_start + train_window_size
            val_end = val_start + int(initial_window * args.validation_size)
            
            # Prepare data
            train_data = data.iloc[train_start:val_start].copy()
            val_data = data.iloc[val_start:val_end].copy()
            
            # Create environments
            train_env = Monitor(TradingEnv(train_data, predict_mode=False, config=env_config))
            val_env = Monitor(TradingEnv(val_data, predict_mode=False, config=env_config))
            
            # Setup callbacks
            callbacks = [
                CustomEpsilonCallback(
                    start_eps=1.0 if model is None else 0.25,
                    end_eps=0.1 if model is None else 0.01,
                    decay_timesteps=int(timesteps_per_iteration * 0.7),
                    iteration=iteration
                ),
                ValidationCallback(
                    eval_env=val_env,
                    eval_freq=args.eval_freq,
                    model_save_path=os.path.join(iterations_dir, f"iteration_{iteration}"),
                    deterministic=True,
                    verbose=args.verbose,
                    iteration=iteration
                )
            ]

            # Create or update model
            if model is None:
                model = RecurrentPPO(
                    "MlpGruPolicy",
                    train_env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=0,
                    device=args.device,
                    seed=args.seed,
                    **MODEL_KWARGS
                )
            else:
                model.set_env(train_env)

            # Train model
            print(f"\nTraining for {timesteps_per_iteration} timesteps...")
            model.learn(
                total_timesteps=timesteps_per_iteration,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=model is None
            )

            # Save checkpoints
            current_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_iter_{iteration}.zip")
            model.save(current_checkpoint_path)

            iteration_time = time.time() - iteration_start_time
            save_training_state(
                state_path, 
                training_start + step_size,
                current_checkpoint_path,
                iteration_time=iteration_time,
                total_iterations=total_iterations,
                step_size=step_size
            )

            training_start += step_size
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved - use same command to resume.")
        if model:
            interrupt_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_interrupt_iter_{iteration}.zip")
            model.save(interrupt_checkpoint_path)
            save_interrupt_state(state_path, iteration, interrupt_checkpoint_path)

    # Save final model
    if model:
        final_model_path = os.path.join(results_dir, "final_evolved_model.zip")
        model.save(final_model_path)
        print(f"Final evolved model saved to: {final_model_path}")

    return model
