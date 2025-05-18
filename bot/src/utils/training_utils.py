"""
Training utilities for PPO-LSTM model with walk-forward optimization.

This module provides functions and configurations for training a PPO-LSTM model
using walk-forward optimization. It includes:
- Model architecture and hyperparameter configurations 
- Training state management
- Model evaluation and comparison functions
- Walk-forward training loop implementation

The implementation is optimized for sparse reward scenarios in financial trading,
with careful management of exploration strategies.
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

# Enhanced model architecture configuration
POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 512,         # Increased from 256 for better temporal patterns
    "n_lstm_layers": 2,              # Keep 2 layers
    "shared_lstm": False,            # Separate LSTM architectures
    "enable_critic_lstm": True,      # Enable LSTM for value estimation
    "net_arch": {
        "pi": [256, 128, 64],       # Simple yet effective feedforward structure
        "vf": [256, 128, 64]        # Mirror policy network structure
    },
    "activation_fn": th.nn.Mish,     # Better activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-6         # Maintain current regularization
    }
}

# Walk-forward optimization configuration
TRAINING_PASSES = 30    # Number of passes through each window's data during training

def calculate_timesteps(window_size: int) -> int:
    """Calculate training timesteps for current window."""
    return window_size * TRAINING_PASSES

def market_regime_lr(initial_lr: float = 2.5e-4, max_lr: float = 5e-4, regime_window: int = 480) -> callable:
    """Create market regime-based cyclic learning rate schedule.
    
    Args:
        initial_lr: Base learning rate
        max_lr: Maximum learning rate during cycle
        regime_window: Window size for regime cycles (480 = 5 days of 15-min bars)
    
    Returns:
        Learning rate schedule function
    """
    def schedule(progress_remaining: float) -> float:
        # Slower cycles based on market regime changes
        cycle = np.floor(1 + progress_remaining * (regime_window/1024))  # Using n_steps=1024
        x = np.abs(progress_remaining * (regime_window/1024) - cycle)
        lr = initial_lr + (max_lr - initial_lr) * max(0, (1 - x))
        return lr
    return schedule

# Enhanced training hyperparameters
INITIAL_MODEL_KWARGS = {
    "learning_rate": market_regime_lr(),  # Market regime-based cyclic learning rate
    "n_steps": 1024,                     # Increased from 512 for better temporal context
    "batch_size": 256,                   # Maintain batch size
    "gamma": 0.99,                       # Keep high gamma for sparse rewards
    "gae_lambda": 0.95,                  # Standard GAE lambda
    "clip_range": 0.2,                   # Standard PPO clip
    "clip_range_vf": 0.2,                # Match policy clipping
    "ent_coef": 0.1,                     # Increased from 0.05 for better exploration
    "vf_coef": 1.0,                      # Maintain value importance
    "max_grad_norm": 0.5,                # Keep conservative gradient clipping
    "n_epochs": 12                       # Maintain training epochs
}

# Enhanced adaptation hyperparameters
ADAPTATION_MODEL_KWARGS = {
    "learning_rate": market_regime_lr(initial_lr=1.5e-4, max_lr=3e-4),  # Lower range for fine-tuning
    "batch_size": 256,            # Maintain batch size
    "n_steps": 1024,             # Match initial n_steps
    "n_epochs": 10,              # Maintain epochs for regime learning
    "clip_range": 0.2,           # Standard PPO clip
    "ent_coef": 0.05,            # Reduced entropy for exploitation
    "gae_lambda": 0.95,          # Standard lambda
    "max_grad_norm": 0.5,        # Keep gradient clipping
    "gamma": 0.99,               # High gamma for sparse rewards
    "clip_range_vf": 0.2,        # Match policy clipping
    "vf_coef": 1.0              # Maintain value importance
}

# Initialize with adaptation-ready parameters
MODEL_KWARGS = {
    **INITIAL_MODEL_KWARGS,
    "ent_coef": 0.05            # Start with higher entropy for better exploration
}

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

def evaluate_model_on_dataset(model_path: str, data: pd.DataFrame, args) -> Dict[str, Any]:
    """Evaluate a model on a given dataset."""
    if not os.path.exists(model_path):
        return None
        
    try:
        from sb3_contrib.ppo_recurrent import RecurrentPPO
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

            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states if 'lstm_states' in locals() else None,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            performance = env.metrics.get_performance_summary()
        finally:
            stop_progress_indicator()
        
        # Extract core metrics
        score = 0.0
        returns = performance['return_pct'] / 100
        max_dd = max(performance['max_drawdown_pct'], performance['max_equity_drawdown_pct']) / 100
        profit_factor = performance['profit_factor']
        win_rate = performance['win_rate'] / 100
        
        # Calculate score based on core metrics
        risk_adj_return = returns / (max_dd + 0.05)
        score += risk_adj_return * 0.40  # Increased weight for risk-adjusted returns
        score += returns * 0.30  # Increased weight for raw returns
        
        # Add profit factor component
        pf_score = 0.0
        if profit_factor > 1.0:
            pf_score = min((profit_factor - 1.0) / 2.0, 1.0)  # Cap at profit factor of 3.0
        score += pf_score * 0.20
        
        # Add win rate component
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

def create_env_config(args, spread_variation: float = 0.0, slippage_range: float = 0.0) -> TradingConfig:
    """Create TradingConfig from args with optional spread and slippage settings."""
    return TradingConfig(
        initial_balance=args.initial_balance,
        balance_per_lot=args.balance_per_lot,
        point_value=args.point_value,
        min_lots=args.min_lots,
        max_lots=args.max_lots,
        contract_size=args.contract_size,
        window_size=args.window_size,
        spread_variation=spread_variation,
        slippage_range=slippage_range
    )

def train_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """
    Train a model using walk-forward optimization with continuous learning.
    
    Implements enhanced walk-forward optimization with:
    - Dynamic timesteps based on window size
    - Single unified training configuration
    - Warm start capability with auto learning rate adjustment
    - Comprehensive validation and historical evaluation
    - State saving for resumable training
    """
    total_periods = len(data)
    training_passes = args.train_passes if hasattr(args, 'train_passes') else TRAINING_PASSES

    total_iterations = (total_periods - initial_window) // step_size + 1
    train_window_size = initial_window - int(initial_window * args.validation_size)
    timesteps_per_iteration = calculate_timesteps(train_window_size)
    total_timesteps = timesteps_per_iteration * total_iterations
    
    print(f"Training Configuration:")
    print(f"Training passes per window: {training_passes}")
    print(f"Timesteps per iteration: {timesteps_per_iteration:,d}")
    print(f"Total training iterations: {total_iterations}")
    print(f"Total training timesteps: {total_timesteps:,d}")

    val_size = int(initial_window * args.validation_size)
    train_size = initial_window - val_size
    print(f"\nWindow Configuration (15-min bars):")
    print(f"Training Size: {train_size} bars")
    print(f"Validation Size: {val_size} bars")
    print(f"Step Size: {step_size} bars")
    
    results_dir = f"../results/{args.seed}"
    state_path = os.path.join(results_dir, "training_state.json")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    plots_dir = os.path.join(results_dir, "plots")
    iterations_dir = os.path.join(results_dir, "iterations")
    
    for directory in [checkpoints_dir, plots_dir, iterations_dir]:
        os.makedirs(directory, exist_ok=True)

    training_start, initial_model_path, state = load_training_state(state_path)
    model = None
    env_config = create_env_config(args)
    
    if training_start > 0:  # Resuming training
        last_completed_iter = state.get('completed_iterations', -1)
        interrupted_iter = state.get('interrupted_iteration')
        
        stale_state = False
        try:
            with open(state_path, 'r') as f:
                current_state = json.load(f)
                if 'interrupted_iteration' in current_state:
                    interrupt_time = datetime.fromisoformat(current_state['timestamp'])
                    if (datetime.now() - interrupt_time).total_seconds() > 86400:
                        stale_state = True
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        if stale_state:
            print("\nWarning: Found stale interrupt state (>24h old)")
            print("Starting fresh training to avoid inconsistent state")
            current_state = {
                'training_start': 0,
                'timestamp': datetime.now().isoformat(),
                'iteration_times': [],
                'avg_iteration_time': 0.0,
                'total_iterations': total_iterations
            }
            with open(state_path, 'w') as f:
                json.dump(current_state, f)
            print("- State file reset successfully")
            print("- Starting fresh training...")
            training_start = 0
            model = None
            interrupted_iter = None

        if interrupted_iter is not None and training_start > 0:
            print(f"\nResuming from interrupted iteration {interrupted_iter}")
            print(f"Last completed iteration: {last_completed_iter}")
            training_start = interrupted_iter * step_size
            train_start = training_start
            print(f"Resuming training at index: {train_start}")
            
            try:
                with open(state_path, 'r') as f:
                    current_state = json.load(f)
                if 'interrupted_iteration' in current_state:
                    interrupted = current_state['interrupted_iteration']
                    del current_state['interrupted_iteration']
                    print(f"\nState Transition:")
                    print(f"- Cleared interrupted state (was iteration {interrupted})")
                    print(f"- Resuming training from iteration {interrupted}")
                    with open(state_path, 'w') as f:
                        json.dump(current_state, f)
                    print("- State file updated successfully")
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        else:
            print(f"\nResuming from last completed iteration: {last_completed_iter}")
            train_start = training_start
            print(f"Resuming training at index: {train_start}")
        
        val_start = train_start + 2880
        train_data = data.iloc[train_start:val_start].copy()
        train_env = Monitor(TradingEnv(train_data, predict_mode=False, config=env_config))
        
        model_paths = [
            os.path.join(iterations_dir, f"iteration_{last_completed_iter}/best_model.zip"),
            os.path.join(checkpoints_dir, f"checkpoint_iter_{last_completed_iter}.zip"),
            os.path.join(checkpoints_dir, f"checkpoint_interrupt_iter_{last_completed_iter}.zip")
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Resuming: Loading model from {model_path}")
                try:
                    model = RecurrentPPO.load(
                        model_path,
                        env=train_env,
                        device=args.device,
                        custom_objects={"learning_rate": ADAPTATION_MODEL_KWARGS["learning_rate"]}
                    )
                    model.set_env(train_env)
                    print(f"Successfully loaded model from {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load model from {model_path}: {e}")
                    continue
                    
        if model is None:
            print(f"\nWarning: Could not find any valid model for iteration {last_completed_iter}")
            print("\nStarting fresh training...")
            training_start = 0
    else:
        print("Starting new training.")
        training_start = 0
        model = None
    
    STEP_SIZE = 240
    if step_size != STEP_SIZE:
        print(f"Warning: Adjusting step size from {step_size} to {STEP_SIZE}")
        step_size = STEP_SIZE

    try:
        while training_start + initial_window <= total_periods:
            iteration = training_start // step_size
            iteration_start_time = time.time()

            _, _, state = load_training_state(state_path)
            
            if state.get('avg_iteration_time'):
                remaining_iterations = total_iterations - state.get('completed_iterations', 0)
                estimated_time = remaining_iterations * state['avg_iteration_time']
                print(f"\nEstimated time remaining: {format_time_remaining(estimated_time)}")
                print(f"Average iteration time: {state['avg_iteration_time']/60:.1f} minutes")
                print(f"Completed iterations: {state.get('completed_iterations', 0) - 1}/{total_iterations}")
            
            train_start = training_start
            val_start = train_start + 2880  # 6 weeks of 15min bars
            val_end = val_start + 960       # 2 weeks validation
            
            train_data = data.iloc[train_start:val_start].copy()
            val_data = data.iloc[val_start:val_end].copy()
            
            train_data.index = data.index[train_start:val_start]
            val_data.index = data.index[val_start:val_end]
            
            print(f"\n=== Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
            print(f"=== Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
            print(f"=== Walk-forward Iteration: {iteration}/{total_iterations} ===")
        
            env_config = create_env_config(args)
        
            train_env = Monitor(TradingEnv(train_data, predict_mode=False, config=env_config))
            val_env = Monitor(TradingEnv(val_data, predict_mode=False, config=env_config))

            train_window_size = val_start - train_start
            period_timesteps = calculate_timesteps(train_window_size)
            
            if model is None:
                print("\nPerforming initial training...")
                
                if args.warm_start:
                    last_completed = state.get('completed_iterations', iteration - 1)
                    model_paths = []
                    
                    if last_completed >= 0:
                        model_paths.extend([
                            os.path.join(iterations_dir, f"iteration_{last_completed}/best_model.zip"),
                            os.path.join(checkpoints_dir, f"checkpoint_iter_{last_completed}.zip"),
                            os.path.join(checkpoints_dir, f"checkpoint_interrupt_iter_{last_completed}.zip")
                        ])
                    
                    if args.initial_model:
                        model_paths.append(args.initial_model)
                    
                    for model_path in model_paths:
                        if os.path.exists(model_path):
                            try:
                                print(f"Warm starting from: {model_path}")
                                model = RecurrentPPO.load(
                                    model_path,
                                    env=train_env,
                                    device=args.device,
                                    custom_objects={"learning_rate": ADAPTATION_MODEL_KWARGS["learning_rate"]}
                                )
                                model.set_env(train_env)
                                break
                            except Exception as e:
                                print(f"Failed to load model from {model_path}: {e}")
                                continue
                    
                    if model is None:
                        print("\nCould not load any existing models. Starting fresh training...")
                
            iteration_dir = os.path.join(iterations_dir, f"iteration_{iteration}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            evaluator = ModelEvaluator(
                save_path=f"../results/{args.seed}",
                device=args.device,
                verbose=args.verbose
            )
            
            callbacks = [
                CustomEpsilonCallback(
                    start_eps=1.0 if model is None else 0.25,
                    end_eps=0.1 if model is None else 0.01,
                    decay_timesteps=int(period_timesteps * 0.7),
                    iteration=iteration
                ),
                ValidationCallback(
                    eval_env=val_env,
                    eval_freq=args.eval_freq,
                    model_save_path=iteration_dir,
                    deterministic=True,
                    verbose=args.verbose,
                    iteration=iteration
                )
            ]

            if model is None:
                print("Creating new model with initial hyperparameters...")
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    train_env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=0,
                    device=args.device,
                    seed=args.seed,
                    **INITIAL_MODEL_KWARGS
                )
                
            is_new_model = model is None
            
            if args.warm_start and not is_new_model:
                model.learning_rate = ADAPTATION_MODEL_KWARGS['learning_rate']
            
            print(f"\nTraining for {period_timesteps} timesteps...")
            model.learn(
                total_timesteps=period_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=is_new_model
            )

            current_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_iter_{iteration}.zip")
            model.save(current_checkpoint_path)
            print(f"Saved iteration checkpoint: {current_checkpoint_path}")

            best_model_path = os.path.join(iteration_dir, "best_model.zip")
            print(f"\nBest model from this iteration saved at: {best_model_path}")
            print("This model will be used for warm starting the next iteration if warm_start=True")
                
            iteration_time = time.time() - iteration_start_time
            
            save_training_state(
                state_path, 
                training_start + step_size,
                best_model_path,
                iteration_time=iteration_time,
                total_iterations=total_iterations,
                step_size=step_size,
                is_completed=True
            )
            
            print(f"\nCompleted iteration {iteration}. Saved best model: {best_model_path}")
            print(f"Time taken: {iteration_time/60:.1f} minutes")
            
            print("\nLoading best validation model for historical evaluation...")
            best_model_path = os.path.join(iteration_dir, "best_model.zip")
            if os.path.exists(best_model_path):
                try:
                    best_model = RecurrentPPO.load(
                        best_model_path,
                        env=train_env,
                        device=args.device
                    )
                    print(f"Successfully loaded best validation model from: {best_model_path}")
                    
                    print("\nPerforming evaluation on full historical dataset...")
                    eval_results = evaluator.select_best_model(
                        model=best_model,
                        val_env=val_env,
                        full_data=data,
                        config=env_config,
                        iteration=iteration,
                        is_final_eval=True
                    )
                except Exception as e:
                    print(f"\nFailed to load best validation model: {e}")
                    print("Falling back to current model for historical evaluation...")
                    eval_results = evaluator.select_best_model(
                        model=model,
                        val_env=val_env,
                        full_data=data,
                        config=env_config,
                        iteration=iteration,
                        is_final_eval=True
                    )
            else:
                print("\nNo best validation model found, using current model for historical evaluation...")
                eval_results = evaluator.select_best_model(
                    model=model,
                    val_env=val_env,
                    full_data=data,
                    config=env_config,
                    iteration=iteration,
                    is_final_eval=True
                )
            
            if 'historical' in eval_results:
                evaluator.print_evaluation_results(
                    eval_results['historical'],
                    phase="Historical",
                    timestep=model.num_timesteps
                )
            
            training_start += step_size
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved - use same command to resume.")
        if model:
            interrupt_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_interrupt_iter_{iteration}.zip")
            model.save(interrupt_checkpoint_path)
            print(f"Saved interrupt checkpoint: {interrupt_checkpoint_path}")
            
            save_interrupt_state(state_path, iteration, interrupt_checkpoint_path)
            print(f"Training state saved. Will resume from iteration {iteration}")
            
        return model

    print(f"\nWalk-forward optimization completed.")
    final_model_path = os.path.join(f"../results/{args.seed}", "final_evolved_model.zip")
    if model:
        model.save(final_model_path)
        print(f"Final evolved model saved to: {final_model_path}")

    return model
