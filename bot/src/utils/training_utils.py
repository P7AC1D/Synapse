"""
Training utilities for PPO-LSTM model with walk-forward optimization.

This module provides functions and configurations for training a PPO-LSTM model
using walk-forward optimization. It includes:
- Model architecture and hyperparameter configurations
- Training state management
- Model evaluation and comparison functions
- Walk-forward training loop implementation

The implementation is optimized for sparse reward scenarios in financial trading,
with careful management of temporal dependencies and exploration strategies.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List
import time
"""
Training utilities for PPO-LSTM model with walk-forward optimization.

This module provides functions and configurations for training a PPO-LSTM model
using walk-forward optimization. It includes:
- Model architecture and hyperparameter configurations
- Training state management
- Model evaluation and comparison functions
- Walk-forward training loop implementation

The implementation is optimized for sparse reward scenarios in financial trading,
with careful management of temporal dependencies and exploration strategies.
"""
import os
import json
import pandas as pd
import threading
from utils.progress import show_progress_continuous, stop_progress_indicator
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv
import torch as th

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.eval_callback import UnifiedEvalCallback

# Model architecture configuration
POLICY_KWARGS = {
    "optimizer_class": th.optim.AdamW,
    "lstm_hidden_size": 320,          # Increased for better temporal memory
    "n_lstm_layers": 3,               # Added layer for deeper temporal processing
    "shared_lstm": False,             # Keep separate architectures
    "enable_critic_lstm": True,       # LSTM for value estimation
    "net_arch": {
        "pi": [256, 128],             # Deeper policy network for pattern recognition
        "vf": [256, 128]              # Matching value network depth
    },
    "activation_fn": th.nn.Mish,      # Better activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-6          # Slightly reduced regularization
    }
}

# Training hyperparameters
MODEL_KWARGS = {
    "learning_rate": 3e-4,           # Further reduced for complex patterns
    "n_steps": 768,                  # Extended sequences for temporal patterns
    "batch_size": 384,               # Increased for stable learning with more features
    "gamma": 0.99,                   # Keep high gamma for sparse rewards
    "gae_lambda": 0.98,              # Keep high lambda for advantage estimation
    "clip_range": 0.15,              # Slightly increased for exploration
    "clip_range_vf": 0.15,           # Match policy clipping
    "ent_coef": 0.08,               # Increased for better pattern exploration
    "vf_coef": 1.0,                 # Keep high value importance
    "max_grad_norm": 0.3,           # More conservative for stability
    "n_epochs": 15,                 # More epochs for pattern learning
    "use_sde": False,               # Keep deterministic
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

def save_training_state(path: str, training_start: int, model_path: str, 
                       iteration_time: float = None, total_iterations: int = None, step_size: int = None) -> None:
    """
    Save current training state to file for resumable training.
    
    Args:
        path: Path to save state file
        training_start: Current training window start index
        model_path: Path to current best model
        iteration_time: Time taken for last iteration in seconds
        total_iterations: Total number of iterations planned
        step_size: Step size for calculating completed iterations
    """
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
    
    # Calculate completed iterations based on current iteration
    if step_size is not None:
        current_iteration = training_start // step_size
        # We count the current iteration as completed since we're saving at the end
        state['completed_iterations'] = current_iteration
    
    if total_iterations is not None:
        state['total_iterations'] = total_iterations
    
    with open(path, 'w') as f:
        json.dump(state, f)

def evaluate_model_on_dataset(model_path: str, data: pd.DataFrame, args) -> Dict[str, Any]:
    """
    Evaluate a model on a given dataset.
    
    Args:
        model_path: Path to the model file
        data: Full dataset for evaluation
        args: Training arguments
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not os.path.exists(model_path):
        return None
        
    try:
        from sb3_contrib.ppo_recurrent import RecurrentPPO
        model = RecurrentPPO.load(model_path)
        
        # Create evaluation environment
        env = TradingEnv(
            data=data,
            initial_balance=args.initial_balance,
            balance_per_lot=args.balance_per_lot,
            random_start=False
        )

        # Start progress indicator
        progress_thread = threading.Thread(
            target=show_progress_continuous,
            args=("Evaluating model",)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Run evaluation
            obs, _ = env.reset()
            done = False
            lstm_states = None
            
            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
            # Get performance metrics
            performance = env.metrics.get_performance_summary()
        finally:
            # Always stop the progress indicator
            stop_progress_indicator()
        
        # Calculate score using same weights as evaluation callback
        score = 0.0
        
        # Return component (60% weight)
        returns = performance['return_pct'] / 100
        score += returns * 0.6
        
        # Drawdown penalty (30% weight)
        max_dd = max(performance['max_drawdown_pct'], performance['max_equity_drawdown_pct']) / 100
        drawdown_penalty = max(0, 1 - max_dd * 2)
        score += drawdown_penalty * 0.3
        
        # Profit factor bonus (up to 10% extra)
        pf_bonus = 0.0
        if performance['profit_factor'] > 1.0:
            pf_bonus = min(performance['profit_factor'] - 1.0, 2.0) * 0.05
        score += pf_bonus
        
        return {
            'score': score,
            'returns': returns,
            'drawdown': max_dd,
            'profit_factor': performance['profit_factor'],
            'win_rate': performance['win_rate'],
            'total_trades': performance['total_trades'],
            'metrics': performance
        }
        
    except Exception as e:
        print(f"Error evaluating model {model_path}: {e}")
        return None

def compare_models_on_full_dataset(current_model_path: str, previous_model_path: str, 
                                 full_data: pd.DataFrame, args) -> bool:
    """
    Compare current best model against previous period model on full dataset.
    
    Args:
        current_model_path: Path to current best model
        previous_model_path: Path to previous period model
        full_data: Complete dataset for evaluation
        args: Training arguments
        
    Returns:
        True if current model performs better, False otherwise
    """
    # Evaluate current model
    current_metrics = evaluate_model_on_dataset(current_model_path, full_data, args)
    if not current_metrics:
        print("Could not evaluate current model")
        return False
        
    # Skip comparison if no previous model exists
    if not os.path.exists(previous_model_path):
        print("No previous model to compare against")
        return True
        
    # Evaluate previous model
    previous_metrics = evaluate_model_on_dataset(previous_model_path, full_data, args)
    if not previous_metrics:
        print("Could not evaluate previous model")
        return True
        
    # Print comparison
    print("\n=== Full Dataset Model Comparison ===")
    print(f"Current Model:")
    print(f"  Score: {current_metrics['score']:.4f}")
    print(f"  Return: {current_metrics['returns']*100:.2f}%")
    print(f"  Max DD: {current_metrics['drawdown']*100:.2f}%")
    print(f"  PF: {current_metrics['profit_factor']:.2f}")
    print(f"  Win Rate: {current_metrics['win_rate']:.2f}%")
    
    print(f"\nPrevious Model:")
    print(f"  Score: {previous_metrics['score']:.4f}")
    print(f"  Return: {previous_metrics['returns']*100:.2f}%")
    print(f"  Max DD: {previous_metrics['drawdown']*100:.2f}%")
    print(f"  PF: {previous_metrics['profit_factor']:.2f}")
    print(f"  Win Rate: {previous_metrics['win_rate']:.2f}%")
    
    # Compare scores
    return current_metrics['score'] > previous_metrics['score']

def load_training_state(path: str) -> Tuple[int, str, Dict[str, Any]]:
    """
    Load training state for resuming interrupted training.
    
    Args:
        path: Path to state file
        
    Returns:
        Tuple of (training_start_index, model_path, state_dict)
        Returns (0, None, {}) if no state file exists
    """
    if not os.path.exists(path):
        return 0, None, {}
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        return state['training_start'], state['model_path'], state
    except (FileNotFoundError, json.JSONDecodeError):
        return 0, None, {}

def train_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> RecurrentPPO:
    """
    Train a model using walk-forward optimization.
    
    Implements walk-forward optimization with:
    - Progressive window training
    - Model validation and selection
    - State saving for resumable training
    - Adaptive exploration decay
    - Best model tracking and comparison
    
    Args:
        data: Full dataset for training
        initial_window: Size of initial training window
        step_size: Step size for moving window forward
        args: Training arguments including hyperparameters
        
    Returns:
        RecurrentPPO: Final trained model or last checkpoint if interrupted
        
    Notes:
        - Uses 'validation_size' from args to split training/validation
        - Saves checkpoints and best models in ../results/{seed}/
        - Maintains training state for resumption
        - Adapts exploration and learning rates over iterations
        
    Raises:
        ValueError: If window/step size parameters are invalid
    """
    total_periods = len(data)
    base_timesteps = args.total_timesteps
    
    # Calculate total number of iterations
    total_iterations = (total_periods - initial_window) // step_size + 1
    
    state_path = f"../results/{args.seed}/training_state.json"
    training_start, _, state = load_training_state(state_path)
    
    best_model_path = os.path.join(f"../results/{args.seed}", "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Resuming training from step {training_start} with best model")
        model = RecurrentPPO.load(best_model_path)
    else:
        print("Starting new training")
        training_start = 0
        model = None
    
    # Validate window and step sizes
    if initial_window < step_size * 5:
        raise ValueError("Initial window should be at least 5x step size for stable training")
    
    min_window_overlap = 0.5  # 50% minimum overlap between windows
    if step_size > initial_window * (1 - min_window_overlap):
        raise ValueError(f"Step size too large. Should be <= {initial_window * (1 - min_window_overlap)} for sufficient overlap")

    try:
        while training_start + initial_window <= total_periods:
            iteration = training_start // step_size
            iteration_start_time = time.time()

            # Load training state
            _, _, state = load_training_state(state_path)
            
            # Display timing estimate if we have data
            if state.get('avg_iteration_time'):
                remaining_iterations = total_iterations - state.get('completed_iterations', 0)
                estimated_time = remaining_iterations * state['avg_iteration_time']
                print(f"\nEstimated time remaining: {format_time_remaining(estimated_time)}")
                print(f"Average iteration time: {state['avg_iteration_time']/60:.1f} minutes")
                print(f"Completed iterations: {state.get('completed_iterations', 0) - 1}/{total_iterations}")
            
            # Calculate window boundaries using validation size parameter
            val_size = int(initial_window * args.validation_size)
            train_size = initial_window - val_size
            
            train_start = training_start
            train_end = train_start + train_size
            val_end = min(train_end + val_size, total_periods)
            
            # Ensure we have enough validation data
            if val_end - train_end < val_size * 0.5:  # Require at least half the validation window
                break
                
            train_data = data.iloc[train_start:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
        
            train_data.index = data.index[train_start:train_end]
            val_data.index = data.index[train_end:val_end]
            
            print(f"\n=== Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
            print(f"=== Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
            print(f"=== Walk-forward Iteration: {iteration}/{total_iterations} ===")
        
            env_params = {
                'initial_balance': args.initial_balance,
                'balance_per_lot': args.balance_per_lot,
                'random_start': args.random_start
            }
        
            train_env = Monitor(TradingEnv(train_data, **env_params))
            val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
            
            period_timesteps = base_timesteps
            
            if model is None:
                print("\nPerforming initial training...")
                
                # Initialize new model with optimized hyperparameters for trading
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    train_env,
                    policy_kwargs=POLICY_KWARGS,
                    verbose=0,
                    device=args.device,
                    seed=args.seed,
                    **MODEL_KWARGS
                )
                
                # Set up initial training callbacks
                callbacks = [
                    # Exploration with initial high epsilon
                    CustomEpsilonCallback(
                        start_eps=1.0,
                        end_eps=0.1,
                        decay_timesteps=int(args.total_timesteps * 0.7),
                        iteration=iteration
                    ),
                    # Evaluation and best model saving
                    UnifiedEvalCallback(
                        val_env,
                        train_data=train_data,
                        val_data=val_data,
                        best_model_save_path=f"../results/{args.seed}",
                        log_path=f"../results/{args.seed}",
                        eval_freq=args.eval_freq,
                        deterministic=True,
                        verbose=1,
                        iteration=iteration,
                        training_timesteps=args.total_timesteps
                    )
                ]
                
                # Train initial model
                model.learn(
                    total_timesteps=args.total_timesteps,
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=True
                )
                
                # Check if we found a best model during training
                curr_best_path = os.path.join(f"../results/{args.seed}", "curr_best_model.zip")
                best_model_path = os.path.join(f"../results/{args.seed}", "best_model.zip")
                
                # Update best model if better curr_best was found
                if os.path.exists(curr_best_path):
                    if os.path.exists(best_model_path):
                        if compare_models_on_full_dataset(curr_best_path, best_model_path, data, args):
                            model = RecurrentPPO.load(curr_best_path)
                            model.save(best_model_path)
                            print("\nCurrent best model outperformed previous - saved as best model")
                        else:
                            model = RecurrentPPO.load(best_model_path)
                            print("\nKeeping previous best model")
                    else:
                        model = RecurrentPPO.load(curr_best_path)
                        model.save(best_model_path)
                        print("\nNo previous best model - using current best as first best model")
                        
                save_training_state(state_path, training_start + step_size, best_model_path)
            else:
                print(f"\nContinuing training with existing model...")
                print(f"Training timesteps: {period_timesteps}")
                args.learning_rate = args.learning_rate * 0.95
                model.set_env(train_env)
                
                # Calculate base timesteps for this iteration
                start_timesteps = iteration * period_timesteps
                
                # Set up callbacks for continued training
                callbacks = [
                    # Exploration with decay
                    CustomEpsilonCallback(
                        start_eps=0.25 if iteration < 3 else 0.1,
                        end_eps=0.01,
                        decay_timesteps=int(period_timesteps * 0.75),
                        iteration=iteration
                    ),
                    # Evaluation and model saving
                    UnifiedEvalCallback(
                        val_env,
                        train_data=train_data,
                        val_data=val_data,
                        best_model_save_path=f"../results/{args.seed}",
                        log_path=f"../results/{args.seed}",
                        eval_freq=args.eval_freq,
                        deterministic=True,
                        verbose=0,
                        iteration=iteration,
                        training_timesteps=period_timesteps
                    )
                ]
                
                # Perform training with updated callbacks
                results = model.learn(
                    total_timesteps=period_timesteps,
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=True
                )
                
                # Update timesteps in evaluation results
                unified_callback = callbacks[1]  # The evaluation callback
                for result in unified_callback.eval_results:
                    result['timesteps'] = (result['timesteps'] - period_timesteps) + start_timesteps

            # Compare curr_best against best_model if it exists
            curr_best_path = os.path.join(f"../results/{args.seed}", "curr_best_model.zip")
            best_model_path = os.path.join(f"../results/{args.seed}", "best_model.zip")
            
            if os.path.exists(curr_best_path):
                if os.path.exists(best_model_path):
                    if compare_models_on_full_dataset(curr_best_path, best_model_path, data, args):
                        model = RecurrentPPO.load(curr_best_path)
                        model.save(best_model_path)
                        print("\nCurrent best model outperformed previous - saved as best model")
                    else:
                        model = RecurrentPPO.load(best_model_path)
                        print("\nKeeping previous best model")
                else:
                    model = RecurrentPPO.load(curr_best_path)
                    model.save(best_model_path)
                    print("\nNo previous best model - using current best as first best model")
                    
                # Clean up curr_best files
                os.remove(curr_best_path)
                metrics_path = curr_best_path.replace(".zip", "_metrics.json")
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)
            else:
                print("\nNo curr_best model found - reloading best model for next iteration")
                if os.path.exists(best_model_path):
                    model = RecurrentPPO.load(best_model_path)
                    print("Loaded best model from previous iterations")
                else:
                    print("No best model found - continuing with current model")
                
            # Calculate iteration time and save state
            iteration_time = time.time() - iteration_start_time
            save_training_state(state_path, training_start + step_size, best_model_path,
                          iteration_time=iteration_time, total_iterations=total_iterations,
                          step_size=step_size)
            
            # Move to next iteration
            training_start += step_size

    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved - use same command to resume.")
        return model

    # Load best model for return
    best_model_path = os.path.join(f"../results/{args.seed}", "best_model.zip")
    if os.path.exists(best_model_path):
        model = RecurrentPPO.load(best_model_path)

    return model
