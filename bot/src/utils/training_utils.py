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
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List
import time
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
    "lstm_hidden_size": 256,          # Larger LSTM for more temporal context
    "n_lstm_layers": 2,               # Keep 2 layers
    "shared_lstm": False,             # Separate LSTM architectures
    "enable_critic_lstm": True,       # Enable LSTM for value estimation
    "net_arch": {
        "pi": [128, 64],              # Policy network
        "vf": [128, 64]               # Value network
    },
    "activation_fn": th.nn.Mish,      # Better activation function
    "optimizer_kwargs": {
        "eps": 1e-5,
        "weight_decay": 1e-6          # Slightly reduced regularization
    }
}

# Initial training hyperparameters
# Training timesteps configuration
INITIAL_TIMESTEPS = 100000        # Full timesteps for initial training
ADAPTATION_TIMESTEPS = 50000      # Reduced timesteps for adaptation phases

# Initial training hyperparameters
INITIAL_MODEL_KWARGS = {
    "learning_rate": 5e-4,        # Standard learning rate for initial training
    "n_steps": 512,              # Shorter sequences with LSTM
    "batch_size": 256,            # Larger batch for stable sparse reward learning
    "gamma": 0.99,                # High gamma for sparse rewards
    "gae_lambda": 0.98,           # Higher lambda for better advantage estimation
    "clip_range": 0.1,            # Smaller clipping for stability
    "clip_range_vf": 0.1,         # Match policy clipping
    "ent_coef": 0.05,             # Lower entropy to focus on sparse signals
    "vf_coef": 1.0,               # Higher value importance for sparse rewards
    "max_grad_norm": 0.5,         # Conservative gradient clipping
    "n_epochs": 12,               # More epochs for thorough learning
    "use_sde": False              # No stochastic dynamics
}

# Adaptation phase hyperparameters (for continuous learning)
# Adaptation phase hyperparameters (for continuous learning)
ADAPTATION_MODEL_KWARGS = {
    "learning_rate": 1e-4,       # Further reduce
    "batch_size": 512,           # Larger batches
    "n_steps": 4096,            # Longer sequences
    "n_epochs": 8,              # Fewer epochs
    "clip_range": 0.15,         # Wider clip
    "ent_coef": 0.02,          # Lower entropy
    "gae_lambda": 0.92,        # Lower lambda
    "max_grad_norm": 0.3,      # Add gradient clipping
    "gamma": 0.99,             # Keep high gamma for sparse rewards
    "clip_range_vf": 0.15,     # Match policy clipping
    "vf_coef": 1.0,           # Keep higher value importance
    "use_sde": False          # No stochastic dynamics
}

# Set MODEL_KWARGS to INITIAL_MODEL_KWARGS by default
MODEL_KWARGS = INITIAL_MODEL_KWARGS

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
        model = RecurrentPPO.load(model_path, device=args.device)
        
        # Create evaluation environment
        env = TradingEnv(
            data=data,
            initial_balance=args.initial_balance,
            balance_per_lot=args.balance_per_lot,
            random_start=False,
            point_value=args.point_value,
            min_lots=args.min_lots,
            max_lots=args.max_lots,
            contract_size=args.contract_size
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
            
            # For regime consistency
            regime_pnl = {'trending': [], 'ranging': []}
            last_balance = env.balance

            while not done:
                # Get trend strength and volatility breakout from observation
                trend_strength = obs[5]  # Index 5 is trend_strength in features
                volatility_breakout = obs[4]  # Index 4 is volatility_breakout in features
                
                # Define trending vs ranging based on both ADX (trend_strength) and volatility_breakout
                is_trending = trend_strength > 0.3 and volatility_breakout > 0.7

                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states if 'lstm_states' in locals() else None,
                    deterministic=True
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track PnL per step for regime analysis
                step_pnl = env.balance - last_balance
                last_balance = env.balance

                if is_trending:
                    regime_pnl['trending'].append(step_pnl)
                else:
                    regime_pnl['ranging'].append(step_pnl)

            # Get performance metrics
            performance = env.metrics.get_performance_summary()
        finally:
            # Always stop the progress indicator
            stop_progress_indicator()
        
        # Calculate score using enhanced scoring system
        score = 0.0
        
        # Get key performance metrics
        returns = performance['return_pct'] / 100
        max_dd = max(performance['max_drawdown_pct'], performance['max_equity_drawdown_pct']) / 100
        profit_factor = performance['profit_factor']
        win_rate = performance['win_rate'] / 100
        
        # Regime Consistency Score
        total_trending_pnl = sum(regime_pnl['trending'])
        total_ranging_pnl = sum(regime_pnl['ranging'])

        # Normalize PnL by initial balance to make it comparable
        norm_trending_pnl = total_trending_pnl / args.initial_balance if args.initial_balance else 0
        norm_ranging_pnl = total_ranging_pnl / args.initial_balance if args.initial_balance else 0

        regime_consistency_bonus = 0.0
        # Basic check: reward if both are positive, penalize if one is positive and other is very negative
        if norm_trending_pnl > 0 and norm_ranging_pnl > 0:
            regime_consistency_bonus += 0.5 # Max bonus if both are profitable
            # Further reward if they are somewhat balanced
            if min(abs(norm_trending_pnl), abs(norm_ranging_pnl)) / (max(abs(norm_trending_pnl), abs(norm_ranging_pnl)) + 1e-6) > 0.25:
                 regime_consistency_bonus += 0.5
        elif (norm_trending_pnl > 0 and norm_ranging_pnl < -0.05 * abs(norm_trending_pnl)) or \
             (norm_ranging_pnl > 0 and norm_trending_pnl < -0.05 * abs(norm_ranging_pnl)):
            regime_consistency_bonus -= 0.5 # Penalize if one regime heavily subsidizes losses in another

        # Score components (weights adjusted)
        # 1. Risk-adjusted return component (35% weight)
        risk_adj_return = returns / (max_dd + 0.05) # Add small constant to avoid division by zero
        score += risk_adj_return * 0.35

        # 2. Raw returns component (25% weight)
        score += returns * 0.25

        # 3. Regime Consistency Bonus (20% weight)
        # regime_consistency_bonus is between -0.5 and 1.0.
        # We want to scale this to contribute positively.
        # Let's scale it: (bonus + 0.5) / 1.5 to get it into [0,1] range, then multiply by weight.
        scaled_regime_consistency = (regime_consistency_bonus + 0.5) / 1.5
        score += scaled_regime_consistency * 0.20

        # 4. Profit factor bonus (10% weight)
        pf_bonus = 0.0
        if profit_factor > 1.0:
            # Scale profit factor: (PF - 1) capped at 2, then map to 0-1 for bonus component
            pf_bonus_raw = min(profit_factor - 1.0, 2.0) / 2.0
            pf_bonus = pf_bonus_raw
        score += pf_bonus * 0.10

        # 5. Win rate component (10% weight)
        score += win_rate * 0.10 # win_rate is already 0-1
        
        return {
            'score': score,
            'returns': returns,
            'risk_adj_return': risk_adj_return,
            'drawdown': max_dd,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': performance['total_trades'],
            'metrics': performance,
            'regime_consistency_metric': scaled_regime_consistency, # Store for analysis
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
    print(f"  Risk-Adjusted Return: {current_metrics['risk_adj_return']:.2f}")
    print(f"  Max DD: {current_metrics['drawdown']*100:.2f}%")
    print(f"  PF: {current_metrics['profit_factor']:.2f}")
    print(f"  Win Rate: {current_metrics['win_rate']:.2f}%")
    
    print(f"\nPrevious Model:")
    print(f"  Score: {previous_metrics['score']:.4f}")
    print(f"  Return: {previous_metrics['returns']*100:.2f}%")
    print(f"  Risk-Adjusted Return: {previous_metrics['risk_adj_return']:.2f}")
    print(f"  Max DD: {previous_metrics['drawdown']*100:.2f}%")
    print(f"  PF: {previous_metrics['profit_factor']:.2f}")
    print(f"  Win Rate: {previous_metrics['win_rate']:.2f}%")
    print(f"  Regime Consistency: {previous_metrics.get('regime_consistency_metric', 'N/A'):.2f}")
    
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
    checkpoints_dir = os.path.join(f"../results/{args.seed}", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = None
    training_start, initial_model_path_from_state, state = load_training_state(state_path)
    overall_best_model_path = os.path.join(f"../results/{args.seed}", "best_model.zip") # Renamed for clarity

    if training_start > 0: # Resuming
        # Try to load the model from the last iteration's checkpoint first
        last_iter_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_iter_{training_start // step_size -1}.zip")
        if os.path.exists(last_iter_checkpoint_path):
            print(f"Resuming: Loading model from last iteration checkpoint: {last_iter_checkpoint_path}")
            model = RecurrentPPO.load(last_iter_checkpoint_path, device=args.device)
        elif initial_model_path_from_state and os.path.exists(initial_model_path_from_state): # Fallback to model_path from state (overall_best_model_path)
            print(f"Resuming: Loading model from state file (overall best): {initial_model_path_from_state}")
            model = RecurrentPPO.load(initial_model_path_from_state, device=args.device)
        elif os.path.exists(overall_best_model_path): # Fallback to current overall_best_model_path
            print(f"Resuming: Loading model from existing overall best model: {overall_best_model_path}")
            model = RecurrentPPO.load(overall_best_model_path, device=args.device)
        else:
            print(f"Resuming: No suitable model found to load for iteration {training_start // step_size}. Will create a new one.")
            training_start = 0 # Effectively start fresh if no model can be loaded
            model = None
    else: # Starting new training
        print("Starting new training.")
        training_start = 0
        model = None
        # If best_model.zip exists from a previous unrelated run, we might want to ignore it or handle it.
        # For now, a fresh start means a new model.
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

            # Load training state for current iteration
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
                'random_start': args.random_start,
                'point_value': args.point_value,
                'min_lots': args.min_lots,
                'max_lots': args.max_lots,
                'contract_size': args.contract_size
            }
        
            train_env = Monitor(TradingEnv(train_data, **env_params))
            val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
            
            if model is None:
                print("\nPerforming initial training...")
                period_timesteps = INITIAL_TIMESTEPS
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
                    **INITIAL_MODEL_KWARGS
                )
                
                # Set up initial training callbacks
                callbacks = [
                    # Exploration with initial high epsilon
                    CustomEpsilonCallback(
                        start_eps=1.0,
                        end_eps=0.1,
                        decay_timesteps=int(period_timesteps * 0.7),
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
                        training_timesteps=period_timesteps
                    )
                ]
                
                # Train initial model
                model.learn(
                    total_timesteps=period_timesteps, # Initial training with full timesteps
                    callback=callbacks,
                    progress_bar=True,
                    reset_num_timesteps=True # Reset for the very first learning phase
                )
                # Initial model is now trained, subsequent .learn() calls will be continuous.
            else: # Continuing training with the existing, evolving model
                print(f"\nContinuing training with existing model for iteration {iteration}...")
                period_timesteps = ADAPTATION_TIMESTEPS
                print(f"Training timesteps: {period_timesteps}")

                # Set adaptation hyperparameters
                model.learning_rate = ADAPTATION_MODEL_KWARGS['learning_rate']
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
                    reset_num_timesteps=False # IMPORTANT: Continue learning from current state
                )

                # Update timesteps in evaluation results (if callback structure requires it)
                # unified_callback = callbacks[1]
                # for result in unified_callback.eval_results:
                #     result['timesteps'] = (result['timesteps'] - period_timesteps) + start_timesteps

            # Save checkpoint of the continuously evolving model
            current_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_iter_{iteration}.zip")
            model.save(current_checkpoint_path)
            print(f"Saved iteration checkpoint: {current_checkpoint_path}")

            # Logic for updating the overall_best_model_path
            # UnifiedEvalCallback saves its best model from the validation window as "curr_best_model.zip"
            curr_best_candidate_path = os.path.join(f"../results/{args.seed}", "curr_best_model.zip")

            if os.path.exists(curr_best_candidate_path):
                if not os.path.exists(overall_best_model_path) or \
                   compare_models_on_full_dataset(curr_best_candidate_path, overall_best_model_path, data, args):
                    # New overall best model found based on full dataset evaluation
                    # Copy curr_best_candidate_path to overall_best_model_path
                    # Note: This does NOT change the 'model' variable which continues to evolve.
                    # We load and save to ensure it's a clean copy.
                    temp_best_model = RecurrentPPO.load(curr_best_candidate_path, device=args.device)
                    temp_best_model.save(overall_best_model_path)
                    print(f"\nOverall best model updated: {overall_best_model_path}")

                # Clean up the temporary curr_best_model.zip and its metrics
                os.remove(curr_best_candidate_path)
                metrics_path = curr_best_candidate_path.replace(".zip", "_metrics.json")
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)
            else:
                print("\nNo new candidate for overall best model from this iteration's validation.")
                
            # Calculate iteration time and save state
            iteration_time = time.time() - iteration_start_time
            # The model_path saved in state should be the overall_best_model_path,
            # as it's the most robust one for a potential full resume.
            # However, for resuming the continuous flow, checkpoint_iter_X.zip is used.
            save_training_state(state_path, training_start + step_size, overall_best_model_path,
                          iteration_time=iteration_time, total_iterations=total_iterations,
                          step_size=step_size)
            
            # Move to next iteration
            training_start += step_size

    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress saved - use same command to resume.")
        # Save the current state of the evolving model upon interruption
        if model:
            interrupt_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_interrupt_iter_{iteration}.zip")
            model.save(interrupt_checkpoint_path)
            print(f"Saved interrupt checkpoint: {interrupt_checkpoint_path}")
        return model # Return the current evolving model

    # At the end of all iterations, the 'model' variable holds the continuously evolved model.
    # 'overall_best_model_path' points to the model that performed best on the full dataset.
    # Decide which one to return as the "final" model.
    # For continuous learning, the latest state of 'model' is often most relevant.
    print(f"\nWalk-forward optimization completed.")
    print(f"Final state of continuously evolved model is in memory (and last checkpoint).")
    if os.path.exists(overall_best_model_path):
        print(f"Overall best performing model on full dataset saved at: {overall_best_model_path}")
        # Optionally, load and return the overall_best_model_path if that's preferred as the final output
        # model = RecurrentPPO.load(overall_best_model_path, device=args.device)

    # Save the final state of the continuously evolved model
    final_evolving_model_path = os.path.join(f"../results/{args.seed}", "model_final_evolved.zip")
    if model:
        model.save(final_evolving_model_path)
        print(f"Final state of continuously evolved model saved to: {final_evolving_model_path}")

    return model # Return the continuously evolved model
