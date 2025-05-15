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

# Walk-forward optimization configuration
TRAINING_PASSES = 30    # Number of passes through each window's data during training

def calculate_timesteps(window_size: int) -> int:
    """
    Calculate training timesteps for current window.
    
    In walk-forward optimization, each iteration uses a window of data.
    The model needs multiple passes through this window to learn effectively.
    Each pass means seeing every bar in the window once.
    
    Example:
    - Window: 2880 bars (6 weeks of 15-min data)
    - Passes: 30 (each bar is seen 30 times during training)
    - Result: 2880 * 30 = 86,400 total training steps
    
    This ensures:
    1. Sufficient learning from the data
    2. Stable convergence through multiple passes
    3. Consistent training effort across different window sizes
    
    Args:
        window_size: Number of bars in current training window
        
    Returns:
        Total timesteps for training on current window
    """
    return window_size * TRAINING_PASSES

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

# Market regime adaptation hyperparameters based on financial DRL research
ADAPTATION_MODEL_KWARGS = {
    "learning_rate": 2.5e-4,     # Higher LR for faster adaptation to regime changes
    "batch_size": 256,           # Smaller batches for more frequent updates
    "n_steps": 1024,            # Balance between stability and adaptability
    "n_epochs": 10,             # More epochs for thorough regime learning
    "clip_range": 0.2,          # Standard PPO clip for stability
    "ent_coef": 0.01,           # Lower entropy to exploit learned strategies
    "gae_lambda": 0.95,         # Higher lambda for better advantage estimation
    "max_grad_norm": 0.5,       # Conservative gradient clipping
    "gamma": 0.99,              # Standard discount for trading
    "clip_range_vf": 0.2,       # Match policy clipping
    "vf_coef": 0.5,            # Balance between value and policy learning
    "use_sde": True,           # Enable state-dependent exploration for regime shifts
    "sde_sample_freq": 4       # Sample new noise every 4 steps for stable exploration
}

# Initialize with adaptation-ready parameters
MODEL_KWARGS = {
    **INITIAL_MODEL_KWARGS,
    "use_sde": True,           # Enable state-dependent exploration from start
    "sde_sample_freq": 4,      # Sample new noise every 4 steps
    "ent_coef": 0.05          # Start with higher entropy for better exploration
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
            contract_size=args.contract_size,
            window_size=args.window_size # Added window_size
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
    Train a model using walk-forward optimization with continuous learning.
    
    Implements enhanced walk-forward optimization with:
    - Dynamic timesteps based on window size (window_size * training_passes)
    - Single unified training configuration for consistent learning
    - Warm start capability with automatic learning rate adjustment
    - Comprehensive validation and test evaluation
    - State saving for resumable training
    
    Args:
        data: Full dataset for training
        initial_window: Size of initial training window
        step_size: Step size for moving window forward
        args: Training arguments including hyperparameters
        
    Returns:
        RecurrentPPO: Final trained model or last checkpoint if interrupted
    """
    total_periods = len(data)
    test_window = args.test_window or initial_window

    # Get training passes from args or use default
    training_passes = args.train_passes if hasattr(args, 'train_passes') else TRAINING_PASSES

    # Calculate total number of iterations and training steps
    total_iterations = (total_periods - initial_window - test_window) // step_size + 1
    train_window_size = initial_window - int(initial_window * args.validation_size)
    timesteps_per_iteration = calculate_timesteps(train_window_size)
    total_timesteps = timesteps_per_iteration * total_iterations
    
    print(f"Training Configuration:")
    print(f"Training passes per window: {training_passes}")
    print(f"Timesteps per iteration: {timesteps_per_iteration:,d}")
    print(f"Total training iterations: {total_iterations}")
    print(f"Total training timesteps: {total_timesteps:,d}")

    # Calculate validation window size for logging
    val_size = int(initial_window * args.validation_size)
    train_size = initial_window - val_size
    print(f"\nWindow Configuration (15-min bars):")
    print(f"Training Size: {train_size} bars")
    print(f"Validation Size: {val_size} bars")
    print(f"Test Size: {test_window} bars")
    print(f"Step Size: {step_size} bars")
    
    state_path = f"../results/{args.seed}/training_state.json"
    checkpoints_dir = os.path.join(f"../results/{args.seed}", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = None
    training_start, initial_model_path, state = load_training_state(state_path)

    if training_start > 0: # Resuming
        # Try to load the model from the last iteration's test model first
        prev_iter_test_model = os.path.join(f"../results/{args.seed}/iteration_{training_start // step_size - 1}", "best_test_model.zip")
        if os.path.exists(prev_iter_test_model):
            print(f"Resuming: Loading model from previous iteration's best test model: {prev_iter_test_model}")
            model = RecurrentPPO.load(prev_iter_test_model, device=args.device)
        else:
            print(f"Resuming: No suitable model found for iteration {training_start // step_size}. Will create a new one.")
            training_start = 0
            model = None
    else: # Starting new training
        print("Starting new training.")
        training_start = 0
        model = None
        # If best_model.zip exists from a previous unrelated run, we might want to ignore it or handle it.
        # For now, a fresh start means a new model.
        training_start = 0
        model = None
    
    # Fixed window sizes for 15min intraday trading
    # Total window = 9 weeks (4320 bars)
    # Step size = 1 week (480 bars) with 50% overlap
    STEP_SIZE = 240  # 1 week of 15min bars with 50% overlap
    
    # Validate that step size matches our sliding window approach
    if step_size != STEP_SIZE:
        print(f"Warning: Adjusting step size from {step_size} to {STEP_SIZE} to match research recommendations")
        step_size = STEP_SIZE

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
            
            """
            Calculate window sizes based on research recommendations for 15min intraday trading:
            - Training window: ~6 weeks (2880 bars) to capture multiple market regimes
            - Validation window: 2 weeks (960 bars) for model selection 
            - Test window: 1 week (480 bars) for out-of-sample evaluation
            - Step size: 1 week (480 bars) with 50% overlap
            Total window: 9 weeks per iteration
            """
            train_start = training_start
            val_start = train_start + 2880  # 6 weeks of 15min bars
            test_start = val_start + 960    # 2 weeks of 15min bars
            test_end = min(test_start + 480, total_periods)  # 1 week of 15min bars
            
            # Ensure we have enough data for all windows
            if test_end - test_start < 240:  # At least half week of test data
                break
                
            # Extract data windows
            train_data = data.iloc[train_start:val_start].copy()
            val_data = data.iloc[val_start:test_start].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Set proper indices
            train_data.index = data.index[train_start:val_start]
            val_data.index = data.index[val_start:test_start]
            test_data.index = data.index[test_start:test_end]
            
            # Ensure we have enough data for minimum trading significance
            min_trading_bars = 240  # At least half week of trading
            if test_end - test_start < min_trading_bars:
                break
            
            print(f"\n=== Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
            print(f"=== Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
            print(f"=== Test Period: {test_data.index[0]} to {test_data.index[-1]} ===")
            print(f"=== Walk-forward Iteration: {iteration}/{total_iterations} ===")
        
            env_params = {
                'initial_balance': args.initial_balance,
                'balance_per_lot': args.balance_per_lot,
                'random_start': args.random_start,
                'point_value': args.point_value,
                'min_lots': args.min_lots,
                'max_lots': args.max_lots,
                'contract_size': args.contract_size,
                'window_size': args.window_size # Added window_size
            }
        
            # Setup environments
            train_env = Monitor(TradingEnv(train_data, **env_params))
            val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
            test_env = Monitor(TradingEnv(test_data, **{**env_params, 'random_start': False}))

            # Handle warm starting from previous best model
            if model is None:
                print("\nPerforming initial training...")
                train_window_size = val_start - train_start
                period_timesteps = calculate_timesteps(train_window_size)
                
                if args.warm_start:
                    # Try loading from previous iteration's best test model
                    prev_iter_model_path = os.path.join(f"../results/{args.seed}/iteration_{iteration-1}", "best_test_model.zip")
                    if os.path.exists(prev_iter_model_path):
                        print(f"Warm starting from previous iteration: {prev_iter_model_path}")
                        model = RecurrentPPO.load(
                            prev_iter_model_path,
                            env=train_env,
                            device=args.device,
                            **ADAPTATION_MODEL_KWARGS  # Use adaptation hyperparameters for warm start
                        )
                    # Fallback to initial model if specified
                    elif args.initial_model and os.path.exists(args.initial_model):
                        print(f"Warm starting from initial model: {args.initial_model}")
                        model = RecurrentPPO.load(
                            args.initial_model,
                            env=train_env,
                            device=args.device,
                            **ADAPTATION_MODEL_KWARGS
                        )
                
            if model is None:
                # Create new model if no warm start or warm start failed
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
                
            # Set up callbacks with test environment
            iteration_dir = os.path.join(f"../results/{args.seed}", f"iteration_{iteration}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            callbacks = [
                # Exploration with initial high epsilon
                CustomEpsilonCallback(
                    start_eps=1.0 if model is None else 0.25,
                    end_eps=0.1 if model is None else 0.01,
                    decay_timesteps=int(period_timesteps * 0.7),
                    iteration=iteration
                ),
                # Evaluation with test set
                UnifiedEvalCallback(
                    eval_env=val_env,
                    test_env=test_env,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    best_model_save_path=iteration_dir,
                    log_path=iteration_dir,
                    eval_freq=args.eval_freq,
                    deterministic=True,
                    verbose=1,
                    iteration=iteration,
                    training_timesteps=period_timesteps
                )
            ]
                
            # Set training parameters based on warm start and model state
            is_new_model = model is None
            # Calculate training timesteps based on window size
            train_window_size = val_start - train_start
            period_timesteps = calculate_timesteps(train_window_size)
            
            # Use adaptation learning rate for warm started models
            if args.warm_start and not is_new_model:
                model.learning_rate = ADAPTATION_MODEL_KWARGS['learning_rate']
            
            print(f"\nTraining for {period_timesteps} timesteps...")
            model.learn(
                total_timesteps=period_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=is_new_model  # Reset only for new models
            )

            # Save checkpoint of the continuously evolving model
            current_checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_iter_{iteration}.zip")
            model.save(current_checkpoint_path)
            print(f"Saved iteration checkpoint: {current_checkpoint_path}")

            # The best model from test set evaluation is already saved by the callback
            # as best_test_model.zip in the iteration directory. This will be used
            # for warm starting the next iteration.
            best_test_model_path = os.path.join(iteration_dir, "best_test_model.zip")
            print(f"\nBest model from this iteration saved at: {best_test_model_path}")
            print("This model will be used for warm starting the next iteration if warm_start=True")
                
            # Calculate iteration time and save state
            iteration_time = time.time() - iteration_start_time
            # Save state with path to best test model for potential resume
            save_training_state(
                state_path, 
                training_start + step_size,
                best_test_model_path,  # Use best test model path for resuming
                iteration_time=iteration_time,
                total_iterations=total_iterations,
                step_size=step_size
            )
            
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

    # Save final evolved model
    print(f"\nWalk-forward optimization completed.")
    final_model_path = os.path.join(f"../results/{args.seed}", "final_evolved_model.zip")
    if model:
        model.save(final_model_path)
        print(f"Final evolved model saved to: {final_model_path}")

    return model
