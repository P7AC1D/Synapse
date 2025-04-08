"""Training utility functions for the PPO model."""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Tuple
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trade_environment import TradingEnv
import torch as th

from callbacks.epsilon_callback import CustomEpsilonCallback
from callbacks.eval_callback import UnifiedEvalCallback

def train_model(train_env, val_env, train_data, val_data, args, iteration=0):
    """Train the PPO model with optimized hyperparameters for BTC trading."""
    lr_schedule = get_linear_fn(
        start=args.learning_rate,
        end=args.final_learning_rate,
        end_fraction=0.95
    )
    
    # Update the policy_kwargs by removing the unsupported dropout parameter
    policy_kwargs = {
        "optimizer_class": th.optim.AdamW,
        "lstm_hidden_size": 256,          # Increased for 11 features
        "n_lstm_layers": 2,               # Maintain 2 layers for temporal learning
        "shared_lstm": True,              # Share LSTM to reduce parameters
        "enable_critic_lstm": False,      # Disable separate critic LSTM
        "net_arch": {
            "pi": [128, 64],              # Wider networks for 11 features
            "vf": [128, 64]               # Symmetric critic network
        },
        "activation_fn": th.nn.ReLU,      # Explicitly define activation
        "optimizer_kwargs": {
            "eps": 1e-5,
            "weight_decay": 5e-7          # Increased weight decay acts as regularization
        }
    }

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=3e-4,           # Reduced learning rate for stability
        n_steps=256,                  # Keep longer sequences for context
        batch_size=64,                # Increased batch size to reduce variance
        gamma=0.99,                   # Keep discount factor
        gae_lambda=0.95,              # Keep lambda value
        clip_range=0.2,               # Reduced clipping for more stability
        clip_range_vf=0.2,            # Match policy clipping
        ent_coef=0.03,                # Slightly reduced entropy for more exploitation
        vf_coef=0.5,                  # Keep value coefficient
        max_grad_norm=0.3,            # Lower gradient norm for more stability
        use_sde=False,                
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=args.device,
        seed=args.seed
    )
    
    callbacks = []
    
    # Configure epsilon exploration tuned for 6 features
    # Configure enhanced exploration with iteration awareness
    epsilon_callback = CustomEpsilonCallback(
        start_eps=0.5,     # Higher initial exploration
        end_eps=0.02,      # Keep same final exploration
        decay_timesteps=int(args.total_timesteps * 0.8),  # Extended decay period
        iteration=iteration  # Pass iteration number for adaptive decay
    )
    callbacks.append(epsilon_callback)
    
    # Add evaluation callback
    unified_callback = UnifiedEvalCallback(
        val_env,
        train_data=train_data,
        val_data=val_data,
        best_model_save_path=f"../results/{args.seed}",
        log_path=f"../results/{args.seed}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1,
        iteration=iteration
    )
    callbacks.append(unified_callback)
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=f"../results/{args.seed}/checkpoints/{args.model_name}",
        name_prefix="ppo_lstm"
    )
    callbacks.append(checkpoint_callback)
    
    # Calculate start timesteps for consistent progression
    start_timesteps = iteration * args.total_timesteps
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True  # Reset timesteps for each iteration
    )
    
    # Update timesteps in evaluation results to maintain sequence
    for result in unified_callback.eval_results:
        result['timesteps'] = (result['timesteps'] - args.total_timesteps) + start_timesteps
    
    final_model_path = f"../results/{args.seed}/{args.model_name}"
    model.save(final_model_path)
    print(f"Model saved as {final_model_path}")
    
    best_model_path = f"../results/{args.seed}/best_balance_model.zip"
    if os.path.exists(best_model_path):
        print(f"Loading best model based on full dataset performance: {best_model_path}")
        model = RecurrentPPO.load(best_model_path)
    
    return model

def save_training_state(path: str, training_start: int, model_path: str) -> None:
    """Save current training state to file."""
    state = {
        'training_start': training_start,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }
    with open(path, 'w') as f:
        json.dump(state, f)

def load_training_state(path: str) -> Tuple[int, str]:
    """Load training state from file."""
    if not os.path.exists(path):
        return 0, None
    with open(path, 'r') as f:
        state = json.load(f)
    return state['training_start'], state['model_path']

def train_walk_forward(data: pd.DataFrame, initial_window: int, step_size: int, args) -> None:
    """Train with walk-forward optimization."""
    total_periods = len(data)
    base_timesteps = args.total_timesteps
    
    state_path = f"../results/{args.seed}/training_state.json"
    training_start, model_path = load_training_state(state_path)
    
    if model_path and os.path.exists(model_path):
        print(f"Resuming training from step {training_start}")
        model = RecurrentPPO.load(model_path)
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

    while training_start + initial_window <= total_periods:
        iteration = training_start // step_size
        
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
        print(f"Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
        print(f"Walk-forward Iteration: {iteration}")
        
        env_params = {
            'initial_balance': args.initial_balance,
            'balance_per_lot': args.balance_per_lot,
            'random_start': args.random_start
        }
        
        train_env = Monitor(TradingEnv(train_data, **env_params))
        val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
        
        period_timesteps = base_timesteps
        
        if model is None:
            model = train_model(train_env, val_env, train_data, val_data, args, iteration=iteration)
        else:
            print(f"\nContinuing training with existing model...")
            print(f"Training timesteps: {period_timesteps}")
            args.learning_rate = args.learning_rate * 0.95
            model.set_env(train_env)
            
            callbacks = []
            
            # Adjust exploration based on iteration progress with better decay
            start_eps = 0.25 if iteration < 3 else 0.1  # Slightly lower exploration
            epsilon_callback = CustomEpsilonCallback(
                start_eps=start_eps,
                end_eps=0.01,    # Lower end exploration for better exploitation
                decay_timesteps=int(period_timesteps * 0.75),  # Extended decay period
                iteration=iteration
            )
            callbacks.append(epsilon_callback)            
            
            # Create evaluation callback for continued training
            unified_callback = UnifiedEvalCallback(
                val_env,
                train_data=train_data,
                val_data=val_data,
                best_model_save_path=f"../results/{args.seed}",
                log_path=f"../results/{args.seed}",
                eval_freq=args.eval_freq,
                deterministic=True,
                verbose=1,
                iteration=iteration
            )
            callbacks.append(unified_callback)
            
            # Calculate base timesteps for this iteration
            start_timesteps = iteration * period_timesteps
            
            model.learn(
                total_timesteps=period_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=True  # Reset timesteps for each iteration
            )
            
            # Update timesteps in evaluation results to maintain sequence
            for result in unified_callback.eval_results:
                result['timesteps'] = (result['timesteps'] - period_timesteps) + start_timesteps
        
        period_model_path = f"../results/{args.seed}/model_period_{training_start}_{train_end}.zip"
        model.save(period_model_path)
        save_training_state(state_path, training_start + step_size, period_model_path)
        print(f"Saved model and state for period {training_start} to {train_end}")
        
        try:
            training_start += step_size
        except KeyboardInterrupt:
            print("\nTraining interrupted. Progress saved - use same command to resume.")
            return model
        
    return model
