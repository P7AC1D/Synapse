import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trade_environment import TradingEnv
import torch as th

class CustomEpsilonCallback(BaseCallback):
    """Custom callback for epsilon decay during training"""
    def __init__(self, start_eps=0.2, end_eps=0.02, decay_timesteps=800000):
        super().__init__()
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_timesteps = decay_timesteps
        
    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.decay_timesteps)
        current_eps = self.start_eps + progress * (self.end_eps - self.start_eps)
        
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'exploration_rate'):
            self.model.policy.exploration_rate = current_eps
            
        return True

class UnifiedEvalCallback(BaseCallback):
    """Optimized evaluation callback with enhanced progress tracking."""
    def __init__(self, eval_env, eval_freq=50000, best_model_save_path=None, 
                 log_path=None, deterministic=True, verbose=1, iteration=0):
        super(UnifiedEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.best_final_balance = -float("inf")
        self.best_return = -float("inf")
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration  # Track walk-forward iteration
        
        if hasattr(self.eval_env, 'env'):
            self.eval_env.env.raw_data_backup = self.eval_env.env.raw_data.copy()
        else:
            self.eval_env.raw_data_backup = self.eval_env.raw_data.copy()
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run complete evaluation episode
            obs, _ = self.eval_env.reset()
            done = False
            lstm_states = None
            episode_reward = 0
            
            while not done:
                action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Get final performance metrics
            if hasattr(self.eval_env, 'env'):
                final_balance = self.eval_env.env.balance
                total_return = ((self.eval_env.env.balance - self.eval_env.env.initial_balance) 
                              / self.eval_env.env.initial_balance)
            else:
                final_balance = self.eval_env.balance
                total_return = ((self.eval_env.balance - self.eval_env.initial_balance) 
                              / self.eval_env.initial_balance)
            
            if self.verbose > 0:
                print(f"\nEval num_timesteps={self.num_timesteps}, "
                      f"balance={final_balance:.2f}, "
                      f"return={total_return*100:.2f}%, "
                      f"reward={episode_reward:.2f}")
            
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'balance': float(final_balance),
                    'return': float(total_return),
                    'reward': float(episode_reward)
                })
                
                # Save iteration-specific results
                iteration_file = os.path.join(self.log_path, f"eval_results_iter_{self.iteration}.json")
                with open(iteration_file, "w") as f:
                    json.dump(self.eval_results, f, indent=2)
                    
                # Also update combined results file
                combined_file = os.path.join(self.log_path, "eval_results_all.json")
                try:
                    with open(combined_file, "r") as f:
                        all_results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_results = {}
                    
                # Get the TradingEnv instance (unwrap Monitor if needed)
                env = self.eval_env
                while hasattr(env, 'env'):
                    env = env.env
                    if isinstance(env, TradingEnv):
                        break

                # Always include base metrics
                period_info = {
                    'results': self.eval_results,
                    'iteration': self.iteration,
                    'balance': float(env.balance),
                    'total_trades': len(env.trades),
                    'win_count': env.win_count,
                    'loss_count': env.loss_count
                }

                # Add time period info if available
                try:
                    data_index = env.raw_data.index if isinstance(env.raw_data.index, pd.DatetimeIndex) else pd.to_datetime(env.raw_data.index)
                    period_info.update({
                        'period_start': str(data_index[0]),
                        'period_end': str(data_index[-1])
                    })
                except Exception as e:
                    print(f"Warning: Could not add period timestamps: {str(e)}")

                all_results[f"iteration_{self.iteration}"] = period_info
                
                with open(combined_file, "w") as f:
                    json.dump(all_results, f, indent=2)
            
            # Save best model based on final balance AND positive return
            if final_balance > self.best_final_balance:
                if self.verbose > 0:
                    print(f"New best balance: {final_balance:.2f}")
                
                self.best_final_balance = final_balance
                
                if total_return > self.best_return:
                    self.best_return = total_return
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_balance_model"))
                        print(f"Saved new best model with {total_return*100:.2f}% return")
                    
            # Show evaluation metrics
            print("\n===== EVALUATION METRICS =====")
            print(f"Final balance: {final_balance:.2f}")
            print(f"Total return: {total_return*100:.2f}%")
            print(f"Total reward: {episode_reward:.2f}")
            
            if hasattr(self.eval_env, 'env'):
                self.eval_env.env.render()
            else:
                self.eval_env.render()
                
            self.last_time_trigger = self.n_calls
        
        return True

def train_model(train_env, val_env, args, iteration=0):
    """Train the PPO model with optimized hyperparameters for BTC trading."""
    lr_schedule = get_linear_fn(
        start=args.learning_rate,
        end=args.final_learning_rate,
        end_fraction=0.95  # Longer learning rate schedule
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr_schedule,
        n_steps=1024,  # Longer sequences to capture patterns
        batch_size=256,  # Larger batches for stability
        gamma=0.99,
        gae_lambda=0.98,  # More emphasis on long-term rewards
        clip_range=0.1,   # More conservative updates
        clip_range_vf=0.1,
        ent_coef=0.005,   # Less random exploration
        vf_coef=0.8,      # Stronger value function
        max_grad_norm=0.3, # More conservative gradient updates
        use_sde=False,
        policy_kwargs={
            "optimizer_class": th.optim.AdamW,  # Using AdamW for better generalization
            "lstm_hidden_size": 128,  # Larger LSTM for pattern recognition
            "n_lstm_layers": 2,       # Two LSTM layers
            "shared_lstm": False,  # Use separate LSTMs
            "enable_critic_lstm": True,  # Enable LSTM for critic
            "net_arch": {
                "pi": [128, 64],  # Larger networks
                "vf": [128, 64]
            },
            "optimizer_kwargs": {
                "eps": 1e-5,
                "weight_decay": 1e-4  # L2 regularization
            }
        },
        verbose=0,
        device=args.device,
        seed=args.seed
    )
    
    callbacks = []
    
    epsilon_callback = CustomEpsilonCallback(
        start_eps=0.05,  # Much less initial exploration
        end_eps=0.005,   # Very conservative final exploration
        decay_timesteps=int(args.total_timesteps * 0.8)  # Longer decay
    )
    callbacks.append(epsilon_callback)
    
    unified_callback = UnifiedEvalCallback(
        val_env,
        best_model_save_path=f"../results/{args.seed}",
        log_path=f"../results/{args.seed}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1,
        iteration=iteration
    )
    callbacks.append(unified_callback)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=f"../results/{args.seed}/checkpoints/{args.model_name}",
        name_prefix="ppo_lstm"
    )
    callbacks.append(checkpoint_callback)
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False if hasattr(model, '_last_obs') else True
    )
    
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
    """Implement walk-forward optimization for training with resume capability."""
    total_periods = len(data)
    base_timesteps = args.total_timesteps
    
    # Setup state tracking
    state_path = f"../results/{args.seed}/training_state.json"
    training_start, model_path = load_training_state(state_path)
    
    if model_path and os.path.exists(model_path):
        print(f"Resuming training from step {training_start}")
        model = RecurrentPPO.load(model_path)
    else:
        print("Starting new training")
        training_start = 0
        model = None
    
    while training_start + initial_window + step_size <= total_periods:
        iteration = training_start // step_size
        
        # Define data windows
        train_end = training_start + initial_window
        val_end = min(train_end + step_size, total_periods)
        
        train_data = data.iloc[training_start:train_end]
        val_data = data.iloc[train_end:val_end]  # Only evaluate on new, unseen data
        
        print(f"\n=== Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
        print(f"Validation Period: {val_data.index[0]} to {val_data.index[-1]}")
        print(f"Walk-forward Iteration: {iteration}")
        
        env_params = {
            'initial_balance': args.initial_balance,
            'bar_count': args.bar_count,
            'lot_percentage': 0.01  # Reduced risk per trade
        }
        
        train_env = Monitor(TradingEnv(train_data, **{**env_params, 'random_start': True}))
        val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
        
        # Keep timesteps consistent across iterations
        period_timesteps = base_timesteps
        
        if model is None:
            # Initial training
            model = train_model(train_env, val_env, args, iteration=iteration)
        else:
            # Continue training with existing model
            print(f"\nContinuing training with existing model...")
            print(f"Training timesteps: {period_timesteps}")
            args.learning_rate = args.learning_rate * 0.95  # More gradual learning rate decay
            model.set_env(train_env)
            
            callbacks = []
            
            epsilon_callback = CustomEpsilonCallback(
                start_eps=0.03,  # Lower exploration for fine-tuning
                end_eps=0.005,
                decay_timesteps=int(period_timesteps * 0.9)  # Longer decay
            )
            callbacks.append(epsilon_callback)
            
            unified_callback = UnifiedEvalCallback(
                val_env,
                best_model_save_path=f"../results/{args.seed}",
                log_path=f"../results/{args.seed}",
                eval_freq=args.eval_freq,
                deterministic=True,
                verbose=1,
                iteration=iteration
            )
            callbacks.append(unified_callback)
            
            model.learn(
                total_timesteps=period_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False
            )
        
        # Save period-specific model and training state
        period_model_path = f"../results/{args.seed}/model_period_{training_start}_{train_end}.zip"
        model.save(period_model_path)
        save_training_state(state_path, training_start + step_size, period_model_path)
        print(f"Saved model and state for period {training_start} to {train_end}")
        
        try:
            # Move window forward
            training_start += step_size
        except KeyboardInterrupt:
            print("\nTraining interrupted. Progress saved - use same command to resume.")
            return model
        
    return model

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
    parser.add_argument('--initial_window', type=int, default=30,
                      help='Initial training window in days')
    parser.add_argument('--step_size', type=int, default=14,
                      help='Walk-forward step size in days')
    parser.add_argument('--bar_count', type=int, default=20,  # Increased history
                      help='Number of bars in observation window')
    
    parser.add_argument('--total_timesteps', type=int, default=500000,  # Increased timesteps
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,  # Reduced learning rate
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=1e-5,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=50000,  # Less frequent evaluation
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
    
    # Convert dates to datetime if they're not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Calculate periods in terms of rows based on 15-minute bars
    bars_per_day = 24 * 4  # 96 bars per day (24 hours * 4 fifteen-minute periods per hour)
    initial_window_bars = args.initial_window * bars_per_day
    step_size_bars = args.step_size * bars_per_day
    
    if args.resume:
        state_path = f"../results/{args.seed}/training_state.json"
        if os.path.exists(state_path):
            print("\nResuming walk-forward optimization...")
        else:
            print("\nNo previous state found. Starting new training...")
    else:
        print("\nStarting new walk-forward optimization...")
    
    try:
        model = train_walk_forward(data, initial_window_bars, step_size_bars, args)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        return
    
    print("\nWalk-forward optimization completed.")
    print(f"Final model saved at: ../results/{args.seed}/model_final.zip")
    model.save(f"../results/{args.seed}/model_final.zip")

if __name__ == "__main__":
    main()
