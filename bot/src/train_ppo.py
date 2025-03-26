import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
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
                 log_path=None, deterministic=True, verbose=1):
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
        
        # Backup original dataset
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
                
                with open(os.path.join(self.log_path, "eval_results.json"), "w") as f:
                    json.dump(self.eval_results, f)
            
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

def train_model(train_env, val_env, args):
    """Train the PPO model with optimized hyperparameters for BTC trading."""
    lr_schedule = get_linear_fn(
        start=args.learning_rate,
        end=args.final_learning_rate,
        end_fraction=0.9
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=lr_schedule,
        n_steps=2048,  # Longer sequences for better temporal context
        batch_size=512,  # Larger batches for more stable updates
        gamma=0.99,  # Keep this for 15-min timeframe
        gae_lambda=0.98,  # Increase advantage estimation horizon
        clip_range=0.1,  # More conservative policy updates
        clip_range_vf=0.1,  # Match policy clip range
        ent_coef=0.005,  # Lower entropy to exploit learned behaviors
        vf_coef=1.0,  # Stronger value estimation
        max_grad_norm=0.3,  # More conservative gradient updates
        use_sde=False,
        policy_kwargs={
            "optimizer_class": th.optim.Adam,
            "lstm_hidden_size": 128,  # Larger memory for pattern retention
            "n_lstm_layers": 2,  # Two layers for hierarchical patterns
            "shared_lstm": False,  # Separate memory streams
            "enable_critic_lstm": True,  # Dedicated value memory
            "net_arch": {
                "pi": [128, 64],  # Deeper policy network
                "vf": [128, 64]  # Matching value network
            },
            "optimizer_kwargs": {
                "eps": 1e-5
            }
        },
        verbose=0,
        device=args.device,
        seed=args.seed
    )
    
    callbacks = []
    
    epsilon_callback = CustomEpsilonCallback(
        start_eps=0.2,  # Higher initial exploration
        end_eps=0.02,  # Keep some exploration
        decay_timesteps=int(args.total_timesteps * 0.8)
    )
    callbacks.append(epsilon_callback)
    
    unified_callback = UnifiedEvalCallback(
        val_env,
        best_model_save_path=f"../results/{args.seed}",
        log_path=f"../results/{args.seed}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1
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

def main():
    parser = argparse.ArgumentParser(description='Train a PPO-LSTM model for trading')
    
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
    parser.add_argument('--bar_count', type=int, default=10,  # Reduced to focus on recent data
                      help='Number of bars to use for each observation')
    
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=1e-4,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=50000,
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
    
    print("\nCreating environments for training and validation")
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    
    print(f"\nEnvironment Configuration:")
    print(f"Training Data: {len(train_data)} bars ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Validation Data: {len(data)} bars (full dataset for comprehensive evaluation)")
    
    env_params = {
        'initial_balance': args.initial_balance,
        'bar_count': args.bar_count
    }
    
    train_env = Monitor(TradingEnv(train_data, **{**env_params, 'random_start': True}))
    val_env = Monitor(TradingEnv(data, **{**env_params, 'random_start': False}))
    
    model = train_model(train_env, val_env, args)
    
    print("\nRunning final evaluation...")
    obs, _ = val_env.reset()
    done = False
    total_reward = 0
    lstm_states = None
    
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"\nFinal Evaluation Results:")
    print(f"Total reward: {total_reward:.2f}")
    val_env.render()

if __name__ == "__main__":
    main()
