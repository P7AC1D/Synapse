import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import argparse
import json
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
    def __init__(self, start_eps=0.3, end_eps=0.01, decay_timesteps=1000000):
        super().__init__()
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_timesteps = decay_timesteps
        
    def _on_step(self) -> bool:
        # Calculate current epsilon using linear decay
        progress = min(1.0, self.num_timesteps / self.decay_timesteps)
        current_eps = self.start_eps + progress * (self.end_eps - self.start_eps)
        
        # Update exploration rate
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'exploration_rate'):
            self.model.policy.exploration_rate = current_eps
            
        return True

class CustomRenderCallback(BaseCallback):
    """Custom callback for explicitly calling render on evaluation environment"""
    def __init__(self, eval_env, eval_freq=10000, verbose=0):
        super(CustomRenderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_time_trigger = 0
        
    def _on_step(self) -> bool:
        if self.n_calls >= self.last_time_trigger + self.eval_freq:
            print("\n===== EVALUATION METRICS =====")
            
            # Run a quick evaluation episode
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            lstm_states = None
            
            while not done:
                action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            # Call render to display metrics
            print(f"\nEvaluation episode complete. Total reward: {total_reward:.2f}")
            if hasattr(self.eval_env, 'env'):
                # For Monitor-wrapped environments
                self.eval_env.env.render(mode="human")
            else:
                self.eval_env.render(mode="human")
                
            self.last_time_trigger = self.n_calls
        
        return True

class BalanceEvalCallback(BaseCallback):
    """
    Callback for evaluating and saving models based on final account balance.
    """
    def __init__(self, eval_env, eval_freq=10000, best_model_save_path=None, 
                 log_path=None, n_eval_episodes=5, deterministic=True, verbose=1):
        super(BalanceEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_mean_balance = -float("inf")
        self.eval_results = []
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_balance, std_balance, mean_reward = self.evaluate_policy()
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"mean_balance={mean_balance:.2f} +/- {std_balance:.2f}, "
                      f"mean_reward={mean_reward:.2f}")
            
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'mean_balance': float(mean_balance),
                    'std_balance': float(std_balance),
                    'mean_reward': float(mean_reward)
                })
                
                with open(os.path.join(self.log_path, "balance_eval_results.json"), "w") as f:
                    json.dump(self.eval_results, f)
            
            if mean_balance > self.best_mean_balance:
                if self.verbose > 0:
                    print(f"New best mean balance: {mean_balance:.2f}")
                
                self.best_mean_balance = mean_balance
                
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_balance_model"))
                    
        return True
    
    def evaluate_policy(self):
        """Run multiple evaluation episodes and return mean balance, std balance and mean reward."""
        balances = []
        rewards = []
        
        for i in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            lstm_states = None
            
            while not done:
                action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            if hasattr(self.eval_env, 'env') and hasattr(self.eval_env.env, 'balance'):
                balances.append(self.eval_env.env.balance)
            elif hasattr(self.eval_env, 'balance'):
                balances.append(self.eval_env.balance)
            else:
                raise AttributeError("Cannot access 'balance' attribute in environment")
            
            rewards.append(episode_reward)
        
        mean_balance = np.mean(balances)
        std_balance = np.std(balances)
        mean_reward = np.mean(rewards)
        
        return mean_balance, std_balance, mean_reward

def linear_schedule(initial_value, final_value):
    """Linear learning rate schedule."""
    def schedule(progress):
        return final_value + progress * (initial_value - final_value)
    return schedule

def load_data(data_path):
    """Load and prepare dataset."""
    df = pd.read_csv(data_path)
    df.set_index('time', inplace=True)
    
    # Drop unnecessary columns if they exist
    columns_to_drop = ['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df.drop(columns=existing_columns, inplace=True)
    
    df.dropna(inplace=True)
    
    print(f"Dataset shape: {df.shape}, from {df.index[0]} to {df.index[-1]}")
    return df

def create_env(data, env_params):
    """Create and configure the trading environment."""
    env = TradingEnv(data, **env_params)
    return Monitor(env)

def train_model(train_env, val_env, args):
    """Train the PPO model with LSTM policy using full dataset for comprehensive evaluation."""
    # Set up learning rate schedule
    lr_schedule = linear_schedule(args.learning_rate, args.final_learning_rate)
    
    # Create model with LSTM policy
    model = RecurrentPPO(
        "MlpLstmPolicy",  # Using built-in LSTM policy
        train_env,
        learning_rate=lr_schedule,
        n_steps=args.steps_per_update,
        batch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=args.entropy_coef,
        vf_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=False,
        policy_kwargs={
            "optimizer_class": th.optim.Adam,
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
        verbose=0,
        device=args.device,
        seed=args.seed
    )
    
    # Setup callbacks
    callbacks = []

    # Epsilon decay callback
    epsilon_callback = CustomEpsilonCallback(
        start_eps=0.3,
        end_eps=0.01,
        decay_timesteps=int(args.total_timesteps * 0.8)  # Decay over 80% of training
    )
    callbacks.append(epsilon_callback)
    
    # Balance evaluation callback for full dataset
    balance_callback = BalanceEvalCallback(
        val_env,
        best_model_save_path=f"../results/{args.seed}",
        log_path=f"../results/{args.seed}",
        eval_freq=args.eval_freq,
        deterministic=True,
        verbose=1,
        n_eval_episodes=5
    )
    callbacks.append(balance_callback)
    
    render_callback = CustomRenderCallback(
        val_env,
        eval_freq=10000
    )
    callbacks.append(render_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"../models/checkpoints/{args.model_name}",
        name_prefix="ppo_lstm"
    )
    callbacks.append(checkpoint_callback)
    
    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False if hasattr(model, '_last_obs') else True
    )
    
    # Save the final model
    model.save(f"../models/{args.model_name}")
    print(f"Model saved as {args.model_name}")
    
    # Load and return best model based on validation performance
    best_model_path = f"../results/{args.seed}/best_balance_model.zip"
    if os.path.exists(best_model_path):
        print(f"Loading best model based on full dataset performance: {best_model_path}")
        model = RecurrentPPO.load(best_model_path)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train a PPO-LSTM model for trading')
    
    # Required arguments
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name for saving the trained model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the input dataset CSV file')
    
    # Optional arguments with defaults
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                      help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Environment parameters
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial balance for trading')
    parser.add_argument('--bar_count', type=int, default=50,
                      help='Number of bars to use for each observation')
    parser.add_argument('--normalization_window', type=int, default=100,
                      help='Window size for data normalization')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=2000000,
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=1e-5,
                      help='Final learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.2,
                      help='GAE lambda parameter')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                      help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                      help='Value function coefficient')
    parser.add_argument('--minibatch_size', type=int, default=512,
                      help='Minibatch size')
    parser.add_argument('--steps_per_update', type=int, default=2048,
                      help='Number of steps per update')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                      help='Maximum gradient norm for clipping')
    parser.add_argument('--eval_freq', type=int, default=50000,
                      help='Evaluation frequency in timesteps')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("../models/checkpoints", exist_ok=True)
    os.makedirs(f"../results/{args.seed}", exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if args.device == 'cuda':
        th.cuda.manual_seed(args.seed)
    
    # Load data
    data = load_data(args.data_path)
    print("\nCreating environments for training and validation")
    
    # Split data for training (80%) but use full dataset for validation
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    
    print(f"\nEnvironment Configuration:")
    print(f"Training Data: {len(train_data)} bars ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Validation Data: {len(data)} bars (full dataset for comprehensive evaluation)")
    
    # Create environments - training with 80% and random starts, validation with full data and sequential
    env_params = {
        'initial_balance': args.initial_balance,
        'bar_count': args.bar_count,
        'normalization_window': args.normalization_window
    }
    
    train_env = create_env(train_data, {**env_params, 'random_start': True})
    val_env = create_env(data, {**env_params, 'random_start': False})
    
    # Check for existing checkpoints
    checkpoint_dir = f"../models/checkpoints/{args.model_name}"
    current_steps = 0
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
        if checkpoints:
            latest_checkpoint = sorted(
                checkpoints,
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x))
            )[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            import re
            match = re.search(r'_(\d+)_steps', latest_checkpoint)
            if match:
                current_steps = int(match.group(1))
            
            remaining_steps = max(0, args.total_timesteps - current_steps)
            
            print(f"\nFound checkpoint at step {current_steps}/{args.total_timesteps}")
            print(f"Remaining steps: {remaining_steps}")
            
            continue_training = input(f"Continue training from checkpoint? (y/n): ")
            if continue_training.lower() == 'y':
                # Initialize model from checkpoint
                print(f"Loading checkpoint: {checkpoint_path}")
                model = RecurrentPPO.load(
                    checkpoint_path,
                    env=train_env,
                    device=args.device
                )
                
                # Set the learning rate schedule
                lr_schedule = linear_schedule(args.learning_rate, args.final_learning_rate)
                model.learning_rate = lr_schedule
                
                # Update total timesteps to remaining steps
                args.total_timesteps = remaining_steps
                print(f"Continuing training for {remaining_steps} steps...")
                
                # Train for remaining steps
                model = train_model(train_env, val_env, args)
            else:
                print("Starting fresh training...")
                model = train_model(train_env, val_env, args)
        else:
            model = train_model(train_env, val_env, args)
    else:
        model = train_model(train_env, val_env, args)
    
    # Final evaluation - use sequential evaluation to properly handle LSTM states
    print("\nRunning final evaluation...")
    obs, _ = val_env.reset()
    done = False
    total_reward = 0
    lstm_states = None
    
    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            deterministic=True
        )
        obs, reward, terminated, truncated, _ = val_env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"\nFinal Evaluation Results:")
    print(f"Total reward: {total_reward:.2f}")
    val_env.env.render(mode="human")

if __name__ == "__main__":
    import torch as th  # Import here to avoid potential warning messages
    main()
