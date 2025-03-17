import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from trade_environment import TradingEnv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

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
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
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
        # Initialize best balance tracking
        self.best_mean_balance = -float("inf")
        # Evaluation logs
        self.eval_results = []
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        """
        Run evaluation and save best model at regular intervals.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            mean_balance, std_balance, mean_reward = self.evaluate_policy()
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"mean_balance={mean_balance:.2f} +/- {std_balance:.2f}, "
                      f"mean_reward={mean_reward:.2f}")
            
            # Save logs to file
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'mean_balance': float(mean_balance),
                    'std_balance': float(std_balance),
                    'mean_reward': float(mean_reward)
                })
                
                with open(os.path.join(self.log_path, "balance_eval_results.json"), "w") as f:
                    json.dump(self.eval_results, f)
            
            # Save best model
            if mean_balance > self.best_mean_balance:
                if self.verbose > 0:
                    print(f"New best mean balance: {mean_balance:.2f}")
                
                self.best_mean_balance = mean_balance
                
                # Save the model
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
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Get final balance - access through proper attributes
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

class DQNTrainer:
    def __init__(self, train_data, full_data, config=None):
        """Initialize the DQN trainer with training data and full dataset."""
        self.train_data = train_data
        self.full_data = full_data
        
        self.config = config or {
            'base_dir': './../',
            'seed': 42,
            'initial_balance': 10000.0,
            'device': 'cuda',
            'eval_freq': 50000,
            'render_freq': 100000
        }
        self.seed = self.config['seed']
        self.results_dir = f"{self.config['base_dir']}results/{self.seed}"
        self.models_dir = f"{self.config['base_dir']}models"
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create logger
        self.create_logger()
        
    def create_logger(self):
        """Set up logging to track training progress."""
        self.log_file = f"{self.results_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.log_data = []
        
    def save_log(self):
        """Save training logs to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
            
    def create_environments(self, env_params):
        """Create training and full dataset environments."""
        # Training environment with random start
        train_params = {
            'initial_balance': self.config['initial_balance'],
            'random_start': True,
            **env_params
        }
        
        # Full environment for evaluation (no random start)
        full_params = {
            'initial_balance': self.config['initial_balance'],
            'random_start': False,  # Always start from beginning for evaluation
            **env_params
        }
        
        # Create environments
        train_env = Monitor(TradingEnv(self.train_data, **train_params))
        full_env = Monitor(TradingEnv(self.full_data, **full_params))
        
        # Set seeds
        train_env.action_space.seed(self.seed)
        full_env.action_space.seed(self.seed)
        
        return train_env, full_env
        
    def create_model(self, train_env, params):
        """Create a DQN model with given parameters."""
        # Extract policy name
        policy = 'MlpPolicy'
        
        # Create model
        model = DQN(
            policy, 
            train_env, 
            verbose=0, 
            seed=self.seed, 
            device=self.config['device'],
            **params
        )
        return model
        
    def create_callbacks(self, full_env):
        """Create training callbacks using full environment for evaluation."""
        # Balance-based evaluation callback
        balance_eval_callback = BalanceEvalCallback(
            full_env,
            best_model_save_path=self.results_dir,
            log_path=self.results_dir,
            eval_freq=self.config.get('eval_freq', 50000),
            deterministic=True,
            verbose=1,
            n_eval_episodes=5
        )
        
        # Checkpoint callback to save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=max(100000, self.config.get('eval_freq', 50000)),
            save_path=f"{self.results_dir}/checkpoints/",
            name_prefix="dqn_model",
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        
        # Custom rendering callback
        render_callback = CustomRenderCallback(
            full_env,
            eval_freq=self.config.get('render_freq', 100000)
        )
        
        return [balance_eval_callback, checkpoint_callback, render_callback]
    
    def train_model(self, env_params, model_params, timesteps=3000000):
        """Train the model with the fixed parameters."""
        print("Starting training with fixed parameters...")
        
        # Create environments and model
        train_env, full_env = self.create_environments(env_params)
        model = self.create_model(train_env, model_params)
        callbacks = self.create_callbacks(full_env)
        
        # Log the parameters
        self.log_data.append({
            'env_params': env_params,
            'model_params': model_params,
            'timesteps': timesteps
        })
        self.save_log()
        
        # Train the model with multiple callbacks
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
        
        # Save final model
        final_model_path = f"{self.models_dir}/dqn_forex_fixed_{self.seed}"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Load and use best model for evaluation (if it exists)
        best_model_path = f"{self.results_dir}/best_balance_model.zip"
        if os.path.exists(best_model_path):
            print(f"Loading best balance model from {best_model_path}")
            best_model = DQN.load(best_model_path)
            
            # Copy best model to a more descriptive location
            best_saved_path = f"{self.models_dir}/dqn_forex_best_balance_{self.seed}.zip"
            import shutil
            shutil.copy(best_model_path, best_saved_path)
            print(f"Best balance model copied to {best_saved_path}")
            
            # Evaluate on full data using best model
            print("\nEvaluating best balance model on full data...")
            self.evaluate_model(best_model, full_env)
            return best_model, best_saved_path
        else:
            # If no best model was saved, use the final model
            print("\nEvaluating final model on full data...")
            self.evaluate_model(model, full_env)
            return model, final_model_path
            
    def evaluate_model(self, model, test_env):
        """Evaluate the final model performance and plot results."""
        # Reset the test environment and get the initial observation
        obs, info = test_env.reset()

        reward_over_time = []
        balance_over_time = []
        actions_log = []
        done = False

        # Run model through test environment
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_log.append(action)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            done = terminated or truncated
            balance_over_time.append(test_env.env.balance)
            reward_over_time.append(reward)

        # Collect metrics 
        if not hasattr(test_env.env, 'trades'):
            print("No trades data available")
            return
            
        trades_df = pd.DataFrame(test_env.env.trades)    
        current_balance = test_env.env.balance
        initial_balance = test_env.env.initial_balance

        # Calculate performance metrics
        self.calculate_and_print_metrics(
            trades_df, current_balance, initial_balance, balance_over_time)
        
        # Save metrics
        if len(trades_df) > 0:
            trades_df.to_csv(f"{self.results_dir}/test_trades.csv")
            
    def calculate_and_print_metrics(self, trades_df, current_balance, initial_balance, balance_over_time):
        """Calculate and print performance metrics."""
        total_trades = len(trades_df)
        if total_trades == 0:
            print("No trades were executed.")
            return
            
        # Calculate key metrics
        num_tp = trades_df[trades_df["pnl"] > 0.0].shape[0]
        num_sl = trades_df[trades_df["pnl"] < 0.0].shape[0]
        perc_tp = (num_tp / total_trades * 100) if total_trades > 0 else 0.0
        perc_sl = (num_sl / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl_tp = trades_df[trades_df["pnl"] > 0.0]["pnl"].mean() if num_tp > 0 else 0.0
        avg_pnl_sl = trades_df[trades_df["pnl"] < 0.0]["pnl"].mean() if num_sl > 0 else 0.0
        total_return = ((current_balance - initial_balance) / initial_balance) * 100
        expected_value = trades_df["pnl"].mean() if total_trades > 0 else 0.0
        avg_pnl_sl = abs(avg_pnl_sl) if num_sl > 0 else 0.0
        rrr = avg_pnl_tp / avg_pnl_sl if avg_pnl_sl > 0 else 0.0
        num_buy = trades_df[trades_df["position"] == 1].shape[0]
        num_sell = trades_df[trades_df["position"] == -1].shape[0]
        buy_win_rate = (trades_df[(trades_df["position"] == 1) & (trades_df["pnl"] > 0.0)].shape[0] / num_buy * 100) if num_buy > 0 else 0.0
        sell_win_rate = (trades_df[(trades_df["position"] == -1) & (trades_df["pnl"] > 0.0)].shape[0] / num_sell * 100) if num_sell > 0 else 0.0
        total_win_rate = (num_tp / total_trades * 100) if total_trades > 0 else 0.0

        # Kelly Criterion
        def kelly_criterion(win_rate, win_loss_ratio):
            if win_loss_ratio == 0:
                return 0.0  # Avoid division by zero
            return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
            
        # Sharpe Ratio
        def sharpe_ratio(returns, risk_free_rate=0.00, trading_periods=252):
            if len(returns) < 2: 
                return 0.0  # Avoid calculation on insufficient data
            excess_returns = np.array(returns) - (risk_free_rate / trading_periods)
            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
            return sharpe * np.sqrt(trading_periods)

        kelly_criteria = kelly_criterion(total_win_rate / 100.0, rrr) if not np.isnan(rrr) else 0.0

        if total_trades > 0:
            daily_returns = trades_df["pnl"] / initial_balance
            sharpe = sharpe_ratio(daily_returns)
        else:
            sharpe = 0.0

        # Print metrics
        metrics_text = (
            f"Current Balance: {current_balance:.2f}\n"
            f"Total Return: {total_return:.2f}%\n"
            f"Total Trades: {total_trades}\n"
            f"Total Win Rate: {total_win_rate:.2f}%\n"
            f"Long Trades: {num_buy}\n"
            f"Long Win Rate: {buy_win_rate:.2f}%\n"
            f"Short Trades: {num_sell}\n"
            f"Short Win Rate: {sell_win_rate:.2f}%\n"
            f"Average Win: {avg_pnl_tp:.2f}\n"
            f"Average Loss: {avg_pnl_sl:.2f}\n"
            f"Average RRR: {rrr:.2f}\n"
            f"Expected Value: {expected_value:.2f}\n"
            f"Kelly Criterion: {kelly_criteria:.2f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )

        print(metrics_text)
        print(trades_df.head(10))
        print(trades_df.tail(10))
        
        # Save metrics
        with open(f"{self.results_dir}/test_metrics.txt", "w") as f:
            f.write(metrics_text)
        
        # Plot balance over time
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.plot(balance_over_time, linewidth=1, linestyle='-')
        ax.set_xlabel("Time")
        ax.set_ylabel("Balance")
        ax.set_title("Balance Over Time")
        ax.ticklabel_format(style='plain', axis='y')

        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(f"{self.results_dir}/balance_plot.png")
        
    def continue_training(self, checkpoint_path, env_params, model_params, additional_timesteps=3000000):
        """Continue training from a checkpoint."""
        print(f"Continuing training from checkpoint: {checkpoint_path}")
        
        # Create environments
        train_env, full_env = self.create_environments(env_params)
        
        # Load model from checkpoint
        model = DQN.load(
            checkpoint_path,
            env=train_env,
            device=self.config['device']
        )
        
        # Extract the timestep from checkpoint filename
        import re
        checkpoint_filename = os.path.basename(checkpoint_path)
        current_steps = 0
        try:
            # Try to get steps from filename (format: "dqn_model_XXXXX_steps.zip")
            match = re.search(r'_(\d+)_steps', checkpoint_filename)
            if match:
                current_steps = int(match.group(1))
                print(f"Resuming from timestep {current_steps}")
        except:
            print("Could not determine current timestep from filename, starting count from 0")
        
        # Update model parameters if needed
        for param, value in model_params.items():
            if hasattr(model, param):
                setattr(model, param, value)
                
        # Create callbacks
        callbacks = self.create_callbacks(full_env)
        
        # Continue training - now we'll see the correct progress
        model.learn(
            total_timesteps=additional_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False,  # Don't reset the counter
            tb_log_name=f"DQN_continued_{self.seed}"
        )
        
        # Save final model
        final_model_path = f"{self.models_dir}/dqn_forex_continued_{self.seed}"
        model.save(final_model_path)
        print(f"Continued model saved to {final_model_path}")
        
        # Evaluate best or final model
        best_model_path = f"{self.results_dir}/best_balance_model.zip"
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            best_model = DQN.load(best_model_path)
            
            # Copy best model to a more descriptive location
            best_saved_path = f"{self.models_dir}/dqn_forex_best_{self.seed}.zip"
            import shutil
            shutil.copy(best_model_path, best_saved_path)
            print(f"Best model copied to {best_saved_path}")
            
            # Evaluate on full data using best model
            print("\nEvaluating best model on full data...")
            self.evaluate_model(best_model, full_env)
            return best_model, best_saved_path
        else:
            # If no best model was saved, use the final model
            print("\nEvaluating final model on full data...")
            self.evaluate_model(model, full_env)
            return model, final_model_path


def load_data():
    """Load and prepare training and full dataset according to specifications."""
    # Load data from CSV
    RATES_CSV_PATH = "../data/BTCUSDm_60min.csv"
    df = pd.read_csv(RATES_CSV_PATH)
    df.set_index('time', inplace=True)
    df.drop(columns=['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower'], inplace=True)
    
    # Print data information
    start_datetime = df.index[0]
    end_datetime = df.index[-1]
    print(f"Data collected from {start_datetime} to {end_datetime}")
    print(df.tail())
    print('\n')
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    # Split data: 80% for training, but keep full dataset too
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    full_data = df  # Keep the full dataset
    
    print(f"Train data shape: {train_data.shape}, from {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Full data shape: {full_data.shape}, from {full_data.index[0]} to {full_data.index[-1]}")
    
    return train_data, full_data


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_value = 65478  # Using the ID from the results folder for reproducibility
    print(f"Using seed: {seed_value}")
    
    # Load data
    train_data, full_data = load_data()
    
    # Fixed parameters from the best trial
    env_params = {
        'bar_count': 50,
        'normalization_window': 100
    }
    
    model_params = {
        'learning_rate': 8.683466805193546e-06,
        'batch_size': 96,
        'gamma': 0.8724738178987883,
        'buffer_size': 25000,
        'target_update_interval': 1500,
        'exploration_fraction': 0.3772233918444839,
        'exploration_final_eps': 0.061288914160504124
    }
    
    print("Fixed parameters for training:")
    print(f"Environment parameters: {env_params}")
    print(f"Model parameters: {model_params}")
    
    # Create trainer with train and full data
    trainer = DQNTrainer(train_data, full_data, config={
        'base_dir': './../',
        'seed': seed_value,
        'initial_balance': 10000.0,
        'device': 'cuda',
        'eval_freq': 50000,
        'render_freq': 100000
    })
    
    # For the main script section:
    checkpoint_dir = f"./../results/{seed_value}/checkpoints/"
    total_steps = 30000000  # Total steps we want to train
    current_steps = 0

    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
        if checkpoints:
            # Sort by modification time to get the latest checkpoint
            latest_checkpoint = sorted(
                checkpoints,
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x))
            )[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            # Extract step number from filename
            import re
            match = re.search(r'_(\d+)_steps', latest_checkpoint)
            if match:
                current_steps = int(match.group(1))
            
            # Calculate remaining steps
            remaining_steps = max(0, total_steps - current_steps)
            
            # Ask user if they want to continue
            continue_training = input(f"Found checkpoint at step {current_steps}/{total_steps}. Continue training for {remaining_steps} more steps? (y/n): ")
            if continue_training.lower() == 'y':
                model, model_path = trainer.continue_training(
                    checkpoint_path,
                    env_params=env_params,
                    model_params=model_params,
                    additional_timesteps=remaining_steps  # Train remaining steps
                )
                print(f"Training continued and completed. Final model saved to {model_path}")
                exit(0)

    # Start new training if no checkpoint or user chose not to continue
    model, model_path = trainer.train_model(
        env_params=env_params,
        model_params=model_params,
        timesteps=total_steps  # Use total steps
    )
    
    print(f"Training complete. Final model saved to {model_path}")