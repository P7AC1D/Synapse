import os
import json
import numpy as np
import pandas as pd
import optuna
import joblib
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from trade_environment import TradingEnv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        self.best_mean_balance = -float("inf")
        self.eval_results = []
        
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Run evaluation and save best model at regular intervals."""
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
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
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

class ModelTrainer:
    def __init__(self, model_type, train_data, full_data, config=None):
        """Initialize the model trainer with training data and full dataset."""
        if model_type not in ['DQN', 'PPO']:
            raise ValueError("model_type must be either 'DQN' or 'PPO'")
            
        self.model_type = model_type
        self.model_class = DQN if model_type == 'DQN' else PPO
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
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
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
        train_params = {
            'initial_balance': self.config['initial_balance'],
            'random_start': True,
            **env_params
        }
        
        full_params = {
            'initial_balance': self.config['initial_balance'],
            'random_start': False,
            **env_params
        }
        
        train_env = Monitor(TradingEnv(self.train_data, **train_params))
        full_env = Monitor(TradingEnv(self.full_data, **full_params))
        
        train_env.action_space.seed(self.seed)
        full_env.action_space.seed(self.seed)
        
        return train_env, full_env
        
    def create_model(self, train_env, params):
        """Create a model with given parameters."""
        return self.model_class(
            'MlpPolicy', 
            train_env, 
            verbose=0, 
            seed=self.seed, 
            device=self.config['device'],
            **params
        )
        
    def create_callbacks(self, full_env):
        """Create training callbacks using full environment for evaluation."""
        balance_eval_callback = BalanceEvalCallback(
            full_env,
            best_model_save_path=self.results_dir,
            log_path=self.results_dir,
            eval_freq=self.config.get('eval_freq', 50000),
            deterministic=True,
            verbose=1,
            n_eval_episodes=5
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(100000, self.config.get('eval_freq', 50000)),
            save_path=f"{self.results_dir}/checkpoints/",
            name_prefix=f"{self.model_type.lower()}_model",
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        
        render_callback = CustomRenderCallback(
            full_env,
            eval_freq=self.config.get('render_freq', 100000)
        )
        
        return [balance_eval_callback, checkpoint_callback, render_callback]
    
    def evaluate_model_balance(self, model, test_env, n_episodes=5):
        """Evaluate model based on final account balance rather than reward."""
        final_balances = []
        
        for i in range(n_episodes):
            obs, _ = test_env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
            
            # Store final balance
            if hasattr(test_env, 'env') and hasattr(test_env.env, 'balance'):
                final_balances.append(test_env.env.balance)
            elif hasattr(test_env, 'balance'):
                final_balances.append(test_env.balance)
        
        # Return the average final balance across episodes
        if final_balances:
            avg_balance = sum(final_balances) / len(final_balances)
            return avg_balance
        else:
            print("Warning: Could not access balance attribute in environment")
            return 0.0
            
    def broad_hp_search(self, n_trials=50):
        """Conduct broad hyperparameter search."""
        print("Starting broad hyperparameter search...")
        
        def objective(trial):
            # Environment parameters
            env_params = {
                'bar_count': 50,  # Fixed for broad search
                'normalization_window': 100  # Fixed for broad search
            }
            
            if self.model_type == 'DQN':
                model_params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                    'gamma': trial.suggest_float('gamma', 0.8, 0.99),
                    'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
                    'target_update_interval': trial.suggest_categorical('target_update_interval', [1000, 2000, 5000]),
                    'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
                    'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1)
                }
            else:  # PPO parameters
                model_params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                    'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.999),
                    'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                    'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
                    'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
                    'n_epochs': trial.suggest_int('n_epochs', 5, 20)
                }
            
            # Create environments and model
            train_env, full_env = self.create_environments(env_params)
            model = self.create_model(train_env, model_params)
            callbacks = self.create_callbacks(full_env)
            
            # Train with limited timesteps for hyperparameter search
            try:
                model.learn(total_timesteps=100000, callback=callbacks)
                final_balance = self.evaluate_model_balance(model, full_env, n_episodes=5)
                print(f"Trial {trial.number} completed with full dataset balance: {final_balance:.2f}")
                
                self.log_data.append({
                    'trial': trial.number,
                    'params': {**env_params, **model_params},
                    'final_balance': float(final_balance)
                })
                self.save_log()
                
                return final_balance
                
            except Exception as e:
                print(f"Error in trial {trial.number}: {e}")
                return float('-inf')
                
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("Broad search best parameters:", study.best_params)
        print("Best mean balance:", study.best_value)
        
        # Save study
        joblib.dump(study, f"{self.results_dir}/broad_study.pkl")
        
        return study.best_params, study.best_value
        
    def narrow_hp_search(self, best_broad_params, n_trials=30):
        """Conduct narrow hyperparameter search around best parameters."""
        print("Starting narrow hyperparameter search...")
        
        def objective(trial):
            # Environment parameters
            env_params = {
                'bar_count': 50,
                'normalization_window': 100
            }
            
            # Model hyperparameters
            if self.model_type == 'DQN':
                model_params = {
                    'learning_rate': trial.suggest_float('learning_rate', 
                                                      best_broad_params['learning_rate'] * 0.5,
                                                      best_broad_params['learning_rate'] * 2.0, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', 
                                                         [int(best_broad_params['batch_size'] * 0.5),
                                                          best_broad_params['batch_size'],
                                                          int(best_broad_params['batch_size'] * 1.5)]),
                    'gamma': trial.suggest_float('gamma', 
                                               max(0.8, best_broad_params['gamma'] - 0.05),
                                               min(0.995, best_broad_params['gamma'] + 0.05)),
                    'buffer_size': trial.suggest_categorical('buffer_size', 
                                                          [int(best_broad_params['buffer_size'] * 0.5),
                                                           best_broad_params['buffer_size'],
                                                           int(best_broad_params['buffer_size'] * 1.5)]),
                    'target_update_interval': trial.suggest_categorical('target_update_interval',
                                                                     [int(best_broad_params['target_update_interval'] * 0.5),
                                                                      best_broad_params['target_update_interval'],
                                                                      int(best_broad_params['target_update_interval'] * 1.5)]),
                    'exploration_fraction': trial.suggest_float('exploration_fraction',
                                                             max(0.05, best_broad_params['exploration_fraction'] - 0.1),
                                                             min(0.7, best_broad_params['exploration_fraction'] + 0.1)),
                    'exploration_final_eps': trial.suggest_float('exploration_final_eps',
                                                              max(0.005, best_broad_params['exploration_final_eps'] * 0.5),
                                                              min(0.2, best_broad_params['exploration_final_eps'] * 2.0))
                }
            else:  # PPO parameters
                model_params = {
                    'learning_rate': trial.suggest_float('learning_rate',
                                                      best_broad_params['learning_rate'] * 0.5,
                                                      best_broad_params['learning_rate'] * 2.0, log=True),
                    'n_steps': trial.suggest_categorical('n_steps',
                                                      [int(best_broad_params['n_steps'] * 0.5),
                                                       best_broad_params['n_steps'],
                                                       int(best_broad_params['n_steps'] * 1.5)]),
                    'batch_size': trial.suggest_categorical('batch_size',
                                                         [int(best_broad_params['batch_size'] * 0.5),
                                                          best_broad_params['batch_size'],
                                                          int(best_broad_params['batch_size'] * 1.5)]),
                    'gamma': trial.suggest_float('gamma',
                                               max(0.8, best_broad_params['gamma'] - 0.05),
                                               min(0.999, best_broad_params['gamma'] + 0.05)),
                    'gae_lambda': trial.suggest_float('gae_lambda',
                                                    max(0.8, best_broad_params['gae_lambda'] - 0.05),
                                                    min(0.999, best_broad_params['gae_lambda'] + 0.05)),
                    'clip_range': trial.suggest_float('clip_range',
                                                    max(0.05, best_broad_params['clip_range'] - 0.1),
                                                    min(0.5, best_broad_params['clip_range'] + 0.1)),
                    'ent_coef': trial.suggest_float('ent_coef',
                                                  max(0.0, best_broad_params['ent_coef'] - 0.005),
                                                  min(0.02, best_broad_params['ent_coef'] + 0.005)),
                    'vf_coef': trial.suggest_float('vf_coef',
                                                 max(0.05, best_broad_params['vf_coef'] - 0.2),
                                                 min(0.95, best_broad_params['vf_coef'] + 0.2)),
                    'n_epochs': trial.suggest_int('n_epochs',
                                               max(3, best_broad_params['n_epochs'] - 5),
                                               min(25, best_broad_params['n_epochs'] + 5))
                }
            
            # Create environments and model
            train_env, full_env = self.create_environments(env_params)
            model = self.create_model(train_env, model_params)
            callbacks = self.create_callbacks(full_env)
            
            try:
                model.learn(total_timesteps=200000, callback=callbacks)
                final_balance = self.evaluate_model_balance(model, full_env, n_episodes=5)
                print(f"Trial {trial.number} completed with final balance: {final_balance:.2f}")
                
                self.log_data.append({
                    'trial': trial.number + 1000,  # Offset to distinguish from broad search
                    'params': {**env_params, **model_params},
                    'final_balance': float(final_balance)
                })
                self.save_log()
                
                return final_balance
                
            except Exception as e:
                print(f"Error in trial {trial.number}: {e}")
                return float('-inf')
                
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("Narrow search best parameters:", study.best_params)
        print("Best mean balance:", study.best_value)
        
        # Save study
        joblib.dump(study, f"{self.results_dir}/narrow_study.pkl")
        
        return study.best_params, study.best_value
        
    def run_full_pipeline(self, broad_trials=30, narrow_trials=20, final_timesteps=5000000):
        """Run the full hyperparameter optimization and training pipeline."""
        # Check for existing hyperparameter optimization results
        best_params_path = f"{self.results_dir}/best_params.json"
        
        if os.path.exists(best_params_path):
            print("Found existing hyperparameter optimization results")
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
        else:
            # Step 1: Broad hyperparameter search
            best_broad_params, _ = self.broad_hp_search(n_trials=broad_trials)
            
            # Step 2: Narrow hyperparameter search
            best_params, _ = self.narrow_hp_search(best_broad_params, n_trials=narrow_trials)
            
            # Save best parameters
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
        
        # Step 3: Final training
        model, model_path = self.train_model(
            env_params={'bar_count': 50, 'normalization_window': 100},
            model_params=best_params,
            timesteps=final_timesteps
        )
        
        return model, model_path
        
    def train_model(self, env_params, model_params, timesteps=3000000):
        """Train the model with the fixed parameters."""
        print("Starting training with fixed parameters...")
        
        train_env, full_env = self.create_environments(env_params)
        model = self.create_model(train_env, model_params)
        callbacks = self.create_callbacks(full_env)
        
        self.log_data.append({
            'env_params': env_params,
            'model_params': model_params,
            'timesteps': timesteps
        })
        self.save_log()
        
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
        
        final_model_path = f"{self.models_dir}/{self.model_type.lower()}_forex_fixed_{self.seed}"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        best_model_path = f"{self.results_dir}/best_balance_model.zip"
        if os.path.exists(best_model_path):
            print(f"Loading best balance model from {best_model_path}")
            best_model = self.model_class.load(best_model_path)
            
            best_saved_path = f"{self.models_dir}/{self.model_type.lower()}_forex_best_balance_{self.seed}.zip"
            import shutil
            shutil.copy(best_model_path, best_saved_path)
            print(f"Best balance model copied to {best_saved_path}")
            
            print("\nEvaluating best balance model on full data...")
            self.evaluate_model(best_model, full_env)
            return best_model, best_saved_path
        else:
            print("\nEvaluating final model on full data...")
            self.evaluate_model(model, full_env)
            return model, final_model_path
            
    def evaluate_model(self, model, test_env):
        """Evaluate the final model performance and plot results."""
        obs, info = test_env.reset()

        reward_over_time = []
        balance_over_time = []
        actions_log = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_log.append(action)
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            done = terminated or truncated
            balance_over_time.append(test_env.env.balance)
            reward_over_time.append(reward)

        if not hasattr(test_env.env, 'trades'):
            print("No trades data available")
            return
            
        trades_df = pd.DataFrame(test_env.env.trades)    
        current_balance = test_env.env.balance
        initial_balance = test_env.env.initial_balance

        self.calculate_and_print_metrics(trades_df, current_balance, initial_balance, balance_over_time)
        
        if len(trades_df) > 0:
            trades_df.to_csv(f"{self.results_dir}/test_trades.csv")
            
    def calculate_and_print_metrics(self, trades_df, current_balance, initial_balance, balance_over_time):
        """Calculate and print performance metrics."""
        total_trades = len(trades_df)
        if total_trades == 0:
            print("No trades were executed.")
            return
            
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

        def kelly_criterion(win_rate, win_loss_ratio):
            if win_loss_ratio == 0:
                return 0.0
            return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
            
        def sharpe_ratio(returns, risk_free_rate=0.00, trading_periods=252):
            if len(returns) < 2: 
                return 0.0
            excess_returns = np.array(returns) - (risk_free_rate / trading_periods)
            sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
            return sharpe * np.sqrt(trading_periods)

        kelly_criteria = kelly_criterion(total_win_rate / 100.0, rrr) if not np.isnan(rrr) else 0.0

        if total_trades > 0:
            daily_returns = trades_df["pnl"] / initial_balance
            sharpe = sharpe_ratio(daily_returns)
        else:
            sharpe = 0.0

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
        
        with open(f"{self.results_dir}/test_metrics.txt", "w") as f:
            f.write(metrics_text)
        
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
        
        train_env, full_env = self.create_environments(env_params)
        
        model = self.model_class.load(
            checkpoint_path,
            env=train_env,
            device=self.config['device']
        )
        
        import re
        checkpoint_filename = os.path.basename(checkpoint_path)
        current_steps = 0
        try:
            match = re.search(r'_(\d+)_steps', checkpoint_filename)
            if match:
                current_steps = int(match.group(1))
                print(f"Resuming from timestep {current_steps}")
        except:
            print("Could not determine current timestep from filename, starting count from 0")
        
        for param, value in model_params.items():
            if hasattr(model, param):
                setattr(model, param, value)
                
        callbacks = self.create_callbacks(full_env)
        
        model.learn(
            total_timesteps=additional_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False,
            tb_log_name=f"{self.model_type}_continued_{self.seed}"
        )
        
        final_model_path = f"{self.models_dir}/{self.model_type.lower()}_forex_continued_{self.seed}"
        model.save(final_model_path)
        print(f"Continued model saved to {final_model_path}")
        
        best_model_path = f"{self.results_dir}/best_balance_model.zip"
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            best_model = self.model_class.load(best_model_path)
            
            best_saved_path = f"{self.models_dir}/{self.model_type.lower()}_forex_best_{self.seed}.zip"
            import shutil
            shutil.copy(best_model_path, best_saved_path)
            print(f"Best model copied to {best_saved_path}")
            
            print("\nEvaluating best model on full data...")
            self.evaluate_model(best_model, full_env)
            return best_model, best_saved_path
        else:
            print("\nEvaluating final model on full data...")
            self.evaluate_model(model, full_env)
            return model, final_model_path

def load_data():
    """Load and prepare training and full dataset according to specifications."""
    RATES_CSV_PATH = "../data/BTCUSDm_60min.csv"
    df = pd.read_csv(RATES_CSV_PATH)
    df.set_index('time', inplace=True)
    df.drop(columns=['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower'], inplace=True)
    
    start_datetime = df.index[0]
    end_datetime = df.index[-1]
    print(f"Data collected from {start_datetime} to {end_datetime}")
    print(df.tail())
    print('\n')
    
    df.dropna(inplace=True)
    
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    full_data = df
    
    print(f"Train data shape: {train_data.shape}, from {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Full data shape: {full_data.shape}, from {full_data.index[0]} to {full_data.index[-1]}")
    
    return train_data, full_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a DQN or PPO model for trading.')
    parser.add_argument('--model_type', type=str, choices=['DQN', 'PPO'], required=True,
                      help='Type of model to train (DQN or PPO)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--optimize', action='store_true',
                      help='Run hyperparameter optimization before training')
    parser.add_argument('--broad_trials', type=int, default=30,
                      help='Number of trials for broad hyperparameter search')
    parser.add_argument('--narrow_trials', type=int, default=20,
                      help='Number of trials for narrow hyperparameter search')
    parser.add_argument('--timesteps', type=int, default=30000000,
                      help='Total timesteps for training')
    args = parser.parse_args()
    
    print(f"Training {args.model_type} model with seed: {args.seed}")
    
    train_data, full_data = load_data()
    
    trainer = ModelTrainer(args.model_type, train_data, full_data, config={
        'base_dir': './../',
        'seed': args.seed,
        'initial_balance': 10000.0,
        'device': args.device,
        'eval_freq': 50000,
        'render_freq': 100000
    })

    print(f"Using device: {args.device}")
    
    if args.optimize:
        # Run the full pipeline with hyperparameter optimization
        model, model_path = trainer.run_full_pipeline(
            broad_trials=args.broad_trials,
            narrow_trials=args.narrow_trials,
            final_timesteps=args.timesteps
        )
    else:
        # Check for checkpoints
        checkpoint_dir = f"./../results/{args.seed}/checkpoints/"
        current_steps = 0

        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints,
                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x))
                )[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                
                match = re.search(r'_(\d+)_steps', latest_checkpoint)
                if match:
                    current_steps = int(match.group(1))
                
                remaining_steps = max(0, args.timesteps - current_steps)
                
                continue_training = input(f"Found checkpoint at step {current_steps}/{args.timesteps}. Continue training for {remaining_steps} more steps? (y/n): ")
                if continue_training.lower() == 'y':
                    # Check for best parameters
                    best_params_path = f"{trainer.results_dir}/best_params.json"
                    if os.path.exists(best_params_path):
                        with open(best_params_path, 'r') as f:
                            model_params = json.load(f)
                    else:
                        model_params = {
                            # DQN default parameters
                            "learning_rate": 1e-4,
                            "batch_size": 128,
                            "gamma": 0.9,
                            "buffer_size": 50000,
                            "target_update_interval": 1000,
                            "exploration_fraction": 0.25,
                            "exploration_final_eps": 0.01
                        } if args.model_type == 'DQN' else {
                            # PPO default parameters
                            "learning_rate": 3e-4,
                            "n_steps": 2048,
                            "batch_size": 64,
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "ent_coef": 0.01,
                            "vf_coef": 0.5,
                            "n_epochs": 10,
                            "clip_range": 0.2
                        }
                    
                    env_params = {'bar_count': 50, 'normalization_window': 100}
                    model, model_path = trainer.continue_training(
                        checkpoint_path,
                        env_params=env_params,
                        model_params=model_params,
                        additional_timesteps=remaining_steps
                    )
                    print(f"Training continued and completed. Final model saved to {model_path}")
                    exit(0)

        # Start new training with default or saved parameters
        best_params_path = f"{trainer.results_dir}/best_params.json"
        if os.path.exists(best_params_path):
            print("Using previously optimized parameters")
            with open(best_params_path, 'r') as f:
                model_params = json.load(f)
        else:
            print("Using default parameters")
            model_params = {
                # DQN default parameters
                "learning_rate": 1e-4,
                "batch_size": 128,
                "gamma": 0.9,
                "buffer_size": 50000,
                "target_update_interval": 1000,
                "exploration_fraction": 0.25,
                "exploration_final_eps": 0.01
            } if args.model_type == 'DQN' else {
                # PPO default parameters
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "n_epochs": 10,
                "clip_range": 0.2
            }

        env_params = {'bar_count': 50, 'normalization_window': 100}
        model, model_path = trainer.train_model(
            env_params=env_params,
            model_params=model_params,
            timesteps=args.timesteps
        )
        print(f"Training complete. Final model saved to {model_path}")
