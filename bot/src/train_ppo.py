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
from gymnasium import spaces

class CustomEpsilonCallback(BaseCallback):
    """Custom callback for epsilon decay during training"""
    def __init__(self, start_eps=0.2, end_eps=0.02, decay_timesteps=1600000):
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
    """Optimized evaluation callback with enhanced progress tracking and comprehensive evaluation."""
    def __init__(self, eval_env, train_data, val_data, eval_freq=100000, best_model_save_path=None, 
                 log_path=None, deterministic=True, verbose=1, iteration=0):
        super(UnifiedEvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.eval_results = []
        self.last_time_trigger = 0
        self.iteration = iteration
        
        # Store separate datasets
        self.train_data = train_data
        self.val_data = val_data
        
        # Create combined evaluation environment
        self.combined_data = pd.concat([train_data, val_data])
        env_params = {
            'initial_balance': eval_env.env.initial_balance,
            'balance_per_lot': eval_env.env.BALANCE_PER_LOT,
            'random_start': False
        }
        self.combined_env = Monitor(TradingEnv(self.combined_data, **env_params))
        
        # Initialize tracking metrics
        self.best_score = -float("inf")
        self.best_metrics = {}
        self.max_drawdown = 0.0
        
        # Back up raw data for reference
        if hasattr(self.eval_env, 'env'):
            self.eval_env.env.raw_data_backup = self.eval_env.env.raw_data.copy()
        else:
            self.eval_env.raw_data_backup = self.eval_env.raw_data.copy()
            
    def _run_eval_episode(self, env) -> Dict[str, float]:
        """Run a complete evaluation episode on given environment."""
        obs, _ = env.reset()
        done = False
        lstm_states = None
        running_balance = env.env.initial_balance
        max_balance = running_balance
        episode_reward = 0
        
        while not done:
            action, lstm_states = self.model.predict(
                obs, state=lstm_states, deterministic=self.deterministic
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Track running metrics
            running_balance = env.env.balance
            max_balance = max(max_balance, running_balance)
        
        # Calculate metrics
        total_return = (running_balance - env.env.initial_balance) / env.env.initial_balance
        max_drawdown = 0.0
        if max_balance > env.env.initial_balance:
            max_drawdown = (max_balance - running_balance) / max_balance
            
        # Calculate trade quality metrics
        trades = env.env.trades
        win_rate = len([t for t in trades if t['pnl'] > 0]) / max(len(trades), 1)
        avg_profit = sum(t['pnl'] for t in trades) / max(len(trades), 1)
        
        return {
            'return': total_return,
            'max_drawdown': max_drawdown,
            'reward': episode_reward,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'balance': running_balance,
            'trades': trades
        }
        
    def _calculate_trade_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall trade quality score."""
        win_rate_score = metrics['win_rate']
        profit_factor = max(0, metrics['avg_profit']) / (abs(min(0, metrics['avg_profit'])) + 1e-8)
        drawdown_penalty = max(0, 1 - metrics['max_drawdown'] * 2)
        
        return (win_rate_score * 0.4 + 
                min(profit_factor, 3) / 3 * 0.4 + 
                drawdown_penalty * 0.2)
        
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        
        if self.log_path:
            os.makedirs(log_path, exist_ok=True)
        
    def _evaluate_performance(self) -> Dict[str, Dict[str, float]]:
        """Run comprehensive evaluation on all datasets."""
        # Evaluate on validation set
        val_metrics = self._run_eval_episode(self.eval_env)
        
        # Evaluate on combined dataset
        combined_metrics = self._run_eval_episode(self.combined_env)
        
        # Calculate consistency score
        consistency_score = val_metrics['return'] / (combined_metrics['return'] + 1e-8)
        
        # Calculate trade quality scores
        val_quality = self._calculate_trade_quality(val_metrics)
        combined_quality = self._calculate_trade_quality(combined_metrics)
        
        return {
            'validation': val_metrics,
            'combined': combined_metrics,
            'scores': {
                'consistency': consistency_score,
                'val_quality': val_quality,
                'combined_quality': combined_quality
            }
        }
    
    def _should_save_model(self, metrics: Dict[str, Dict[str, float]]) -> bool:
        """Determine if current model should be saved as best."""
        combined = metrics['combined']
        scores = metrics['scores']
        
        # Calculate composite score
        score = (
            combined['return'] * 0.4 +                # Weight overall return
            -combined['max_drawdown'] * 0.3 +        # Penalize drawdowns
            scores['consistency'] * 0.2 +            # Reward consistency
            scores['combined_quality'] * 0.1         # Consider trade quality
        )
        
        if score > self.best_score:
            self.best_score = score
            self.best_metrics = metrics
            return True
        return False
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Run comprehensive evaluation
            metrics = self._evaluate_performance()
            combined = metrics['combined']
            val = metrics['validation']
            
            if self.verbose > 0:
                print(f"\n===== Evaluation at timesteps={self.num_timesteps} =====")
                print(f"Combined Dataset Metrics:")
                print(f"  Balance: {combined['balance']:.2f}")
                print(f"  Return: {combined['return']*100:.2f}%")
                print(f"  Max Drawdown: {combined['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {combined['win_rate']*100:.2f}%")
                print(f"\nValidation Set Metrics:")
                print(f"  Balance: {val['balance']:.2f}")
                print(f"  Return: {val['return']*100:.2f}%")
                print(f"  Max Drawdown: {val['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {val['win_rate']*100:.2f}%")
            
            if self.log_path is not None:
                self.eval_results.append({
                    'timesteps': self.num_timesteps,
                    'combined': {
                        'balance': float(combined['balance']),
                        'return': float(combined['return']),
                        'max_drawdown': float(combined['max_drawdown']),
                        'win_rate': float(combined['win_rate'])
                    },
                    'validation': {
                        'balance': float(val['balance']),
                        'return': float(val['return']),
                        'max_drawdown': float(val['max_drawdown']),
                        'win_rate': float(val['win_rate'])
                    }
                })
                
                iteration_file = os.path.join(self.log_path, f"eval_results_iter_{self.iteration}.json")
                with open(iteration_file, "w") as f:
                    json.dump(self.eval_results, f, indent=2)
                    
                combined_file = os.path.join(self.log_path, "eval_results_all.json")
                try:
                    with open(combined_file, "r") as f:
                        all_results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    all_results = {}
                    
                eval_env = self.eval_env
                while hasattr(eval_env, 'env'):
                    eval_env = eval_env.env
                    if isinstance(eval_env, TradingEnv):
                        break

                # Calculate drawdown metrics
                running_balance = eval_env.initial_balance
                max_balance = eval_env.initial_balance
                period_max_drawdown = 0.0
                
                # Calculate running drawdown using trade history
                for trade in eval_env.trades:
                    running_balance += trade['pnl']
                    max_balance = max(max_balance, running_balance)
                    if max_balance > 0:
                        current_drawdown = (max_balance - running_balance) / max_balance
                        period_max_drawdown = max(period_max_drawdown, current_drawdown)

                # Update historical max drawdown
                self.max_drawdown = max(self.max_drawdown, period_max_drawdown)

                # Calculate basic metrics
                active_positions = len(eval_env.positions)
                num_winning_trades = sum(1 for t in eval_env.trades if t['pnl'] > 0)
                num_losing_trades = sum(1 for t in eval_env.trades if t['pnl'] < 0)
                
                try:
                    period_start = str(eval_env.original_index[0])
                    period_end = str(eval_env.original_index[-1])
                except (AttributeError, IndexError) as e:
                    period_start = period_end = "NA"
                    print(f"Warning: Could not get period timestamps: {str(e)}")

                # Print drawdown information
                print("\n===== Drawdown Analysis =====")
                print(f"Period Max Drawdown: {period_max_drawdown*100:.2f}%")
                print(f"Historical Max Drawdown: {self.max_drawdown*100:.2f}%")

                period_info = {
                    'results': self.eval_results,
                    'iteration': self.iteration,
                    'balance': float(eval_env.balance),
                    'total_trades': len(eval_env.trades),
                    'active_positions': active_positions,
                    'win_count': num_winning_trades,
                    'loss_count': num_losing_trades,
                    'win_rate': (num_winning_trades / len(eval_env.trades) * 100) if eval_env.trades else 0.0,
                    'period_start': period_start,
                    'period_end': period_end,
                    'grid_metrics': eval_env.grid_metrics,
                    'max_drawdown': period_max_drawdown * 100,
                    'historical_max_drawdown': self.max_drawdown * 100
                }

                all_results[f"iteration_{self.iteration}"] = period_info
                
                with open(combined_file, "w") as f:
                    json.dump(all_results, f, indent=2)
            
            # Check if model should be saved as best
            if self._should_save_model(metrics) and self.best_model_save_path is not None:
                model_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(model_path)
                
                print(f"\n=== New Best Model Saved ===")
                print(f"Combined Return: {metrics['combined']['return']*100:.2f}%")
                print(f"Validation Return: {metrics['validation']['return']*100:.2f}%")
                print(f"Consistency Score: {metrics['scores']['consistency']:.2f}")
                print(f"Trade Quality: {metrics['scores']['combined_quality']:.2f}")
            
            # Print final scores summary
            print("\n===== Final Performance Metrics =====")
            print(f"Combined Dataset Score: {metrics['combined_quality']:.3f}")
            print(f"Validation Score: {metrics['validation_quality']:.3f}")
            print(f"Overall Score: {metrics['combined']['return'] * 0.4 - metrics['combined']['max_drawdown'] * 0.3 + metrics['scores']['consistency'] * 0.2 + metrics['scores']['combined_quality'] * 0.1:.3f}")
            
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
        end_fraction=0.95
    )
    
    # Configure policy for discrete action space
    policy_kwargs = {
        "optimizer_class": th.optim.AdamW,
        "lstm_hidden_size": 128,      # Increased capacity
        "n_lstm_layers": 2,           # Deeper LSTM
        "shared_lstm": True,          # Use shared LSTM
        "enable_critic_lstm": False,  # Disable separate critic LSTM
        "net_arch": {
            "pi": [64, 32],          # Deeper actor network
            "vf": [64, 32]           # Deeper critic network
        },
        "optimizer_kwargs": {
            "eps": 1e-5,
            "weight_decay": 1e-4
        }
    }
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        learning_rate=8e-4,          # Higher learning rate for faster adaptation
        n_steps=256,                 # Keep stride length
        batch_size=256,              # Keep batch size
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,              # Increased from 0.2 for more aggressive updates
        clip_range_vf=0.3,           # Match policy clip range
        ent_coef=0.05,              # Higher entropy to encourage exploration
        vf_coef=0.6,                # Slightly higher value function importance
        max_grad_norm=0.6,          # Allow slightly larger gradients
        use_sde=False,              # No SDE for discrete actions
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=args.device,
        seed=args.seed
    )
    
    callbacks = []
    
    # Configure epsilon exploration for discrete actions
    epsilon_callback = CustomEpsilonCallback(
        start_eps=0.4,     # Higher initial exploration
        end_eps=0.15,      # Higher minimum exploration
        decay_timesteps=int(args.total_timesteps * 0.7)  # Faster decay for more aggressive exploration
    )
    callbacks.append(epsilon_callback)
    
    # Add evaluation callback
    unified_callback = UnifiedEvalCallback(
        val_env,
        train_data=train_env.env.raw_data,
        val_data=val_env.env.raw_data,
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
    
    while training_start + initial_window + step_size <= total_periods:
        iteration = training_start // step_size
        
        train_end = training_start + initial_window
        val_end = min(train_end + step_size, total_periods)
        
        train_data = data.iloc[training_start:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        
        train_data.index = data.index[training_start:train_end]
        val_data.index = data.index[train_end:val_end]
        
        print(f"\n=== Training Period: {train_data.index[0]} to {train_data.index[-1]} ===")
        print(f"Validation Period: {val_data.index[0]} to {val_data.index[-1]} ===")
        print(f"Walk-forward Iteration: {iteration}")
        
        env_params = {
            'initial_balance': args.initial_balance,
            'balance_per_lot': args.balance_per_lot
        }
        
        train_env = Monitor(TradingEnv(train_data, **{**env_params, 'random_start': True}))
        val_env = Monitor(TradingEnv(val_data, **{**env_params, 'random_start': False}))
        
        period_timesteps = base_timesteps
        
        if model is None:
            model = train_model(train_env, val_env, args, iteration=iteration)
        else:
            print(f"\nContinuing training with existing model...")
            print(f"Training timesteps: {period_timesteps}")
            args.learning_rate = args.learning_rate * 0.95
            model.set_env(train_env)
            
            callbacks = []
            
            epsilon_callback = CustomEpsilonCallback(
                start_eps=0.15,  # Keep exploration high in later iterations
                end_eps=0.05,    # Maintain minimum exploration
                decay_timesteps=int(period_timesteps * 0.95)  # Even slower decay for continued learning
            )
            callbacks.append(epsilon_callback)
            
            unified_callback = UnifiedEvalCallback(
                val_env,
                train_data=train_env.env.raw_data,
                val_data=val_env.env.raw_data,
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
    parser.add_argument('--initial_window', type=int, default=28,
                      help='Initial training window in days (4 weeks)')
    parser.add_argument('--step_size', type=int, default=14,
                      help='Walk-forward step size in days (2 weeks)')
    parser.add_argument('--balance_per_lot', type=float, default=1000.0,
                      help='Account balance required per 0.01 lot')
    
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                      help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=1e-5,
                      help='Final learning rate')
    parser.add_argument('--eval_freq', type=int, default=100000,
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
    
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    bars_per_day = 24 * 4
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
