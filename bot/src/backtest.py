import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import json
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from trade_environment import TradingEnv
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def load_model(model_path, model_type, env, device='cpu'):
    """Load a trained model with proper environment context."""
    model_class = DQN if model_type == 'DQN' else PPO
    
    # Define custom objects for loading
    custom_objects = {
        "lr_schedule": lambda _: 1e-4,
        "exploration_schedule": lambda _: 0.01  # Fixed exploration value at the final epsilon
    }
    
    try:
        model = model_class.load(
            model_path,
            env=env,
            device=device,
            custom_objects=custom_objects
        )
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def backtest_model(model, env, initial_balance):
    """Run backtest on the model and return performance metrics."""
    obs, _ = env.reset()
    done = False
    
    # Track metrics
    balance_history = [initial_balance]
    reward_history = []
    position_history = []
    trades = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record metrics
        balance_history.append(env.balance)
        reward_history.append(reward)
        position_history.append(action)
    
    if hasattr(env, 'trades'):
        trades = env.trades
    
    return {
        'balance_history': balance_history,
        'reward_history': reward_history,
        'position_history': position_history,
        'trades': trades,
        'final_balance': env.balance,
        'total_reward': sum(reward_history)
    }

def plot_results(results, save_path=None):
    """Plot backtest results."""
    plt.figure(figsize=(15, 10))
    
    # Plot balance over time
    plt.subplot(311)
    plt.plot(results['balance_history'], label='Account Balance')
    plt.title('Backtest Results')
    plt.xlabel('Steps')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative rewards
    plt.subplot(312)
    plt.plot(np.cumsum(results['reward_history']), label='Cumulative Reward')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot positions
    plt.subplot(313)
    plt.plot(results['position_history'], label='Position')
    plt.xlabel('Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_metrics(results, initial_balance):
    """Calculate trading performance metrics."""
    trades_df = pd.DataFrame(results['trades'])
    if len(trades_df) == 0:
        return "No trades were executed during backtesting."
        
    # Basic metrics
    final_balance = results['final_balance']
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    total_trades = len(trades_df)
    
    # Win rate calculations
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    # Average trade metrics
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
    
    # Risk/Reward
    risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    # Position-specific metrics
    long_trades = trades_df[trades_df['position'] == 1]
    short_trades = trades_df[trades_df['position'] == -1]
    long_win_rate = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
    short_win_rate = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
    
    metrics = {
        'Initial Balance': initial_balance,
        'Final Balance': final_balance,
        'Total Return (%)': total_return,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Risk/Reward Ratio': risk_reward_ratio,
        'Long Trades': len(long_trades),
        'Long Win Rate (%)': long_win_rate,
        'Short Trades': len(short_trades),
        'Short Win Rate (%)': short_win_rate
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Backtest a trained trading model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--model_type', type=str, choices=['DQN', 'PPO'], required=True,
                      help='Type of model (DQN or PPO)')
    parser.add_argument('--bar_count', type=int, default=50,
                      help='Number of bars to include in state (default: 50)')
    parser.add_argument('--normalization_window', type=int, default=100,
                      help='Window size for normalization (default: 100)')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--results_dir', type=str, default='../results/backtest',
                      help='Directory to save backtest results')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                      help='Device to run inference on')
    parser.add_argument('--risk_percentage', type=float, default=0.01,
                      help='Risk percentage per trade (default: 0.01)')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = pd.read_csv(args.data_path)
        print(f"Initial columns: {df.columns.tolist()}")
        print(f"Initial shape: {df.shape}")
        
        df.set_index('time', inplace=True)
        
        # Handle technical indicator columns - wrapped in try/except for flexibility
        columns_to_drop = ['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
            print(f"Dropped columns: {existing_columns}")
        
        # Print final data shape and columns
        print(f"Final shape: {df.shape}")
        print(f"Final columns: {df.columns.tolist()}")
        
        # Calculate expected observation space size
        expected_obs_size = df.shape[1] * args.bar_count
        print(f"Expected observation space size: {expected_obs_size}")
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {str(e)}")
        return
    
    # Print environment parameters
    print(f"\nEnvironment Parameters:")
    print(f"Bar Count: {args.bar_count}")
    print(f"Normalization Window: {args.normalization_window}")
    print(f"Risk Percentage: {args.risk_percentage}")
    print(f"Initial Balance: {args.initial_balance}")
    
    # Create environment
    env = TradingEnv(
        df,
        initial_balance=args.initial_balance,
        bar_count=args.bar_count,
        normalization_window=args.normalization_window,
        random_start=False,  # No random start for backtesting
        lot_percentage=args.risk_percentage  # Using risk_percentage as lot_percentage
    )
    
    # Load model
    try:
        model = load_model(args.model_path, args.model_type, env, args.device)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # Run backtest
    print("Running backtest...")
    results = backtest_model(model, env, args.initial_balance)
    
    # Calculate metrics
    metrics = calculate_metrics(results, args.initial_balance)
    
    # Handle string response (no trades case)
    if isinstance(metrics, str):
        print("\nBacktest Results:")
        print(metrics)
        return
    
    # Save metrics
    metrics_file = os.path.join(args.results_dir, 'backtest_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics
    print("\nBacktest Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Plot and save results if we have trades
    plot_path = os.path.join(args.results_dir, 'backtest_plot.png')
    plot_results(results, save_path=plot_path)
    
    # Save trades log if available
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = os.path.join(args.results_dir, 'trades_log.csv')
        trades_df.to_csv(trades_file)
        print(f"\nTrades log saved to: {trades_file}")

if __name__ == "__main__":
    main()
