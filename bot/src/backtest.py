import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import json
import numpy as np
import pandas as pd
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trade_environment import TradingEnv
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def load_model(model_path, env, device='cpu'):
    """Load a trained PPO-LSTM model with proper environment context."""
    try:
        model = RecurrentPPO.load(model_path, env=env, device=device)
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
    rrr_history = []  # Track RRR for trades
    trades = []
    
    # Initialize LSTM states
    lstm_states = None
    
    while not done:
        # Get prediction with LSTM state management
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Process action for metrics
        position, sl_points, tp_points = env._process_action(action)
        rrr = tp_points / sl_points if sl_points > 0 else 0
        
        # Record metrics
        balance_history.append(env.balance)
        reward_history.append(reward)
        position_history.append(position)
        rrr_history.append(rrr)
    
    if hasattr(env, 'trades'):
        trades = env.trades
    
    return {
        'balance_history': balance_history,
        'reward_history': reward_history,
        'position_history': position_history,
        'rrr_history': rrr_history,
        'trades': trades,
        'final_balance': env.balance,
        'total_reward': sum(reward_history)
    }

def plot_results(results, save_path=None):
    """Plot backtest results with RRR."""
    plt.figure(figsize=(15, 12))
    
    # Plot balance over time
    plt.subplot(411)
    plt.plot(results['balance_history'], label='Account Balance')
    plt.title('Backtest Results')
    plt.xlabel('Steps')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative rewards
    plt.subplot(412)
    plt.plot(np.cumsum(results['reward_history']), label='Cumulative Reward')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot positions
    plt.subplot(413)
    plt.plot(results['position_history'], label='Position')
    plt.xlabel('Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    
    # Plot RRR
    plt.subplot(414)
    plt.plot(results['rrr_history'], label='Risk-Reward Ratio')
    plt.xlabel('Steps')
    plt.ylabel('RRR')
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
    
    # Risk/Reward metrics
    avg_rrr = trades_df['rrr'].mean() if 'rrr' in trades_df.columns else 0
    win_rrr = winning_trades['rrr'].mean() if len(winning_trades) > 0 and 'rrr' in trades_df.columns else 0
    lose_rrr = losing_trades['rrr'].mean() if len(losing_trades) > 0 and 'rrr' in trades_df.columns else 0
    
    # Position-specific metrics
    long_trades = trades_df[trades_df['position'] == 1]
    short_trades = trades_df[trades_df['position'] == -1]
    long_win_rate = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
    short_win_rate = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
    
    # Kelly Criterion
    kelly = win_rate/100 - ((1 - win_rate/100) / avg_rrr) if avg_rrr > 0 else 0
    
    metrics = {
        'Initial Balance': initial_balance,
        'Final Balance': final_balance,
        'Total Return (%)': total_return,
        'Total Trades': total_trades,
        'Win Rate (%)': win_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Average RRR': avg_rrr,
        'Winning Trades RRR': win_rrr,
        'Losing Trades RRR': lose_rrr,
        'Long Trades': len(long_trades),
        'Long Win Rate (%)': long_win_rate,
        'Short Trades': len(short_trades),
        'Short Win Rate (%)': short_win_rate,
        'Kelly Criterion': kelly
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Backtest a trained PPO-LSTM trading model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
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
        print("\nLoading and preprocessing data...")
        df = pd.read_csv(args.data_path)
        print(f"Initial shape: {df.shape}")
        
        df.set_index('time', inplace=True)
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
        lot_percentage=args.risk_percentage
    )
    
    # Load model
    try:
        model = load_model(args.model_path, env, args.device)
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
