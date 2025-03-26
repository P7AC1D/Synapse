"""Backtesting script for trained trading models."""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from trade_model import TradeModel

def plot_results(results: dict, save_path: str = None):
    """Plot backtest results."""
    plt.figure(figsize=(15, 12))
    
    # Convert trades to DataFrame for plotting
    trades_df = pd.DataFrame(results['trades'])
    
    # Extract balance history from trades
    balance_history = []
    current_balance = results['initial_balance']
    balance_history.append(current_balance)
    
    for trade in results['trades']:
        current_balance += trade['pnl']
        balance_history.append(current_balance)
    
    # Plot balance curve
    plt.subplot(411)
    plt.plot(balance_history, label='Account Balance')
    plt.title('Backtest Results')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    
    # Plot trade PnLs
    plt.subplot(412)
    profits = trades_df['pnl'].cumsum()
    plt.plot(profits, label='Cumulative PnL')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Profit/Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot RRR distribution
    plt.subplot(413)
    plt.hist(trades_df['rrr'], bins=50, label='RRR Distribution')
    plt.xlabel('Risk-Reward Ratio')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(414)
    balance_series = pd.Series(balance_history)
    rolling_max = balance_series.expanding().max()
    drawdowns = ((balance_series - rolling_max) / rolling_max) * 100
    plt.plot(drawdowns, label='Drawdown %')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown %')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_metrics(results: dict):
    """Print formatted backtest metrics."""
    metrics = [
        ('Initial Balance', results['initial_balance'], '.2f'),
        ('Final Balance', results['final_balance'], '.2f'),
        ('Return', results['return_pct'], '.2f%'),
        ('Total Trades', results['total_trades'], 'd'),
        ('Win Rate', results['win_rate'], '.2f%'),
        ('Profit Factor', results['profit_factor'], '.2f'),
        ('Max Drawdown', results['max_drawdown_pct'], '.2f%'),
        ('Long Trades', results['long_trades'], 'd'),
        ('Long Win Rate', results['long_win_rate'], '.2f%'),
        ('Short Trades', results['short_trades'], 'd'),
        ('Short Win Rate', results['short_win_rate'], '.2f%'),
        ('Average RRR', results['avg_rrr'], '.2f'),
        ('Expected Value', results['expected_value'], '.2f'),
        ('Kelly Criterion', results['kelly_criterion'], '.3f'),
        ('Sharpe Ratio', results['sharpe_ratio'], '.2f')
    ]
    
    print("\n=== Backtest Results ===")
    for name, value, format_spec in metrics:
        if 'd' in format_spec:
            print(f"{name}: {value:d}")
        elif '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")

def main():
    parser = argparse.ArgumentParser(description='Backtest a trained trading model')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--bar_count', type=int, default=10,
                      help='Number of bars to include in state (default: 10)')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance')
    parser.add_argument('--risk_percentage', type=float, default=0.01,
                      help='Risk percentage per trade (default: 0.01)')
    parser.add_argument('--results_dir', type=str, default='../results/backtest',
                      help='Directory to save backtest results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        # Load data
        print("\nLoading market data...")
        df = pd.read_csv(args.data_path)
        df.set_index('time', inplace=True)
        print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Initialize model
        print("\nInitializing trading model...")
        model = TradeModel(
            model_path=args.model_path,
            bar_count=args.bar_count
        )
        
        # Run backtest
        print("\nRunning backtest...")
        results = model.backtest(
            data=df,
            initial_balance=args.initial_balance,
            risk_percentage=args.risk_percentage/100  # Convert to decimal
        )
        
        # Save results
        results_file = os.path.join(args.results_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, bool):
                    return bool(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray, )):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                return obj

            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=4)
        
        # Print metrics
        print_metrics(results)
        
        # Plot and save results
        plot_path = os.path.join(args.results_dir, 'backtest_plot.png')
        plot_results(results, save_path=plot_path)
        
        # Save detailed trade log
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_file = os.path.join(args.results_dir, 'trades_log.csv')
            trades_df.to_csv(trades_file)
            print(f"\nTrades log saved to: {trades_file}")
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        return

if __name__ == "__main__":
    main()
