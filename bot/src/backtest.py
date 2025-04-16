"""Backtesting script for single trained trading model."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Dict, Any
from trade_model import TradeModel
import time
import sys
import threading
from utils.progress import show_progress_continuous, stop_progress_indicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def convert_to_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format."""
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
        return {k: convert_to_serializable(v) for k, v in obj.items() }
    return obj

def print_metrics(results: dict):
    """Print formatted backtest performance metrics."""
    # Performance Metrics
    print("\n=== Performance Metrics ===")
    performance_metrics = [
        ('Initial Balance', results.get('initial_balance', 0.0), '.2f'),
        ('Final Balance', results.get('final_balance', 0.0), '.2f'),
        ('Total Return', results.get('return_pct', 0.0), '.2f%'),
        ('Total Trades', results.get('total_trades', 0), 'd'),
        ('Win Rate', results.get('win_rate', 0.0), '.2f%'),
        ('Profit Factor', results.get('profit_factor', 0.0), '.2f'),
        ('Expected Value', results.get('expected_value', 0.0), '.2f'),
        ('Sharpe Ratio', results.get('sharpe_ratio', 0.0), '.2f')
    ]
    for name, value, format_spec in performance_metrics:
        if 'd' in format_spec:
            print(f"{name}: {value:d}")
        elif '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")
    print("")

    # Risk Metrics
    print("=== Risk Metrics ===")
    risk_metrics = [
        ('Max Balance Drawdown', results.get('max_drawdown_pct', 0.0), '.2f%'),
        ('Max Equity Drawdown', results.get('max_equity_drawdown_pct', 0.0), '.2f%'),
        ('Current Balance DD', results.get('current_drawdown_pct', 0.0), '.2f%'),
        ('Current Equity DD', results.get('current_equity_drawdown_pct', 0.0), '.2f%')  # Fixed typo
    ]
    for name, value, format_spec in risk_metrics:
        if '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")
    print("")

    # Directional Analysis
    print("=== Directional Analysis ===")
    directional_metrics = [
        ('Long Trades', results.get('long_trades', 0), 'd'),
        ('Long Win Rate', results.get('long_win_rate', 0.0), '.2f%'),
        ('Short Trades', results.get('short_trades', 0), 'd'),
        ('Short Win Rate', results.get('short_win_rate', 0.0), '.2f%')
    ]
    for name, value, format_spec in directional_metrics:
        if 'd' in format_spec:
            print(f"{name}: {value:d}")
        elif '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")
    print("")

    # Hold Time and Pips Analysis as Table
    print("=== Hold Time and Pips Analysis ===")
    print("\n| Metric Type   | Average | Median | 90th Percentile |")
    print("|--------------|---------|---------|----------------|")
    
    # Winners Hold Time
    print(f"| Winners Hold | {results.get('win_hold_time', 0.0):7.1f} | {results.get('win_hold_time_median', 0.0):7.1f} | {results.get('win_hold_time_90th', 0.0):14.1f} |")
    
    # Losers Hold Time
    print(f"| Losers Hold  | {results.get('loss_hold_time', 0.0):7.1f} | {results.get('loss_hold_time_median', 0.0):7.1f} | {results.get('loss_hold_time_90th', 0.0):14.1f} |")
    
    # Winners Pips
    print(f"| Winners Pips | {results.get('avg_win_pips', 0.0):7.1f} | {results.get('median_win_pips', 0.0):7.1f} | {results.get('win_pips_90th', 0.0):14.1f} |")
    
    # Losers Pips
    print(f"| Losers Pips  | {results.get('avg_loss_pips', 0.0):7.1f} | {results.get('median_loss_pips', 0.0):7.1f} | {results.get('loss_pips_90th', 0.0):14.1f} |")
    
    print("")

    # Consecutive Trade Analysis
    print("=== Consecutive Trade Analysis ===")
    consecutive_metrics = [
        ('Max Consecutive Wins', results.get('max_consecutive_wins', 0), 'd'),
        ('Max Consecutive Losses', results.get('max_consecutive_losses', 0), 'd'),
        ('Current Win Streak', results.get('current_consecutive_wins', 0), 'd'),
        ('Current Loss Streak', results.get('current_consecutive_losses', 0), 'd')
    ]
    for name, value, format_spec in consecutive_metrics:
        if 'd' in format_spec:
            print(f"{name}: {value:d}")
        elif '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")
    print("")

def plot_results(results: dict, save_path: str = None):
    """Plot backtest results and performance metrics."""
    fig = plt.figure(figsize=(20, 16))  # Adjusted height for four plots
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Convert trades to DataFrame for plotting
    trades = results.get('trades', [])
    
    # Handle empty trades case gracefully
    if not trades:
        print("No trades to plot. Skipping visualization.")
        return
        
    trades_df = pd.DataFrame(trades)
    
    # Extract equity history from trades and track streaks (including unrealized PnL)
    equity_history = []
    current_balance = results.get('initial_balance', 0.0)
    current_equity = current_balance
    equity_history.append(current_equity)
    
    unrealized_pnl = 0.0  # Track unrealized PnL
    for i, trade in enumerate(trades):
        pnl = trade.get('pnl', 0)
        current_balance += pnl
        
        # For the last trade, include unrealized PnL if it's active
        if i == len(trades) - 1 and results.get('active_positions', 0) > 0:
            unrealized_pnl = trade.get('unrealized_pnl', 0)
        
        current_equity = current_balance + unrealized_pnl
        equity_history.append(current_equity)
    
    # Plot balance curve with drawdown overlay (spans both columns)
    ax1 = plt.subplot(gs[0, :])
    equity_series = pd.Series(equity_history)
    rolling_max = equity_series.expanding().max()
    drawdowns = ((equity_series - rolling_max) / rolling_max) * 100
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot equity
    ax1.plot(equity_series, 'b-', label='Balance')
    ax1.set_ylabel('Balance ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot drawdown
    ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown %', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Equity and Drawdown')
    plt.grid(True)
    
    # Plot trade size distribution (left column)
    plt.subplot(gs[1, 0])
    if not trades_df.empty and 'lot_size' in trades_df.columns:
        plt.hist(trades_df['lot_size'], bins=30, alpha=0.7, color='b', label='Lots')
        mean_lots = trades_df['lot_size'].mean()
        median_lots = trades_df['lot_size'].median()
        plt.axvline(mean_lots, color='b', linestyle='--', label=f'Mean: {mean_lots:.2f}')
        plt.axvline(median_lots, color='b', linestyle=':', label=f'Median: {median_lots:.2f}')
        plt.title('Trade Size Distribution')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    # Plot hold time distribution (right column)
    plt.subplot(gs[1, 1])
    if not trades_df.empty and 'hold_time' in trades_df.columns:
        plt.hist(trades_df['hold_time'], bins=30, alpha=0.7, color='purple', label='Bars')
        mean_hold = trades_df['hold_time'].mean()
        median_hold = trades_df['hold_time'].median()
        plt.axvline(mean_hold, color='purple', linestyle='--', label=f'Mean: {mean_hold:.1f}')
        plt.axvline(median_hold, color='purple', linestyle=':', label=f'Median: {median_hold:.1f}')
        plt.title('Hold Time Distribution')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    # Create side-by-side plots for profit and loss pips (third row, split into columns)
    if not trades_df.empty and 'profit_pips' in trades_df.columns:
        winning_trades = trades_df[trades_df['profit_pips'] > 0]
        losing_trades = trades_df[trades_df['profit_pips'] <= 0]
        
        # Plot winning trades (profit pips) - left column
        ax_profit = plt.subplot(gs[2, 0])
        if not winning_trades.empty:
            plt.hist(winning_trades['profit_pips'], bins=30, alpha=0.7, color='g', label='Profit Pips')
            mean_profit = winning_trades['profit_pips'].mean()
            median_profit = winning_trades['profit_pips'].median()
            plt.axvline(mean_profit, color='g', linestyle='--', label=f'Mean: {mean_profit:.1f}')
            plt.axvline(median_profit, color='g', linestyle=':', label=f'Median: {median_profit:.1f}')
            plt.title('Profit Distribution')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
        
        # Plot losing trades (absolute loss pips) - right column
        ax_loss = plt.subplot(gs[2, 1])
        if not losing_trades.empty:
            abs_loss_pips = abs(losing_trades['profit_pips'])
            plt.hist(abs_loss_pips, bins=30, alpha=0.7, color='r', label='Loss Pips')
            mean_loss = abs_loss_pips.mean()
            median_loss = abs_loss_pips.median()
            plt.axvline(mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.1f}')
            plt.axvline(median_loss, color='r', linestyle=':', label=f'Median: {median_loss:.1f}')
            plt.title('Loss Distribution')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)

    # Adjust layout with padding
    plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def show_progress(message="Running backtest"):
    """Simple progress indicator for long-running processes."""
    chars = "|/-\\"
    for char in chars:
        sys.stdout.write(f'\r{message}... {char}')
        sys.stdout.flush()
        time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description='Backtest trained trading model')
    
    parser.add_argument('--model_path', type=str, required=True,
                     help='Path to the model file to backtest')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance')
    parser.add_argument('--balance_per_lot', type=float, default=1000.0,
                      help='Account balance required per 0.01 lot (default: 1000)')
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true',
                      help='Generate plots')
    parser.add_argument('--quiet', action='store_true',
                      help='Run backtesting without verbose logging')
    parser.add_argument('--output_json', type=str, default=None,
                      help='Path to save backtest results in JSON format')
    parser.add_argument('--output_plot', type=str, default=None,
                      help='Path to save backtest plot')
    
    args = parser.parse_args()
    
    try:
        
        # Load and validate data
        print("\nLoading market data...")
        df = pd.read_csv(args.data_path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        
        # Apply date filters if provided
        if args.start_date:
            start_ts = pd.Timestamp(args.start_date).tz_localize('UTC')
            df = df[df.index >= start_ts]
        if args.end_date:
            end_ts = pd.Timestamp(args.end_date).tz_localize('UTC')
            df = df[df.index <= end_ts]
        
        if len(df) == 0:
            raise ValueError("No data available for specified date range")
        
        print(f"Loaded {len(df):,d} bars from {df.index[0]} to {df.index[-1]}")
        
        # Validate data
        missing_columns = [col for col in ['open', 'close', 'high', 'low', 'spread'] 
                          if col not in df.columns]
        if missing_columns:
            print(f"WARNING: Dataset missing columns: {missing_columns}")
            print("These columns are required for backtesting. Will attempt to continue.")
        
        # Initialize model
        print(f"\nInitializing model from: {args.model_path}")
        model = TradeModel(
            model_path=args.model_path,
            balance_per_lot=args.balance_per_lot  # Pass the parameter consistently
        )
        
        # Run backtest
        if args.quiet:
            print("Running quiet backtest...")
            progress_thread = None
            
            # Start continuous progress indicator
            if len(df) > 1000:  # For any substantial dataset
                progress_thread = threading.Thread(
                    target=show_progress_continuous,
                    args=("Running backtest",)
                )
                progress_thread.daemon = True
                progress_thread.start()
            
            try:    
                results = model.evaluate(
                    data=df,
                    initial_balance=args.initial_balance,
                    balance_per_lot=args.balance_per_lot
                )
            finally:
                # Always stop the progress indicator
                if progress_thread and progress_thread.is_alive():
                    stop_progress_indicator()
        else:
            print("Running detailed backtest...")
            results = model.backtest(
                data=df,
                initial_balance=args.initial_balance,
                balance_per_lot=args.balance_per_lot
            )
        
        # Print metrics
        print("\nBacktest Results:")
        print_metrics(results)
        
        # Save results to JSON if requested
        if args.output_json:
            import json
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=4)
                print(f"Results saved to {args.output_json}")
            except Exception as e:
                print(f"Error saving results to JSON: {e}")

        # Plot results if requested
        if args.plot or args.output_plot:
            plot_results(results, save_path=args.output_plot)
            if args.output_plot:
                print(f"Plot saved to {args.output_plot}")
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        return

if __name__ == "__main__":
    main()
