"""Backtesting script for single trained trading model."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from trade_model import TradeModel
import time
import sys
import threading

# Global flag to control the progress indicator
stop_progress = False

def show_progress_continuous(message="Running backtest"):
    """Continuous progress indicator that runs until stopped."""
    global stop_progress
    stop_progress = False
    chars = "|/-\\"
    i = 0
    while not stop_progress:
        char = chars[i % len(chars)]
        sys.stdout.write(f'\r{message}... {char}')
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

def stop_progress_indicator():
    """Stop the progress indicator thread."""
    global stop_progress
    stop_progress = True
    time.sleep(0.2)  # Give thread time to terminate
    sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the progress line
    sys.stdout.flush()

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
        ('Current Drawdown', results.get('current_drawdown_pct', 0.0), '.2f%'),
        ('Current Equity Drawdown', results.get('current_equity_drawndown_pct', 0.0), '.2f%'),
        ('Historical Max DD', results.get('historical_max_drawdown_pct', 0.0), '.2f%')
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

    # Hold Time Analysis
    print("=== Hold Time Analysis ===")
    hold_time_metrics = [
        ('Avg Hold Time', results.get('avg_hold_time', 0.0), '.1f'),
        ('Winners Hold Time', results.get('win_hold_time', 0.0), '.1f'),
        ('Losers Hold Time', results.get('loss_hold_time', 0.0), '.1f'),
        ('Max Hold Time', results.get('max_hold_bars', 64), '.1f'),  # Use default if not available
        ('Avg Hold Time %', results.get('avg_hold_time', 0.0) / results.get('max_hold_bars', 64) * 100, '.1f%')
    ]
    for name, value, format_spec in hold_time_metrics:
        if '%' in format_spec:
            print(f"{name}: {value:{format_spec[:-1]}}%")
        else:
            print(f"{name}: {value:{format_spec}}")
    print("")

def plot_results(results: dict, save_path: str = None):
    """Plot backtest results and performance metrics."""
    plt.figure(figsize=(20, 20))
    
    # Convert trades to DataFrame for plotting
    trades = results.get('trades', [])
    
    # Handle empty trades case gracefully
    if not trades:
        print("No trades to plot. Skipping visualization.")
        return
        
    trades_df = pd.DataFrame(trades)
    
    # Extract balance history from trades
    balance_history = []
    current_balance = results.get('initial_balance', 0.0)
    balance_history.append(current_balance)
    
    for trade in trades:
        current_balance += trade.get('pnl', 0)
        balance_history.append(current_balance)
    
    # Plot balance curve and equity curve
    plt.subplot(611)
    plt.plot(balance_history, label='Account Balance')
    plt.title('Backtest Results')
    plt.ylabel('Balance')
    plt.legend()
    plt.grid(True)
    
    # Plot balance curve with drawdown overlay
    plt.subplot(612)
    balance_series = pd.Series(balance_history)
    rolling_max = balance_series.expanding().max()
    drawdowns = ((balance_series - rolling_max) / rolling_max) * 100
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot balance
    ax1.plot(balance_series, 'b-', label='Balance')
    ax1.set_ylabel('Balance', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot drawdown
    ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown %', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Balance and Drawdown')
    plt.grid(True)
    
    # Plot trade PnLs with performance analytics
    plt.subplot(613)
    if not trades_df.empty and 'pnl' in trades_df.columns:
        trades_df['pnl_cum'] = trades_df['pnl'].cumsum()
        plt.plot(trades_df['pnl_cum'], label='Cumulative PnL')
        plt.title('Trade Performance')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.grid(True)
    
    # Plot win rate and drawdown analysis - FIX HERE
    plt.subplot(614)
    if not trades_df.empty and 'pnl' in trades_df.columns:
        rolling_window = min(50, len(trades_df))
        trades_df['win'] = trades_df['pnl'] > 0
        rolling_winrate = trades_df['win'].rolling(rolling_window).mean() * 100
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(rolling_winrate, 'g-', label=f'{rolling_window}-Trade Win Rate')
        ax1.set_ylabel('Win Rate %', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        # Calculate drawdown per trade - FIXED!
        # This ensures we have a drawdown value for each trade
        trade_drawdowns = pd.Series(drawdowns.iloc[1:].values[:len(trades_df)])
        
        ax2.plot(trade_drawdowns, 'r-', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown %', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        plt.title('Rolling Win Rate and Drawdown')
        plt.grid(True)
    
    # Plot trade size distribution
    plt.subplot(615)
    if not trades_df.empty and 'lot_size' in trades_df.columns:
        plt.hist(trades_df['lot_size'], bins=30, alpha=0.7)
        plt.axvline(trades_df['lot_size'].mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(trades_df['lot_size'].median(), color='g', linestyle='--', label='Median')
        plt.title('Trade Size Distribution (lots)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    # Plot hold time distribution
    plt.subplot(616)
    if not trades_df.empty and 'hold_time' in trades_df.columns:
        plt.hist(trades_df['hold_time'], bins=30, alpha=0.7)
        plt.axvline(trades_df['hold_time'].mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(trades_df['hold_time'].median(), color='g', linestyle='--', label='Median')
        plt.title('Hold Time Distribution (bars)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
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
