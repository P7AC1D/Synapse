"""Backtesting script for trained trading models."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional
from trade_model import TradeModel

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
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def print_metrics(results: dict):
    """Print formatted backtest metrics with grid trading statistics."""
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
    
    print("\n=== Risk Metrics ===")
    risk_metrics = [
        ('Max Drawdown', results.get('max_drawdown_pct', 0.0), '.2f%'),
        ('Current Drawdown', results.get('current_drawdown_pct', 0.0), '.2f%'),
        ('Historical Max DD', results.get('historical_max_drawdown_pct', 0.0), '.2f%')
    ]
    
    print("\n=== Directional Analysis ===")
    directional_metrics = [
        ('Long Trades', results.get('long_trades', 0), 'd'),
        ('Long Win Rate', results.get('long_win_rate', 0.0), '.2f%'),
        ('Short Trades', results.get('short_trades', 0), 'd'),
        ('Short Win Rate', results.get('short_win_rate', 0.0), '.2f%')
    ]
    
    print("\n=== Hold Time Analysis ===")
    hold_time_metrics = [
        ('Avg Hold Time', results.get('avg_hold_time', 0.0), '.1f'),
        ('Winners Hold Time', results.get('win_hold_time', 0.0), '.1f'),
        ('Losers Hold Time', results.get('loss_hold_time', 0.0), '.1f')
    ]
    
    print("\n=== Grid Metrics ===")
    if 'grid_metrics' in results:
        grid_metrics = results.get('grid_metrics', {})
        grid_metrics = [
            ('Total Grids', grid_metrics.get('total_grids', 0), 'd'),
            ('Avg Positions/Grid', grid_metrics.get('avg_positions_per_grid', 0.0), '.2f'),
            ('Grid Efficiency', grid_metrics.get('grid_efficiency', 0.0), '.2f%'),
            ('Position Count', grid_metrics.get('position_count', 0), 'd')
        ]
    else:
        grid_metrics = []

    # Print all metrics sections
    for metrics_list in [performance_metrics, risk_metrics, directional_metrics, 
                        hold_time_metrics, grid_metrics]:
        if metrics_list:  # Only print sections with metrics
            for name, value, format_spec in metrics_list:
                if 'd' in format_spec:
                    print(f"{name}: {value:d}")
                elif '%' in format_spec:
                    print(f"{name}: {value:{format_spec[:-1]}}%")
                else:
                    print(f"{name}: {value:{format_spec}}")
            print("")  # Add blank line between sections

def plot_results(results: dict, save_path: str = None):
    """Plot backtest results with grid trading metrics."""
    plt.figure(figsize=(20, 20))
    
    # Convert trades to DataFrame for plotting
    trades = results.get('trades', [])
    trades_df = pd.DataFrame(trades)
    
    # Extract balance history from trades
    balance_history = []
    current_balance = results.get('initial_balance', 0.0)
    balance_history.append(current_balance)
    
    for trade in trades:
        current_balance += trade['pnl']
        balance_history.append(current_balance)
    
    # Plot balance curve and equity curve
    plt.subplot(611)
    plt.plot(balance_history, label='Account Balance')
    plt.title('Backtest Results')
    plt.xlabel('Trade Number')
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
    if not trades_df.empty:
        trades_df['pnl_cum'] = trades_df['pnl'].cumsum()
        plt.plot(trades_df['pnl_cum'], label='Cumulative PnL')
        plt.title('Trade Performance')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.grid(True)
    
    # Plot win rate and drawdown analysis
    plt.subplot(614)
    if not trades_df.empty:
        rolling_window = min(50, len(trades_df))
        trades_df['win'] = trades_df['pnl'] > 0
        rolling_winrate = trades_df['win'].rolling(rolling_window).mean() * 100
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(rolling_winrate, 'g-', label=f'{rolling_window}-Trade Win Rate')
        ax1.set_ylabel('Win Rate %', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        ax2.plot(trades_df['drawdown'], 'r-', alpha=0.3, label='Drawdown')
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
        plt.title('Trade Size Distribution')
        plt.xlabel('Lot Size')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    # Plot hold time distribution
    plt.subplot(616)
    if not trades_df.empty and 'hold_time' in trades_df.columns:
        plt.hist(trades_df['hold_time'], bins=30, alpha=0.7)
        plt.axvline(trades_df['hold_time'].mean(), color='r', linestyle='--', label='Mean')
        plt.axvline(trades_df['hold_time'].median(), color='g', linestyle='--', label='Median')
        plt.title('Hold Time Distribution')
        plt.xlabel('Hold Time (bars)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def monte_carlo_simulation(df: pd.DataFrame, model: TradeModel, params: dict,
                         n_sims: int = 100, random_seed: int = 42) -> Dict:
    """Run Monte Carlo simulation by varying trade entry/exit points."""
    np.random.seed(random_seed)
    results = []
    
    print(f"\nRunning {n_sims} Monte Carlo simulations...")
    for i in range(n_sims):
        # Randomly shift trade timings within a small window
        jitter = np.random.normal(0, 2, len(df))  # 2-bar standard deviation
        df_sim = df.copy()
        df_sim.index = df_sim.index + pd.Timedelta(seconds=int(jitter.mean()))
        
        # Run backtest with jittered data
        sim_result = model.backtest(
            data=df_sim,
            initial_balance=params['initial_balance'],
            balance_per_lot=params['balance_per_lot']
        )
        results.append(sim_result)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} simulations")
    
    # Aggregate results
    returns = [r['return_pct'] for r in results]
    drawdowns = [r['max_drawdown_pct'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    
    summary = {
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'return_quartiles': [float(np.percentile(returns, q)) for q in [25, 50, 75]],
        'drawdown_mean': float(np.mean(drawdowns)),
        'drawdown_worst': float(np.max(drawdowns)),
        'win_rate_mean': float(np.mean(win_rates)),
        'win_rate_std': float(np.std(win_rates)),
        'simulations': results
    }
    
    # Plot Monte Carlo results
    plt.figure(figsize=(15, 10))
    
    # Plot distributions
    plt.subplot(221)
    plt.hist(returns, bins=30, alpha=0.7)
    plt.axvline(summary['return_mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(summary['return_quartiles'][1], color='g', linestyle='--', label='Median')
    plt.title('Return Distribution')
    plt.xlabel('Return %')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(222)
    plt.hist(drawdowns, bins=30, alpha=0.7)
    plt.axvline(summary['drawdown_mean'], color='r', linestyle='--', label='Mean')
    plt.title('Max Drawdown Distribution')
    plt.xlabel('Max Drawdown %')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(223)
    plt.hist(win_rates, bins=30, alpha=0.7)
    plt.axvline(summary['win_rate_mean'], color='r', linestyle='--', label='Mean')
    plt.title('Win Rate Distribution')
    plt.xlabel('Win Rate %')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot sample equity curves
    plt.subplot(224)
    for i, result in enumerate(results[:min(20, n_sims)]):
        trades_df = pd.DataFrame(result['trades'])
        equity = trades_df['pnl'].cumsum()
        plt.plot(equity.values, alpha=0.3, color='blue')
    plt.title('Sample Equity Curves')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL')
    
    plt.tight_layout()
    plt.show()
    
    # Print Monte Carlo summary
    print("\n=== Monte Carlo Simulation Results ===")
    print(f"Number of simulations: {n_sims}")
    print(f"\nReturn Distribution:")
    print(f"Mean: {summary['return_mean']:.2f}%")
    print(f"Std Dev: {summary['return_std']:.2f}%")
    print(f"Quartiles (25/50/75): {summary['return_quartiles'][0]:.2f}%/"
          f"{summary['return_quartiles'][1]:.2f}%/{summary['return_quartiles'][2]:.2f}%")
    print(f"\nDrawdown Analysis:")
    print(f"Mean Max DD: {summary['drawdown_mean']:.2f}%")
    print(f"Worst DD: {summary['drawdown_worst']:.2f}%")
    print(f"\nWin Rate Analysis:")
    print(f"Mean: {summary['win_rate_mean']:.2f}%")
    print(f"Std Dev: {summary['win_rate_std']:.2f}%")
    
    return summary

def compare_backtests(results_list: List[Dict], plot_path: Optional[str] = None) -> None:
    """Generate comparison plots for multiple backtest results."""
    plt.figure(figsize=(20, 15))
    
    # Plot equity curves
    ax1 = plt.subplot(311)
    for result in results_list:
        trades = result['results']['trades']
        trades_df = pd.DataFrame(trades)
        profits = trades_df['pnl'].cumsum()
        label = f"Seed {result['metadata']['model']['seed']} ({result['metadata']['model']['period']})"
        ax1.plot(profits.index, profits, label=label, alpha=0.7)
    
    ax1.set_title('Equity Curves Comparison')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative PnL')
    ax1.legend()
    ax1.grid(True)
    
    # Compare performance metrics
    ax2 = plt.subplot(312)
    metrics = []
    for result in results_list:
        r = result['results']
        model_info = result['metadata']['model']
        
        metrics.append({
            'Model': f"Seed {model_info['seed']} ({model_info['period']})",
            'Return %': round(r.get('return_pct', 0), 2),
            'Win Rate %': round(r.get('win_rate', 0), 2),
            'Max DD %': round(r.get('max_drawdown_pct', 0), 2),
            'Profit Factor': round(r.get('profit_factor', 0), 2),
            'Trades': r.get('total_trades', 0)
        })
    
    metrics_df = pd.DataFrame(metrics).set_index('Model')
    metrics_df[['Return %', 'Win Rate %', 'Max DD %', 'Profit Factor']].plot(kind='bar', ax=ax2)
    ax2.set_title('Performance Metrics Comparison')
    ax2.grid(True)
    plt.xticks(rotation=45)
    
    # Print detailed metrics table
    print("\n=== Model Comparison ===")
    print(metrics_df.to_string())
    
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Backtest trained trading models')
    
    # Add arguments for model configuration
    parser.add_argument('--seeds', type=str, required=True,
                     help='Comma-separated list of model seeds to backtest')
    parser.add_argument('--periods', type=str, default='best',
                     help='Comma-separated list of periods (best,final,latest) or single value for all')
    
    # Add arguments for data and parameters
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance')
    parser.add_argument('--balance_per_lot', type=float, default=1000.0,
                      help='Account balance required per 0.01 lot (default: 1000)')
    parser.add_argument('--results_dir', type=str, default='../results/backtest',
                      help='Directory to save backtest results')
    
    # Add arguments for date range and Monte Carlo simulation
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--monte_carlo', type=int, default=0,
                      help='Number of Monte Carlo simulations for best model (0 to disable)')
    parser.add_argument('--monte_carlo_seed', type=int, default=42,
                      help='Random seed for Monte Carlo simulations')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        # Parse seeds and periods
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        periods = args.periods.split(',') if ',' in args.periods else [args.periods] * len(seeds)
        
        if len(periods) == 1:
            periods = periods * len(seeds)
        
        if len(periods) != len(seeds):
            raise ValueError("Number of periods must match number of seeds or be a single value")
        
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
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(args.results_dir, f"backtest_comparison_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Run backtests and store results
        all_results = []
        best_result = None
        best_score = float('-inf')
        
        for seed, period in zip(seeds, periods):
            try:
                model_path = f"../results/{seed}/{'model_final.zip' if period == 'final' else 'best_balance_model.zip'}"
                if not os.path.exists(model_path):
                    print(f"Warning: Model not found at {model_path}, skipping...")
                    continue
                
                print(f"\nInitializing model: Seed {seed}, Period {period}")
                model = TradeModel(model_path=model_path)
                
                # Run backtest
                results = model.backtest(
                    data=df,
                    initial_balance=args.initial_balance,
                    balance_per_lot=args.balance_per_lot
                )
                
                # Calculate score for determining best model
                score = results['return_pct'] * (1 - results['max_drawdown_pct']/100)
                if score > best_score:
                    best_score = score
                    best_result = {'model': model, 'results': results}
                
                # Add metadata and save results
                results_with_meta = {
                    'metadata': {
                        'model': {'seed': seed, 'period': period},
                        'data': {
                            'path': args.data_path,
                            'bars': len(df),
                            'start': str(df.index[0]),
                            'end': str(df.index[-1])
                        }
                    },
                    'results': results
                }
                
                all_results.append(results_with_meta)
                print(f"\nResults for Seed {seed}, Period {period}:")
                print_metrics(results)
                
            except Exception as e:
                print(f"Error processing seed {seed}, period {period}: {str(e)}")
                continue
        
        if not all_results:
            raise ValueError("No successful backtests to analyze")
        
        # Save combined results
        results_file = os.path.join(results_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=convert_to_serializable)
        
        # Plot comparison results
        print("\nGenerating comparison plots...")
        plot_path = os.path.join(results_dir, 'comparison_plot.png')
        compare_backtests(all_results, plot_path)
        
        # Run Monte Carlo simulation if requested
        if args.monte_carlo > 0 and best_result is not None:
            print("\nRunning Monte Carlo simulation on best performing model...")
            monte_carlo_results = monte_carlo_simulation(
                df=df,
                model=best_result['model'],
                params={
                    'initial_balance': args.initial_balance,
                    'balance_per_lot': args.balance_per_lot
                },
                n_sims=args.monte_carlo,
                random_seed=args.monte_carlo_seed
            )
            
            # Save Monte Carlo results
            mc_file = os.path.join(results_dir, 'monte_carlo_results.json')
            with open(mc_file, 'w') as f:
                json.dump(monte_carlo_results, f, indent=4, default=convert_to_serializable)
        
        print(f"\nResults saved to: {results_dir}")
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        return

if __name__ == "__main__":
    main()
