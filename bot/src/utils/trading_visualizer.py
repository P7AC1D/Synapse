"""Visualization utilities for trading results.

This module provides utilities for visualizing trading performance including:
- Performance metrics printing
- Trading results plots (equity curve, distributions, etc.)
- Data serialization helpers
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class TradingVisualizer:
    """Handles visualization of trading results and metrics."""

    @staticmethod
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
            return [TradingVisualizer.convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: TradingVisualizer.convert_to_serializable(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def print_metrics(results: dict) -> None:
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
            ('Current Equity DD', results.get('current_equity_drawdown_pct', 0.0), '.2f%')
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

    @staticmethod
    def plot_results(results: dict, save_path: str = None, title: str = None) -> None:
        """Plot trading results and performance metrics."""
        trades = results.get('trades', [])
        if not trades:
            print("No trades to plot. Saving empty plot if path provided.")
            if save_path:
                fig = plt.figure(figsize=(20, 16))
                plt.text(0.5, 0.5, 'No trades to plot', ha='center', va='center')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
            return

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Convert trades to DataFrame for plotting
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df.copy()
        
        # Common style settings for subplots
        subplot_style = {
            'grid': {'alpha': 0.3, 'linestyle': '--'},
            'title_size': 12,
            'label_size': 10,
            'hist_alpha': 0.7,
            'legend_loc': 'upper right',
            'bins': 30
        }
        
        try:
            # Plot equity curve with drawdown overlay
            ax1 = plt.subplot(gs[0, :])
            
            # Calculate equity history
            equity_history = []
            current_balance = results.get('initial_balance', 0.0)
            for trade in trades:
                current_balance += trade.get('pnl', 0)
                equity_history.append(current_balance)
            
            equity_series = pd.Series(equity_history)
            x_range = range(len(equity_series))
            initial_balance = [results.get('initial_balance', 0.0)] * len(x_range)
            
            # Plot balance curve
            ax1.plot(x_range, equity_series, 'b-', label='Balance', linewidth=2)
            ax1.fill_between(x_range, initial_balance, equity_series, alpha=0.3, color='lightblue')
            ax1.axhline(y=results.get('initial_balance', 0.0), color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
            
            # Configure axes
            ax1.set_ylabel('Balance ($)', color='b', fontsize=subplot_style['label_size'])
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            ax1.set_xlabel('Trade Number', fontsize=subplot_style['label_size'])
            ax1.grid(True, **subplot_style['grid'])
            
            # Plot drawdown on secondary axis
            ax2 = ax1.twinx()
            rolling_max = equity_series.expanding().max()
            drawdowns = ((equity_series - rolling_max) / rolling_max) * 100
            ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3, label='Drawdown')
            ax2.set_ylim(bottom=min(drawdowns)*1.1, top=0)
            ax2.set_ylabel('Drawdown %', color='r', fontsize=subplot_style['label_size'])
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Set title
            plt_title = title if title else 'Trading Performance'
            plt.title(plt_title, fontsize=subplot_style['title_size'])
            
            # Plot distributions
            # Trade size distribution
            ax_lots = plt.subplot(gs[1, 0])
            if not trades_df.empty and 'lot_size' in trades_df.columns:
                plt.hist(trades_df['lot_size'], bins=subplot_style['bins'], alpha=subplot_style['hist_alpha'], 
                        color='b', label='Lots')
                mean_lots = trades_df['lot_size'].mean()
                median_lots = trades_df['lot_size'].median()
                plt.axvline(mean_lots, color='b', linestyle='--', label=f'Mean: {mean_lots:.2f}')
                plt.axvline(median_lots, color='b', linestyle=':', label=f'Median: {median_lots:.2f}')
                plt.title('Trade Size Distribution', fontsize=subplot_style['title_size'])
                plt.ylabel('Frequency', fontsize=subplot_style['label_size'])
                plt.xlabel('Lots', fontsize=subplot_style['label_size'])
                plt.grid(**subplot_style['grid'])
                plt.legend(loc=subplot_style['legend_loc'])
            
            # Hold time distribution
            ax_hold = plt.subplot(gs[1, 1])
            if not trades_df.empty and 'hold_time' in trades_df.columns:
                plt.hist(trades_df['hold_time'], bins=subplot_style['bins'], alpha=subplot_style['hist_alpha'],
                        color='purple', label='Bars')
                mean_hold = trades_df['hold_time'].mean()
                median_hold = trades_df['hold_time'].median()
                plt.axvline(mean_hold, color='purple', linestyle='--', label=f'Mean: {mean_hold:.1f}')
                plt.axvline(median_hold, color='purple', linestyle=':', label=f'Median: {median_hold:.1f}')
                plt.title('Hold Time Distribution', fontsize=subplot_style['title_size'])
                plt.ylabel('Frequency', fontsize=subplot_style['label_size'])
                plt.xlabel('Price Bars', fontsize=subplot_style['label_size'])
                plt.grid(**subplot_style['grid'])
                plt.legend(loc=subplot_style['legend_loc'])
            
            # Profit/Loss distributions
            if not trades_df.empty and 'profit_points' in trades_df.columns:
                winning_trades = trades_df[trades_df['profit_points'] > 0]
                losing_trades = trades_df[trades_df['profit_points'] <= 0]
                
                # Plot winning trades (profit points)
                ax_profit = plt.subplot(gs[2, 0])
                if not winning_trades.empty:
                    plt.hist(winning_trades['profit_points'], bins=subplot_style['bins'],
                            alpha=subplot_style['hist_alpha'], color='g', label='Profit Points')
                    mean_profit = winning_trades['profit_points'].mean()
                    median_profit = winning_trades['profit_points'].median()
                    plt.axvline(mean_profit, color='g', linestyle='--', label=f'Mean: {mean_profit:.1f}')
                    plt.axvline(median_profit, color='g', linestyle=':', label=f'Median: {median_profit:.1f}')
                    plt.title('Profit Points Distribution', fontsize=subplot_style['title_size'])
                    plt.ylabel('Frequency', fontsize=subplot_style['label_size'])
                    plt.xlabel('Points', fontsize=subplot_style['label_size'])
                    plt.grid(**subplot_style['grid'])
                    plt.legend(loc=subplot_style['legend_loc'])
                
                # Plot losing trades
                ax_loss = plt.subplot(gs[2, 1])
                if not losing_trades.empty:
                    abs_loss_points = abs(losing_trades['profit_points'])
                    plt.hist(abs_loss_points, bins=subplot_style['bins'],
                            alpha=subplot_style['hist_alpha'], color='r', label='Loss Points')
                    mean_loss = abs_loss_points.mean()
                    median_loss = abs_loss_points.median()
                    plt.axvline(mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.1f}')
                    plt.axvline(median_loss, color='r', linestyle=':', label=f'Median: {median_loss:.1f}')
                    plt.title('Loss Points Distribution', fontsize=subplot_style['title_size'])
                    plt.ylabel('Frequency', fontsize=subplot_style['label_size'])
                    plt.xlabel('Points', fontsize=subplot_style['label_size'])
                    plt.grid(**subplot_style['grid'])
                    plt.legend(loc=subplot_style['legend_loc'])

            plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logging.info(f"Plot saved to {save_path}")
                plt.close()
            else:
                plt.show()
            
        except Exception as e:
            logging.error(f"Error in plotting: {str(e)}")
            if save_path:
                # Create a simple error plot if the main plotting fails
                plt.figure(figsize=(20, 16))
                plt.text(0.5, 0.5, f'Error creating plot: {str(e)}', ha='center', va='center')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
