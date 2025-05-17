"""Backtesting script for single trained trading model."""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path
from trade_model import TradeModel, ModelConfig
from onnx_trade_model import OnnxTradeModel
from trading.environment import TradingEnv, TradingConfig
from trading.trade_tracker import TradeTracker
import time
import sys
import os
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

    # Hold Time and Points Analysis as DataFrame
    print("\n=== Hold Time and Points Analysis ===\n")
    
    # Create data for the DataFrame
    data = {
        'Metric Type': ['Winners Hold', 'Losers Hold', 'Winners Points', 'Losers Points'],
        '0th Pct': [
            results.get('win_hold_time_0th', 0.0),
            results.get('loss_hold_time_0th', 0.0),
            results.get('win_points_0th', 0.0),
            results.get('loss_points_0th', 0.0)
        ],
        '1st Pct': [
            results.get('win_hold_time_1st', 0.0),
            results.get('loss_hold_time_1st', 0.0),
            results.get('win_points_1st', 0.0),
            results.get('loss_points_1st', 0.0)
        ],
        '10th Pct': [
            results.get('win_hold_time_10th', 0.0),
            results.get('loss_hold_time_10th', 0.0),
            results.get('win_points_10th', 0.0),
            results.get('loss_points_10th', 0.0)
        ],
        '20th Pct': [
            results.get('win_hold_time_20th', 0.0),
            results.get('loss_hold_time_20th', 0.0),
            results.get('win_points_20th', 0.0),
            results.get('loss_points_20th', 0.0)
        ],
        'Median': [
            results.get('win_hold_time_median', 0.0),
            results.get('loss_hold_time_median', 0.0),
            results.get('median_win_points', 0.0),
            results.get('median_loss_points', 0.0)
        ],
        '80th Pct': [
            results.get('win_hold_time_80th', 0.0),
            results.get('loss_hold_time_80th', 0.0),
            results.get('win_points_80th', 0.0),
            results.get('loss_points_80th', 0.0)
        ],
        '90th Pct': [
            results.get('win_hold_time_90th', 0.0),
            results.get('loss_hold_time_90th', 0.0),
            results.get('win_points_90th', 0.0),
            results.get('loss_points_90th', 0.0)
        ],
        '99th Pct': [
            results.get('win_hold_time_99th', 0.0),
            results.get('loss_hold_time_99th', 0.0),
            results.get('win_points_99th', 0.0),
            results.get('loss_points_99th', 0.0)
        ],
        '100th Pct': [
            results.get('win_hold_time_100th', 0.0),
            results.get('loss_hold_time_100th', 0.0),
            results.get('win_points_100th', 0.0),
            results.get('loss_points_100th', 0.0)
        ]
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('Metric Type')
    
    # Format all numeric columns to 1 decimal place
    for col in df.columns:
        df[col] = df[col].map('{:,.1f}'.format)
        
    print(df.to_string())
    
    print("")

    # Open Position Details (if any)
    if results.get('active_positions', 0) > 0 and 'position' in results:
        print("\n=== Open Position Details ===")
        pos = results['position']
        position_metrics = [
            ('Direction', 'Long' if pos['direction'] == 1 else 'Short'),
            ('Entry Price', f"{pos['entry_price']:.5f}"),
            ('Lot Size', f"{pos['lot_size']:.2f}"),
            ('Hold Time', f"{pos['hold_time']} bars"),
            ('Unrealized PnL', f"{pos['unrealized_pnl']:+.2f}"),
            ('Profit Points', f"{pos['profit_points']:+.1f}")
        ]
        for name, value in position_metrics:
            print(f"{name}: {value}")
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
    trades = results.get('trades', [])
    if not trades:
        print("No trades to plot. Saving empty plot if path provided.")
        if save_path:
            fig = plt.figure(figsize=(20, 16))
            plt.text(0.5, 0.5, 'No trades to plot', ha='center', va='center')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        return

    fig = plt.figure(figsize=(20, 16))  # Adjusted height for four plots
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # Convert trades to DataFrame for plotting
    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.copy()  # Create explicit copy to avoid warnings
    
    # Debug info
    print(f"\nPlotting {len(trades)} trades")
    if not trades_df.empty:
        print(f"DataFrame columns: {trades_df.columns.tolist()}")
    
    try:
        # Extract equity history from trades and track streaks (including unrealized PnL)
        equity_history = []
        current_balance = results.get('initial_balance', 0.0)
        current_equity = current_balance
        equity_history.append(current_equity)
        
        unrealized_pnl = 0.0  # Track unrealized PnL
        for i, trade in enumerate(trades):
            try:
                pnl = trade.get('pnl', 0)
                current_balance += pnl
                
                # For the last trade, include unrealized PnL if it's active
                if i == len(trades) - 1 and results.get('active_positions', 0) > 0:
                    unrealized_pnl = trade.get('unrealized_pnl', 0)
                
                current_equity = current_balance + unrealized_pnl
                equity_history.append(current_equity)
            except (KeyError, TypeError) as e:
                logging.error(f"Error processing trade {i} for equity calculation: {str(e)}")
                logging.error(f"Trade data: {trade}")
            except Exception as e:
                logging.error(f"Unexpected error in equity calculation for trade {i}: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Error calculating equity history: {str(e)}")
        # Provide fallback equity history if calculation fails
        equity_history = [results.get('initial_balance', 0.0)] * (len(trades) + 1)
    
    # Plot balance curve and performance metrics with error handling
    try:
        # Plot balance curve with drawdown overlay
        ax1 = plt.subplot(gs[0, :])
        
        # Convert to pandas Series for calculations
        try:
            equity_series = pd.Series(equity_history)
            x_range = range(len(equity_series))
            initial_balance = [results.get('initial_balance', 0.0)] * len(x_range)
            
            # Calculate drawdown with error handling
            try:
                rolling_max = equity_series.expanding().max()
                if (rolling_max == 0).any():
                    logging.error("Zero values found in rolling max calculation")
                    drawdowns = pd.Series([0.0] * len(equity_series))
                else:
                    drawdowns = ((equity_series - rolling_max) / rolling_max) * 100
            except Exception as e:
                logging.error(f"Error calculating drawdowns: {str(e)}")
                drawdowns = pd.Series([0.0] * len(equity_series))
                
            # Configure main balance axis with improved formatting
            ax1.fill_between(x_range, initial_balance, equity_series, alpha=0.3, color='lightblue')
            ax1.plot(x_range, equity_series, 'b-', label='Balance', linewidth=2)
            ax1.axhline(y=results.get('initial_balance', 0.0), color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
            
            # Common style settings for subplots
            subplot_style = {
                'grid': {'alpha': 0.3, 'linestyle': '--'},
                'title_size': 12,
                'label_size': 10,
                'hist_alpha': 0.7,
                'legend_loc': 'upper right',
                'bins': 30
            }
            
            ax1.set_ylabel('Balance ($)', color='b', fontsize=subplot_style['label_size'])
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Format x-axis to show trade numbers
            ax1.set_xlabel('Trade Number', fontsize=subplot_style['label_size'])
            ax1.set_xlim(0, len(equity_series))
            ax1.grid(True, **subplot_style['grid'])
            
            # Configure drawdown axis with improved visibility
            ax2 = ax1.twinx()
            ax2.set_ylim(bottom=min(drawdowns)*1.1, top=0)  # Invert and add 10% padding
            
            # Plot drawdown
            ax2.fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown %', color='r', fontsize=subplot_style['label_size'])
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            plt.title('Equity and Drawdown', fontsize=subplot_style['title_size'])
            
        except Exception as e:
            logging.error(f"Error plotting equity curve: {str(e)}")
            plt.text(0.5, 0.5, 'Error plotting equity curve', ha='center', va='center')
            
    except Exception as e:
        logging.error(f"Error in plot setup: {str(e)}")
    
    # Plot trade size distribution (left column) with error handling
    try:
        ax_lots = plt.subplot(gs[1, 0])
        if not trades_df.empty and 'lot_size' in trades_df.columns:
            try:
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
            except Exception as e:
                logging.error(f"Error plotting lot size distribution: {str(e)}")
                plt.text(0.5, 0.5, 'Error plotting lot sizes', ha='center', va='center')
                plt.title('Trade Size Distribution (Error)', fontsize=subplot_style['title_size'])
    except Exception as e:
        logging.error(f"Error setting up lot size plot: {str(e)}")
    
    # Plot hold time distribution (right column) with error handling
    try:
        ax_hold = plt.subplot(gs[1, 1])
        if not trades_df.empty and 'hold_time' in trades_df.columns:
            try:
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
            except Exception as e:
                logging.error(f"Error plotting hold time distribution: {str(e)}")
                plt.text(0.5, 0.5, 'Error plotting hold times', ha='center', va='center')
    except Exception as e:
        logging.error(f"Error setting up hold time plot: {str(e)}")
    
    # Create side-by-side plots for profit and loss points (third row, split into columns) with error handling
    try:
        if not trades_df.empty and 'profit_points' in trades_df.columns:
            try:
                winning_trades = trades_df[trades_df['profit_points'] > 0]
                losing_trades = trades_df[trades_df['profit_points'] <= 0]
                
                # Plot winning trades (profit points) - left column
                ax_profit = plt.subplot(gs[2, 0])
                if not winning_trades.empty:
                    try:
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
                    except Exception as e:
                        logging.error(f"Error plotting winning trades distribution: {str(e)}")
                        plt.text(0.5, 0.5, 'Error plotting winning trades', ha='center', va='center')
                
                # Plot losing trades (absolute loss points) - right column
                ax_loss = plt.subplot(gs[2, 1])
                if not losing_trades.empty:
                    try:
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
                    except Exception as e:
                        logging.error(f"Error plotting losing trades distribution: {str(e)}")
                        plt.text(0.5, 0.5, 'Error plotting losing trades', ha='center', va='center')
            except Exception as e:
                logging.error(f"Error processing trades for profit/loss plots: {str(e)}")
                plt.text(0.5, 0.5, 'Error processing trade data', ha='center', va='center')
    except Exception as e:
        logging.error(f"Error in profit/loss plot setup: {str(e)}")
        # Create empty subplots with error messages if plotting fails
        ax_profit = plt.subplot(gs[2, 0])
        ax_profit.text(0.5, 0.5, 'Error plotting profit distribution', ha='center', va='center')
        ax_profit.set_title('Profit Points Distribution (Error)', fontsize=subplot_style['title_size'])
        
        ax_loss = plt.subplot(gs[2, 1])
        ax_loss.text(0.5, 0.5, 'Error plotting loss distribution', ha='center', va='center')
        ax_loss.set_title('Loss Points Distribution (Error)', fontsize=subplot_style['title_size'])

    # Adjust layout with padding and save/show with error handling
    try:
        plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)
        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight')
                logging.info(f"Plot saved to {save_path}")
            except Exception as e:
                logging.error(f"Error saving plot to {save_path}: {str(e)}")
        plt.show()
    except Exception as e:
        logging.error(f"Error in final plot rendering: {str(e)}")

def show_progress(message="Running backtest"):
    """Simple progress indicator for long-running processes."""
    chars = "|/-\\"
    for char in chars:
        sys.stdout.write(f'\r{message}... {char}')
        sys.stdout.flush()
        time.sleep(0.1)

def backtest_with_predictions(model: Union[TradeModel, OnnxTradeModel], data: pd.DataFrame, args,
                            initial_balance: float = 10000.0, balance_per_lot: float = 1000.0, 
                            verbose: bool = False, trades_log_path: str = None,
                            point_value: float = 0.01,
                            min_lots: float = 0.01, max_lots: float = 200.0,
                            contract_size: float = 100.0,
                            currency_conversion: float = None,
                            spread_variation: float = 0.0,
                            slippage_range: float = 0.0,
                            balance_recheck_bars: int = 0,
                            reset_states_on_gap: bool = False,
                            window_size: int = 50) -> Dict[str, Any]: # Added window_size
    """Run a backtest using the predict_single method to simulate the live trading process."""
    
    # Perform LSTM warmup if model supports it
    if isinstance(model, TradeModel) and model.is_lstm_model():
        warmup_window = min(100, len(data) // 10)  # Use 100 bars or 10% of data
        model.warmup_lstm_state(data.iloc[:-warmup_window], warmup_window)
        print(f"\nWarmed up LSTM states using {warmup_window} bars")
        
    # Create trading environment config and initialize environment
    config = TradingConfig(
        initial_balance=initial_balance,
        balance_per_lot=balance_per_lot,
        point_value=point_value,
        min_lots=min_lots,
        max_lots=max_lots,
        contract_size=contract_size,
        window_size=window_size,
        currency_conversion=currency_conversion if currency_conversion else 1.0,
        spread_variation=spread_variation,
        slippage_range=slippage_range
    )
    
    env = TradingEnv(
        data=data,
        predict_mode=True,
        config=config
    )
    obs, _ = env.reset()
    action_handler = env.action_handler

    # Initialize variables
    current_position = None
    total_steps = 0
    total_reward = 0.0
    
    # Initialize trade tracker if path provided
    trade_tracker = None
    if trades_log_path:
        os.makedirs(trades_log_path, exist_ok=True)
        trade_tracker = TradeTracker(trades_log_path)
    
    # Progress tracking (using aligned data length)
    progress_steps = max(1, env.data_length // 100)  # Update every 1%
    
    # Ensure we have enough data after preprocessing
    if env.data_length < 100:  # Minimum data requirement
        raise ValueError(f"Insufficient data after preprocessing: need at least 100 bars, got {env.data_length}")
    
    # Main prediction loop
    while True:
        try:
            if total_steps % progress_steps == 0:
                progress = (total_steps / env.data_length) * 100
                print(f"\rProgress: {progress:.1f}% (step {total_steps}/{env.data_length})", end="")
            
            # Log raw features periodically if verbose
            if verbose:
                current_index = env.original_index[total_steps]  # Use aligned index
                current_data = data.loc[:current_index]
                feature_processor = env.feature_processor
                atr, rsi, (upper_band, lower_band), trend_strength = feature_processor._calculate_indicators(
                    current_data['high'].values,
                    current_data['low'].values,
                    current_data['close'].values
                )
                print(f"\nRaw features at step {total_steps}:")
                print(f"ATR: {atr[-1]:.6f}")
                print(f"RSI: {rsi[-1]:.6f}")
                print(f"BB Upper: {upper_band[-1]:.6f}")
                print(f"BB Lower: {lower_band[-1]:.6f}")
                print(f"Trend Strength: {trend_strength[-1]:.6f}")
            
            # Recheck balance if configured
            if balance_recheck_bars > 0 and total_steps % balance_recheck_bars == 0:
                env.metrics.balance = env.metrics.get_current_balance()
            
            # Add random spread variation if configured
            if spread_variation > 0:
                # Apply spread variation using environment's aligned data
                current_spread = env.prices['spread'][total_steps]
                variation = np.random.uniform(-spread_variation, spread_variation)
                env.prices['spread'][total_steps] = max(0, current_spread + variation)
            
            # Check for market gaps and reset LSTM states if needed
            if reset_states_on_gap and hasattr(model, 'is_recurrent') and model.is_recurrent:
                # Get current and previous timestamp
                if total_steps > 0:
                    current_time = env.original_index[total_steps]
                    prev_time = env.original_index[total_steps-1]
                    time_diff = (current_time - prev_time).total_seconds() / 60
                    expected_diff = 15  # For 15-minute data
                    
                    # Reset LSTM states if gap is significantly larger than expected
                    if time_diff > expected_diff * 2:  # Gap is more than double the expected timeframe
                        if verbose:
                            print(f"Market gap detected: {time_diff/60:.2f}h - Resetting LSTM states")
                        model.reset_lstm_states()
            
            # Create normalized feature dictionary for tracking
            feature_dict = {}
            try:
                if obs is not None and isinstance(obs, np.ndarray):
                    feature_names = env.feature_processor.get_feature_names()
                    for i, feat in enumerate(obs):
                        try:
                            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                            feature_dict[feature_name] = float(feat)
                            if verbose:
                                print(f"  {feature_name}: {feat:.6f}")
                        except IndexError as e:
                            logging.error(f"Feature index error at step {total_steps}: {str(e)}")
                            logging.error(f"Feature array length: {len(obs)}, Feature names length: {len(feature_names)}")
                        except ValueError as e:
                            logging.error(f"Feature value error at step {total_steps}: {str(e)}")
                        except Exception as e:
                            logging.error(f"Error processing feature at step {total_steps}: {str(e)}")
            except Exception as e:
                logging.error(f"Error creating feature dictionary at step {total_steps}: {str(e)}")

            # Get prediction from model - handle both standard PPO and ONNX models
            try:
                if hasattr(model, 'model') and model.model is not None:
                    try:
                        # Standard PPO/PPO-LSTM model
                        action, lstm_states = model.model.predict(
                            obs,
                            state=lstm_states if 'lstm_states' in locals() else None,
                            deterministic=True
                        )
                        # Convert action to discrete value
                        action_value = int(action.item()) if isinstance(action, np.ndarray) else int(action)
                    except (ValueError, IndexError, AttributeError) as e:
                        logging.error(f"Error in PPO model prediction at step {total_steps}: {str(e)}")
                        logging.error(f"Observation shape: {obs.shape if isinstance(obs, np.ndarray) else 'not numpy array'}")
                        action_value = 0  # Default to HOLD on error
                else:
                    try:
                        # ONNX model - get logits and add debug info
                        logits, info = model.predict(obs)
                        action_value = int(np.argmax(logits, axis=-1)[0])
                        
                        if args.verbose_features:
                            print(f"\nAction selection step {total_steps}:")
                            print(f"Raw logits: {logits[0]}")
                            print(f"Selected action: {action_value}")
                    except (IndexError, ValueError, AttributeError) as e:
                        logging.error(f"Error in ONNX model prediction at step {total_steps}: {str(e)}")
                        logging.error(f"Observation shape: {obs.shape if isinstance(obs, np.ndarray) else 'not numpy array'}")
                        action_value = 0  # Default to HOLD on error
            except Exception as e:
                logging.error(f"Unexpected error during model prediction at step {total_steps}: {str(e)}")
                action_value = 0  # Default to HOLD on error
            
            try:
                discrete_action = action_value % 4
                
                # Execute step and let environment handle action validation naturally
                obs, reward, done, truncated, info = env.step(discrete_action)
                total_reward += reward
                total_steps += 1

                # Track trade events
                try:
                    # Get current timestamp from the data index
                    current_timestamp = env.original_index[total_steps] if total_steps < env.data_length else env.original_index[-1]

                    # Check for position changes
                    if env.current_position and not current_position:  # New position opened
                        try:
                            if verbose:
                                print(f"\nOpening {current_timestamp}: {'Long' if discrete_action == 1 else 'Short'} "
                                    f"{env.current_position['lot_size']:.2f} lots at {env.current_position['entry_price']:.5f}")
                            # Track in optional trade log
                            if trade_tracker:
                                try:
                                    trade_tracker.log_trade_entry(
                                        'buy' if discrete_action == 1 else 'sell',
                                        feature_dict,
                                        env.current_position['entry_price'],
                                        env.current_position['lot_size'],
                                        timestamp=current_timestamp
                                    )
                                except Exception as e:
                                    logging.error(f"Error logging trade entry at step {total_steps}: {str(e)}")
                        except (KeyError, TypeError) as e:
                            logging.error(f"Error processing new position data at step {total_steps}: {str(e)}")
                        except Exception as e:
                            logging.error(f"Unexpected error handling new position at step {total_steps}: {str(e)}")

                    elif not env.current_position and current_position:  # Position closed
                        try:
                            close_price = info.get('close_price', env.prices['close'][min(total_steps, env.data_length-1)])
                            pnl = info.get('pnl', 0.0)
                            if verbose:
                                print(f"\nClosing {current_timestamp}: PnL={pnl:+.2f} at {close_price:.5f}")
                            # Track in optional trade log
                            if trade_tracker:
                                try:
                                    trade_tracker.log_trade_exit(
                                        'model_close',
                                        close_price,
                                        pnl,
                                        feature_dict,
                                        timestamp=current_timestamp
                                    )
                                except Exception as e:
                                    logging.error(f"Error logging trade exit at step {total_steps}: {str(e)}")
                        except (KeyError, IndexError) as e:
                            logging.error(f"Error processing closed position data at step {total_steps}: {str(e)}")
                        except Exception as e:
                            logging.error(f"Unexpected error handling closed position at step {total_steps}: {str(e)}")
                            
                    elif env.current_position and trade_tracker:  # Position update (optional)
                        try:
                            trade_tracker.log_trade_update(
                                feature_dict,
                                env.prices['close'][min(total_steps, env.data_length-1)],
                                env.metrics.current_unrealized_pnl,
                                timestamp=current_timestamp
                            )
                        except Exception as e:
                            logging.error(f"Error logging trade update at step {total_steps}: {str(e)}")
                except Exception as e:
                    logging.error(f"Error in trade tracking at step {total_steps}: {str(e)}")
                    
                # Update position tracking
                current_position = env.current_position.copy() if env.current_position else None
                
                if done or truncated:
                    break
                    
            except (ValueError, TypeError) as e:
                print(f"\nError processing action at step {total_steps}: {str(e)}")
                discrete_action = 0  # Default to HOLD on error
                obs, reward, done, truncated, info = env.step(discrete_action)
                total_steps += 1
                
            except Exception as e:
                print(f"\nUnexpected error at step {total_steps}: {str(e)}")
                print("Continuing with next step...")
                total_steps += 1
                continue
                
        except Exception as e:
            print(f"\nError at step {total_steps}: {str(e)}")
            print("Continuing with next step...")
            total_steps += 1
            continue
    
    print(f"\nBacktest completed: {total_steps} steps processed")
    
    # Handle any open position at the end
    if env.current_position:
        env.current_step = env.data_length - 1
        pnl, trade_info = action_handler.close_position()
        if pnl != 0:
            env.trades.append(trade_info)
            env.metrics.add_trade(trade_info)
            env.metrics.update_balance(pnl)
    
    # Calculate metrics and ensure trades are included
    metrics = model._calculate_backtest_metrics(env, total_steps, total_reward)
    
    # Add trades list to results
    metrics['trades'] = env.trades
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Backtest trained trading model')
    
    # Default values converted to ZAR using rate of 19
    default_initial_balance = 10000.0 * 19  # 10,000 USD in ZAR
    default_balance_per_lot = 1000.0 * 19   # 1,000 USD per lot in ZAR
    
    # Trading environment settings
    parser.add_argument('--model_path', type=str, required=True,
                     help='Path to the model file to backtest (.zip for PPO or .onnx for ONNX models)')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--initial_balance', type=float, default=default_initial_balance,
                      help='Initial account balance in account currency (default: 190,000 ZAR)')
    parser.add_argument('--balance_per_lot', type=float, default=default_balance_per_lot,
                      help='Account balance required per 0.01 lot in account currency (default: 19,000 ZAR)')
    parser.add_argument('--currency_conversion', type=float, default=19.0,
                      help='Conversion rate from USD to account currency (default: 19.0 ZAR/USD)')
                        # ONNX model settings (only needed if auto-detection fails)
    parser.add_argument('--is_recurrent', action='store_true',
                      help='Force ONNX model to be treated as recurrent/LSTM-based (auto-detected by default)')
    parser.add_argument('--hidden_size', type=int, default=64,
                      help='LSTM hidden size for recurrent ONNX models (default: 64)')
    parser.add_argument('--num_layers', type=int, default=1,
                      help='Number of LSTM layers for recurrent ONNX models (default: 1)')
                      
    # Market simulation settings
    parser.add_argument('--reset_states_on_gap', action='store_true',
                      help='Reset LSTM states when market gaps are detected')
    parser.add_argument('--spread_variation', type=float, default=0.0,
                      help='Random spread variation range (e.g., 0.2 for ±0.2 spread variation)')
    parser.add_argument('--slippage_range', type=float, default=0.0,
                      help='Maximum price slippage range in points')
    parser.add_argument('--balance_recheck_bars', type=int, default=0,
                      help='Recheck balance every N bars (0 to disable)')
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
    parser.add_argument('--point_value', type=float, default=0.01,
                      help='Value of one price point movement (default: 0.01)')
    parser.add_argument('--min_lots', type=float, default=0.01,
                      help='Minimum lot size (default: 0.01)')
    parser.add_argument('--max_lots', type=float, default=200.0,
                      help='Maximum lot size (default: 200.0)')
    parser.add_argument('--contract_size', type=float, default=100.0,
                      help='Standard contract size (default: 100.0)')
    parser.add_argument('--method', type=str, choices=['evaluate', 'predict_single'], default='evaluate',
                      help='Backtesting method to use: evaluate (quiet) or predict_single (simulates live trading)')
    parser.add_argument('--verbose_features', action='store_true',
                      help='Log detailed feature values during prediction (only applicable with predict_single method)')
    parser.add_argument('--trades_log_path', type=str, default=None,
                      help='Directory path to store trade tracking logs (use with predict_single method)')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Number of past timesteps for market features in observation (default: 50)')
    
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
            print("These columns are required for backtesting. Will attempt to continue.")        # Auto-detect model type based on file extension
        use_onnx = args.model_path.lower().endswith('.onnx')
        
        # Initialize model
        print(f"\nInitializing model from: {args.model_path}")
        
        if use_onnx:
            print("Using ONNX model integration")
            
            # Attempt to auto-detect if the ONNX model is recurrent by checking for LSTM inputs
            is_recurrent = args.is_recurrent
            if not args.is_recurrent:
                try:
                    import onnx
                    onnx_model = onnx.load(args.model_path)
                    input_names = [input.name for input in onnx_model.graph.input]
                    is_recurrent = any(name in ['lstm_h', 'lstm_c'] for name in input_names)
                    if is_recurrent:
                        print("Auto-detected recurrent ONNX model (found LSTM inputs)")
                except Exception as e:
                    print(f"Error during ONNX model inspection: {e}")
                    print("Continuing with user-specified settings")
            
            print(f"LSTM recurrent model: {'Yes' if is_recurrent else 'No'}")
            
            model = OnnxTradeModel(
                model_path=args.model_path,
                is_recurrent=is_recurrent,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                balance_per_lot=args.balance_per_lot,
                initial_balance=args.initial_balance,
                point_value=args.point_value,
                min_lots=args.min_lots,
                max_lots=args.max_lots,
                contract_size=args.contract_size,
                window_size=args.window_size # Added for OnnxTradeModel if it uses it
            )
        else:
            print("Using standard PPO model")
            config = ModelConfig(
                model_path=Path(args.model_path),
                balance_per_lot=args.balance_per_lot,
                initial_balance=args.initial_balance,
                point_value=args.point_value,
                min_lots=args.min_lots,
                max_lots=args.max_lots,
                contract_size=args.contract_size,
                window_size=args.window_size
            )
            model = TradeModel(config)
          # Run backtest based on selected method
        if args.method == 'predict_single':
            # Use the method that simulates live trading
            print("\nRunning prediction backtest (simulates live trading)...")
            results = backtest_with_predictions(
                model=model,
                data=df,
                args=args,
                initial_balance=args.initial_balance,
                balance_per_lot=args.balance_per_lot,
                verbose=args.verbose_features,
                point_value=args.point_value,
                min_lots=args.min_lots,
                max_lots=args.max_lots,
                contract_size=args.contract_size,
                currency_conversion=args.currency_conversion,
                spread_variation=args.spread_variation,
                slippage_range=args.slippage_range,
                balance_recheck_bars=args.balance_recheck_bars,
                trades_log_path=args.trades_log_path,
                reset_states_on_gap=args.reset_states_on_gap,
                window_size=args.window_size # Added window_size
            )
        else:  # method == 'evaluate' or args.quiet
            print("\nRunning quiet backtest with evaluate method...")
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
                    spread_variation=args.spread_variation,
                    slippage_range=args.slippage_range
                )
            finally:
                # Always stop the progress indicator
                if progress_thread and progress_thread.is_alive():
                    stop_progress_indicator()
        
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
