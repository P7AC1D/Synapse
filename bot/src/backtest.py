"""Backtesting script for single trained trading model."""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Dict, Any
from trade_model import TradeModel
from trading.environment import TradingEnv
from trading.trade_tracker import TradeTracker
import time
import sys
import threading
import traceback
from utils.progress import show_progress_continuous, stop_progress_indicator

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Configure enhanced logging with call stack support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest.log', mode='a')
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

def log_exception(e: Exception, context: str = "", step: int = None) -> None:
    """Log exception with full call stack and context information."""
    error_msg = f"Exception in {context}"
    if step is not None:
        error_msg += f" at step {step}"
    error_msg += f": {type(e).__name__}: {str(e)}"
    
    logger.error(error_msg)
    logger.error("Call stack:")
    
    # Get the full traceback
    tb_lines = traceback.format_exc().splitlines()
    for line in tb_lines:
        logger.error(f"  {line}")
    
    # Additional context logging
    logger.error(f"Exception type: {type(e).__module__}.{type(e).__name__}")
    logger.error(f"Exception args: {e.args}")
    
def safe_execute(func, *args, context: str = "", default_return=None, **kwargs):
    """Safely execute a function with comprehensive error logging."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_exception(e, context)
        logger.warning(f"Returning default value: {default_return}")
        return default_return

def convert_to_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format."""
    try:
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
    except Exception as e:
        log_exception(e, f"converting object of type {type(obj)} to serializable format")
        logger.warning(f"Returning string representation of object: {str(obj)}")
        return str(obj)

def print_metrics(results: dict):
    """Print formatted backtest performance metrics."""
    try:
        logger.info("Starting to print performance metrics")
        
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
            try:
                if 'd' in format_spec:
                    print(f"{name}: {value:d}")
                elif '%' in format_spec:
                    print(f"{name}: {value:{format_spec[:-1]}}%")
                else:
                    print(f"{name}: {value:{format_spec}}")
            except Exception as e:
                log_exception(e, f"formatting metric {name} with value {value}")
                print(f"{name}: {value}")
        print("")

        # Risk Metrics
        try:
            print("=== Risk Metrics ===")
            risk_metrics = [
                ('Max Balance Drawdown', results.get('max_drawdown_pct', 0.0), '.2f%'),
                ('Max Equity Drawdown', results.get('max_equity_drawdown_pct', 0.0), '.2f%'),
                ('Current Balance DD', results.get('current_drawdown_pct', 0.0), '.2f%'),
                ('Current Equity DD', results.get('current_equity_drawdown_pct', 0.0), '.2f%')  # Fixed typo
            ]
            for name, value, format_spec in risk_metrics:
                try:
                    if '%' in format_spec:
                        print(f"{name}: {value:{format_spec[:-1]}}%")
                    else:
                        print(f"{name}: {value:{format_spec}}")
                except Exception as e:
                    log_exception(e, f"formatting risk metric {name}")
                    print(f"{name}: {value}")
            print("")
        except Exception as e:
            log_exception(e, "printing risk metrics section")        # Directional Analysis
        try:
            print("=== Directional Analysis ===")
            directional_metrics = [
                ('Long Trades', results.get('long_trades', 0), 'd'),
                ('Long Win Rate', results.get('long_win_rate', 0.0), '.2f%'),
                ('Short Trades', results.get('short_trades', 0), 'd'),
                ('Short Win Rate', results.get('short_win_rate', 0.0), '.2f%')
            ]
            for name, value, format_spec in directional_metrics:
                try:
                    if 'd' in format_spec:
                        print(f"{name}: {value:d}")
                    elif '%' in format_spec:
                        print(f"{name}: {value:{format_spec[:-1]}}%")
                    else:
                        print(f"{name}: {value:{format_spec}}")
                except Exception as e:
                    log_exception(e, f"formatting directional metric {name}")
                    print(f"{name}: {value}")
            print("")
        except Exception as e:
            log_exception(e, "printing directional analysis section")        # Hold Time and Points Analysis as DataFrame
        try:
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
        except Exception as e:
            log_exception(e, "creating and printing hold time analysis table")

        # Open Position Details (if any)
        try:
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
        except Exception as e:
            log_exception(e, "printing open position details")
            
        # Consecutive Trade Analysis
        try:
            print("=== Consecutive Trade Analysis ===")
            consecutive_metrics = [
                ('Max Consecutive Wins', results.get('max_consecutive_wins', 0), 'd'),
                ('Max Consecutive Losses', results.get('max_consecutive_losses', 0), 'd'),
                ('Current Win Streak', results.get('current_consecutive_wins', 0), 'd'),
                ('Current Loss Streak', results.get('current_consecutive_losses', 0), 'd')
            ]
            for name, value, format_spec in consecutive_metrics:
                try:
                    if 'd' in format_spec:
                        print(f"{name}: {value:d}")
                    elif '%' in format_spec:
                        print(f"{name}: {value:{format_spec[:-1]}}%")
                    else:
                        print(f"{name}: {value:{format_spec}}")
                except Exception as e:
                    log_exception(e, f"formatting consecutive metric {name}")
                    print(f"{name}: {value}")
            print("")
        except Exception as e:
            log_exception(e, "printing consecutive trade analysis")
        
        logger.info("Performance metrics printed successfully")
        
    except Exception as e:
        log_exception(e, "print_metrics function")
        print(f"Error printing metrics: {str(e)}")

def plot_results(results: dict, save_path: str = None):
    """Plot backtest results and performance metrics."""
    try:
        logger.info("Starting plot generation")
        fig = plt.figure(figsize=(20, 16))  # Adjusted height for four plots
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Convert trades to DataFrame for plotting
        trades = results.get('trades', [])
        
        # Handle empty trades case gracefully
        if not trades:
            logger.warning("No trades to plot. Skipping visualization.")
            print("No trades to plot. Skipping visualization.")
            return
            
        trades_df = pd.DataFrame(trades)
        logger.info(f"Plotting {len(trades)} trades")
        
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
            except Exception as e:
                log_exception(e, f"processing trade {i} in equity history calculation")
                continue
        
        try:
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
            logger.info("Successfully created equity and drawdown plot")
        except Exception as e:
            log_exception(e, "creating equity and drawdown plot")
        
        try:
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
                logger.info("Successfully created trade size distribution plot")
        except Exception as e:
            log_exception(e, "creating trade size distribution plot")
        
        try:
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
                logger.info("Successfully created hold time distribution plot")
        except Exception as e:
            log_exception(e, "creating hold time distribution plot")
        
        try:
            # Create side-by-side plots for profit and loss points (third row, split into columns)
            if not trades_df.empty and 'profit_points' in trades_df.columns:
                winning_trades = trades_df[trades_df['profit_points'] > 0]
                losing_trades = trades_df[trades_df['profit_points'] <= 0]
                
                # Plot winning trades (profit points) - left column
                ax_profit = plt.subplot(gs[2, 0])
                if not winning_trades.empty:
                    plt.hist(winning_trades['profit_points'], bins=30, alpha=0.7, color='g', label='Profit Points')
                    mean_profit = winning_trades['profit_points'].mean()
                    median_profit = winning_trades['profit_points'].median()
                    plt.axvline(mean_profit, color='g', linestyle='--', label=f'Mean: {mean_profit:.1f}')
                    plt.axvline(median_profit, color='g', linestyle=':', label=f'Median: {median_profit:.1f}')
                    plt.title('Profit Distribution')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True)
                    logger.info("Successfully created profit distribution plot")
                
                # Plot losing trades (absolute loss points) - right column
                ax_loss = plt.subplot(gs[2, 1])
                if not losing_trades.empty:
                    abs_loss_points = abs(losing_trades['profit_points'])
                    plt.hist(abs_loss_points, bins=30, alpha=0.7, color='r', label='Loss Points')
                    mean_loss = abs_loss_points.mean()
                    median_loss = abs_loss_points.median()
                    plt.axvline(mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.1f}')
                    plt.axvline(median_loss, color='r', linestyle=':', label=f'Median: {median_loss:.1f}')
                    plt.title('Loss Distribution')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True)
                    logger.info("Successfully created loss distribution plot")
        except Exception as e:
            log_exception(e, "creating profit/loss distribution plots")

        # Adjust layout with padding
        try:
            plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=2.0)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            plt.show()
            logger.info("Plot generation completed successfully")
        except Exception as e:
            log_exception(e, "finalizing and displaying plot")
            
    except Exception as e:
        log_exception(e, "plot_results function")
        print(f"Error generating plots: {str(e)}")
        logger.error("Plot generation failed completely")

def show_progress(message="Running backtest"):
    """Simple progress indicator for long-running processes."""
    chars = "|/-\\"
    for char in chars:
        sys.stdout.write(f'\r{message}... {char}')
        sys.stdout.flush()
        time.sleep(0.1)

def backtest_with_predictions(model: TradeModel, data: pd.DataFrame, initial_balance: float = 10000.0, 
                            balance_per_lot: float = 1000.0, verbose: bool = False,
                            trades_log_path: str = None,
                            point_value: float = 0.01,
                            min_lots: float = 0.01, max_lots: float = 200.0,
                            contract_size: float = 100.0,
                            currency_conversion: float = None,
                            reset_states_on_gap: bool = True,
                            spread_variation: float = 0.0,
                            slippage_range: float = 0.0,
                            balance_recheck_bars: int = 0) -> Dict[str, Any]:
    """Run a backtest using the predict_single method to simulate the live trading process."""
    
    try:
        logger.info("Starting backtest with predictions")
        logger.info(f"Data shape: {data.shape}, Initial balance: {initial_balance}")
        
        # Create a trading environment for tracking trades and metrics
        env = safe_execute(
            TradingEnv,
            data=data,
            initial_balance=initial_balance,
            balance_per_lot=balance_per_lot,
            random_start=False,
            point_value=point_value,
            min_lots=min_lots,
            max_lots=max_lots,
            contract_size=contract_size,
            currency_conversion=currency_conversion,
            context="creating TradingEnv"
        )
        
        if env is None:
            logger.error("Failed to create trading environment")
            raise RuntimeError("Failed to initialize trading environment")
            
        obs, _ = env.reset()
        action_handler = env.action_handler

        # Initialize variables
        current_position = None
        total_steps = 0
        total_reward = 0.0
        
        # Initialize trade tracker if path provided
        trade_tracker = None
        if trades_log_path:
            try:
                os.makedirs(trades_log_path, exist_ok=True)
                trade_tracker = TradeTracker(trades_log_path)
                logger.info(f"Trade tracker initialized at {trades_log_path}")
            except Exception as e:
                log_exception(e, "initializing trade tracker")
                logger.warning("Continuing without trade tracking")
        
        # Progress tracking
        progress_steps = max(1, len(data) // 100)  # Update every 1%
        
        # Initialize LSTM states - match how the bot initializes them
        try:
            model.reset_states()
            model.lstm_states = None  # Ensure clean start
            logger.info("Model states reset successfully")
        except Exception as e:
            log_exception(e, "resetting model states")
            raise RuntimeError("Failed to reset model states")
        
        # Ensure we have enough data
        if len(data) < 100:  # Minimum data requirement
            error_msg = f"Insufficient data: need at least 100 bars, got {len(data)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Starting main prediction loop with {len(data)} data points")
        
        # Main prediction loop
        while True:
            try:
                if total_steps % progress_steps == 0:
                    progress = (total_steps / len(data)) * 100
                    print(f"\rProgress: {progress:.1f}% (step {total_steps}/{len(data)})", end="")
                    
                    # Log raw features periodically if verbose
                    if verbose:
                        try:
                            current_data = data.iloc[:total_steps+1]
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
                        except Exception as e:
                            log_exception(e, f"calculating verbose features at step {total_steps}", total_steps)
                
                # Recheck balance if configured
                if balance_recheck_bars > 0 and total_steps % balance_recheck_bars == 0:
                    try:
                        env.metrics.balance = env.metrics.get_current_balance()
                    except Exception as e:
                        log_exception(e, f"rechecking balance at step {total_steps}", total_steps)
                
                # Add random spread variation if configured
                if spread_variation > 0:
                    try:
                        current_spread = data['spread'].iloc[total_steps]
                        variation = np.random.uniform(-spread_variation, spread_variation)
                        data.at[data.index[total_steps], 'spread'] = max(0, current_spread + variation)
                    except Exception as e:
                        log_exception(e, f"applying spread variation at step {total_steps}", total_steps)
                
                # Create normalized feature dictionary for tracking
                feature_dict = {}
                try:
                    if obs is not None and isinstance(obs, np.ndarray):
                        feature_names = env.feature_processor.get_feature_names()
                        for i, feat in enumerate(obs):
                            feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                            feature_dict[feature_name] = float(feat)
                            if verbose:
                                print(f"  {feature_name}: {feat:.6f}")
                except Exception as e:
                    log_exception(e, f"creating feature dictionary at step {total_steps}", total_steps)

                # Get prediction from model
                try:
                    action, new_lstm_states = model.model.predict(
                        obs,
                        state=model.lstm_states,
                        deterministic=True
                    )
                    model.lstm_states = new_lstm_states  # Update states in model object
                except Exception as e:
                    log_exception(e, f"model prediction at step {total_steps}", total_steps)
                    # Use default action on prediction failure
                    action = np.array([0])  # HOLD action
                
                try:
                    # Convert action to discrete value and handle position checks
                    if isinstance(action, np.ndarray):
                        action_value = int(action.item())
                    else:
                        action_value = int(action)
                    discrete_action = action_value % 4
                    
                    # Force HOLD if trying to open new position while one exists
                    if env.current_position is not None and discrete_action in [1, 2]:  # Buy or Sell
                        discrete_action = 0  # Force HOLD
                    
                    # Execute step
                    obs, reward, done, truncated, info = env.step(discrete_action)
                    total_reward += reward
                    total_steps += 1
                    
                    # Track trade events if enabled
                    if trade_tracker:
                        try:
                            # Get current timestamp from the data index
                            current_timestamp = data.index[total_steps]
                            
                            if env.current_position and not current_position:  # New position opened
                                trade_tracker.log_trade_entry(
                                    'buy' if discrete_action == 1 else 'sell',
                                    feature_dict,
                                    env.current_position['entry_price'],
                                    env.current_position['lot_size'],
                                    timestamp=current_timestamp
                                )
                            elif not env.current_position and current_position:  # Position closed
                                trade_tracker.log_trade_exit(
                                    'model_close',
                                    info.get('close_price', data['close'].iloc[total_steps]),
                                    info.get('pnl', 0.0),
                                    feature_dict,
                                    timestamp=current_timestamp
                                )
                            elif env.current_position:  # Position update
                                trade_tracker.log_trade_update(
                                    feature_dict,
                                    data['close'].iloc[total_steps],
                                    env.metrics.current_unrealized_pnl,
                                    timestamp=current_timestamp
                                )
                        except Exception as e:
                            log_exception(e, f"trade tracking at step {total_steps}", total_steps)
                    
                    # Update position tracking
                    current_position = env.current_position.copy() if env.current_position else None
                    
                    if done or truncated:
                        logger.info(f"Backtest completed normally: done={done}, truncated={truncated}")
                        break
                        
                except (ValueError, TypeError) as e:
                    log_exception(e, f"processing action at step {total_steps}", total_steps)
                    discrete_action = 0  # Default to HOLD on error
                    try:
                        obs, reward, done, truncated, info = env.step(discrete_action)
                        total_steps += 1
                    except Exception as step_e:
                        log_exception(step_e, f"recovery step execution at step {total_steps}", total_steps)
                        total_steps += 1
                        continue
                    
                except Exception as e:
                    log_exception(e, f"unexpected error during step processing at step {total_steps}", total_steps)
                    print(f"\nUnexpected error at step {total_steps}: {str(e)}")
                    print("Continuing with next step...")
                    total_steps += 1
                    continue
                    
            except Exception as e:
                log_exception(e, f"main loop iteration at step {total_steps}", total_steps)
                print(f"\nError at step {total_steps}: {str(e)}")
                print("Continuing with next step...")
                total_steps += 1
                continue
        
        print(f"\nBacktest completed: {total_steps} steps processed")
        logger.info(f"Backtest completed: {total_steps} steps processed")
        
        # Leave any open positions as-is (no automatic closing)
        if env.current_position:
            logger.info(f"Backtest ended with open position: {env.current_position['direction']} at {env.current_position['entry_price']}")
        
        # Return metrics using same method as evaluate
        try:
            results = model._calculate_backtest_metrics(env, total_steps, total_reward)
            logger.info("Backtest metrics calculated successfully")
            return results
        except Exception as e:
            log_exception(e, "calculating backtest metrics")
            raise RuntimeError("Failed to calculate backtest metrics")
            
    except Exception as e:
        log_exception(e, "backtest_with_predictions function")
        raise

def main():
    parser = argparse.ArgumentParser(description='Backtest trained trading model')
    
    # Default values converted to ZAR using rate of 19
    default_initial_balance = 10000.0 * 19  # 10,000 USD in ZAR
    default_balance_per_lot = 1000.0 * 19   # 1,000 USD per lot in ZAR
    
    # Trading environment settings
    
    parser.add_argument('--model_path', type=str, required=True,
                     help='Path to the model file to backtest')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the test data CSV file')
    parser.add_argument('--initial_balance', type=float, default=default_initial_balance,
                      help='Initial account balance in account currency (default: 190,000 ZAR)')
    parser.add_argument('--balance_per_lot', type=float, default=default_balance_per_lot,
                      help='Account balance required per 0.01 lot in account currency (default: 19,000 ZAR)')
    parser.add_argument('--currency_conversion', type=float, default=1.0,                      help='Conversion rate from USD to account currency (default: 1.0)')
                      
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
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting main backtest function")
        
        # Load and validate data
        print("\nLoading market data...")
        logger.info(f"Loading data from: {args.data_path}")
        
        try:
            df = pd.read_csv(args.data_path)
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
            logger.info(f"Successfully loaded {len(df)} rows of data")
        except Exception as e:
            log_exception(e, "loading and parsing CSV data")
            raise RuntimeError(f"Failed to load data from {args.data_path}")
        
        # Apply date filters if provided
        try:
            if args.start_date:
                start_ts = pd.Timestamp(args.start_date).tz_localize('UTC')
                df = df[df.index >= start_ts]
                logger.info(f"Applied start date filter: {args.start_date}")
            if args.end_date:
                end_ts = pd.Timestamp(args.end_date).tz_localize('UTC')
                df = df[df.index <= end_ts]
                logger.info(f"Applied end date filter: {args.end_date}")
        except Exception as e:
            log_exception(e, "applying date filters")
            logger.warning("Continuing without date filters")
        
        if len(df) == 0:
            error_msg = "No data available for specified date range"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        print(f"Loaded {len(df):,d} bars from {df.index[0]} to {df.index[-1]}")
        logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Validate data
        try:
            missing_columns = [col for col in ['open', 'close', 'high', 'low', 'spread'] 
                              if col not in df.columns]
            if missing_columns:
                warning_msg = f"Dataset missing columns: {missing_columns}"
                print(f"WARNING: {warning_msg}")
                logger.warning(warning_msg)
                print("These columns are required for backtesting. Will attempt to continue.")
        except Exception as e:
            log_exception(e, "validating data columns")
        
        # Initialize model
        print(f"\nInitializing model from: {args.model_path}")
        logger.info(f"Initializing model from: {args.model_path}")
        
        try:
            model = TradeModel(
                model_path=args.model_path,
                balance_per_lot=args.balance_per_lot,  # Pass the parameter consistently
                point_value=args.point_value,
                min_lots=args.min_lots,
                max_lots=args.max_lots,
                contract_size=args.contract_size
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            log_exception(e, "initializing TradeModel")
            raise RuntimeError(f"Failed to initialize model from {args.model_path}")
        
        # Run backtest based on selected method
        try:
            if args.method == 'predict_single':
                # Use the method that simulates live trading
                print("\nRunning prediction backtest (simulates live trading)...")
                logger.info("Starting prediction backtest method")
                results = backtest_with_predictions(
                    model=model,
                    data=df,
                    initial_balance=args.initial_balance,
                    balance_per_lot=args.balance_per_lot,
                    verbose=args.verbose_features,
                    point_value=args.point_value,
                    min_lots=args.min_lots,
                    max_lots=args.max_lots,
                    contract_size=args.contract_size,
                    currency_conversion=args.currency_conversion,
                    reset_states_on_gap=args.reset_states_on_gap,
                    spread_variation=args.spread_variation,
                    slippage_range=args.slippage_range,
                    balance_recheck_bars=args.balance_recheck_bars,
                    trades_log_path=args.trades_log_path
                )
            else:  # method == 'evaluate' or args.quiet
                print("\nRunning quiet backtest with evaluate method...")
                logger.info("Starting evaluate backtest method")
                progress_thread = None
                
                # Start continuous progress indicator
                try:
                    if len(df) > 1000:  # For any substantial dataset
                        progress_thread = threading.Thread(
                            target=show_progress_continuous,
                            args=("Running backtest",)
                        )
                        progress_thread.daemon = True
                        progress_thread.start()
                        logger.info("Progress indicator started")
                except Exception as e:
                    log_exception(e, "starting progress indicator")
                
                try:    
                    results = model.evaluate(
                        data=df,
                        initial_balance=args.initial_balance,
                        balance_per_lot=args.balance_per_lot,
                        spread_variation=args.spread_variation,
                        slippage_range=args.slippage_range
                    )
                    logger.info("Model evaluation completed successfully")
                except Exception as e:
                    log_exception(e, "running model evaluation")
                    raise
                finally:
                    # Always stop the progress indicator
                    try:
                        if progress_thread and progress_thread.is_alive():
                            stop_progress_indicator()
                            logger.info("Progress indicator stopped")
                    except Exception as e:
                        log_exception(e, "stopping progress indicator")
        except Exception as e:
            log_exception(e, "running backtest")
            raise
        
        # Print metrics
        try:
            print("\nBacktest Results:")
            print_metrics(results)
            logger.info("Backtest results printed successfully")
        except Exception as e:
            log_exception(e, "printing backtest results")
            print(f"Error printing results: {str(e)}")
        
        # Save results to JSON if requested
        if args.output_json:
            try:
                import json
                with open(args.output_json, 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=4)
                print(f"Results saved to {args.output_json}")
                logger.info(f"Results saved to JSON: {args.output_json}")
            except Exception as e:
                log_exception(e, f"saving results to JSON file {args.output_json}")
                print(f"Error saving results to JSON: {e}")

        # Plot results if requested
        if args.plot or args.output_plot:
            try:
                plot_results(results, save_path=args.output_plot)
                if args.output_plot:
                    print(f"Plot saved to {args.output_plot}")
                    logger.info(f"Plot saved to: {args.output_plot}")
            except Exception as e:
                log_exception(e, "generating plots")
                print(f"Error generating plots: {str(e)}")
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        log_exception(e, "main backtest execution")
        print(f"\nError during backtesting: {str(e)}")
        logger.error("Backtest failed with critical error")
        return

if __name__ == "__main__":
    main()
