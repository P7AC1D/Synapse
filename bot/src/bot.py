#!/usr/bin/env python3

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Trading Bot using PPO-LSTM model.
"""

import time
import logging
import signal
import sys
import numpy as np
import argparse
from datetime import datetime
from typing import Optional
import pandas as pd

# Import specific config values instead of using wildcard imports
from mt5_connector import MT5Connector
from data_fetcher import DataFetcher
from trade_model import TradeModel
from trade_executor import TradeExecutor
from trading.environment import TradingEnv
from trading.trade_tracker import TradeTracker
from config import (
    LOG_FILE_PATH,
    TRADE_TRACKING_PATH,
    MT5_BASE_SYMBOL,
    MT5_SYMBOL,
    MT5_PATH,
    MT5_TIMEFRAME_MINUTES,
    BARS_TO_FETCH,
    MODEL_PATH,
    BALANCE_PER_LOT,
    MT5_COMMENT,
    MAX_SPREAD,
    STOP_LOSS_PIPS,
    TAKE_PROFIT_PIPS
)


class TradingBot:
    """Trading bot that uses a PPO-LSTM model to make trading decisions."""
    
    def __init__(self, model_path=MODEL_PATH, symbol=MT5_SYMBOL, max_spread=MAX_SPREAD, 
                 balance_per_lot=BALANCE_PER_LOT, stop_loss_pips=STOP_LOSS_PIPS,
                 take_profit_pips=TAKE_PROFIT_PIPS):
        """Initialize the trading bot components."""
        self.model_path = model_path
        self.symbol = symbol
        self.max_spread = max_spread
        self.balance_per_lot = balance_per_lot
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        
        self.setup_logging()
        self.running = True
        
        # Create symbol-specific trades directory
        trades_log_path = os.path.join(TRADE_TRACKING_PATH, self.symbol)
        os.makedirs(trades_log_path, exist_ok=True)
        self.trade_tracker = TradeTracker(trades_log_path)
        self.mt5 = None
        self.data_fetcher = None
        self.model = None
        self.trade_executor = None
        self.last_bar_index = None
        self.current_position = None  # Track current position info
        self.full_historical_data = None  # Store complete historical data to match backtest approach
    
    def setup_logging(self) -> None:
        """Configure logging with both console and file output."""
        log_file = datetime.now().strftime(f"{MT5_COMMENT}_{self.symbol}_%Y-%m-%d.log")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{LOG_FILE_PATH}/{log_file}", mode='a', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def verify_positions(self) -> None:
        """Verify and synchronize internal position tracking with MT5 positions."""
        try:
            positions = self.mt5.get_open_positions(self.symbol, MT5_COMMENT)
            
            # Case 1: We think we have a position but MT5 doesn't
            if self.current_position and not positions:
                self.logger.warning(
                    f"Position tracking mismatch: Internal position exists but no MT5 position found. "
                    f"Clearing internal position tracking."
                )
                self.current_position = None
                
            # Case 2: MT5 has a position but we don't think we do
            elif not self.current_position and positions:
                position = positions[0]  # Get first position if multiple exist
                self.logger.warning(
                    f"Position tracking mismatch: MT5 position found but no internal tracking. "
                    f"Updating internal tracking."
                )
                self.current_position = {
                    "direction": 1 if position.type == 0 else -1,  # 0=buy, 1=sell in MT5
                    "entry_price": position.price_open,
                    "lot_size": position.volume,
                    "entry_step": 0,  # Will be updated in trading cycle
                    "entry_time": str(position.time),
                    "profit": position.profit  # Get profit directly from MT5
                }
                
            # Case 3: Both have positions - verify details match
            elif self.current_position and positions:
                position = positions[0]
                mt5_direction = 1 if position.type == 0 else -1
                
                # Log current values before update
                self.logger.debug(
                    f"Syncing position details: "
                    f"Direction: {self.current_position['direction']} -> {mt5_direction} | "
                    f"Lot Size: {self.current_position['lot_size']:.2f} -> {position.volume:.2f} | "
                    f"Profit: {self.current_position.get('profit', 0.0):.2f} -> {position.profit:.2f}"
                )
                
                self.current_position.update({
                    "direction": mt5_direction,
                    "lot_size": position.volume,
                    "profit": position.profit  # Also update profit from MT5
                })
            
        except Exception as e:
            self.logger.error(f"Error verifying positions: {e}")

    def initialize(self) -> bool:
        """Initialize connections and components."""
        try:
            self.logger.info("Initializing trading bot...")
            
            # Connect to MT5
            self.mt5 = MT5Connector()
            # Give MT5Connector access to trade tracker
            self.mt5.trade_tracker = self.trade_tracker
            if not self.mt5.connect():
                self.logger.error("Failed to connect to MT5")
                return False
            
            # Verify MT5 connection before proceeding
            if not self.mt5.is_connected():
                self.logger.error("MT5 connection verification failed")
                return False

            # Initialize data fetcher with connection verification
            try:
                self.data_fetcher = DataFetcher(
                    self.mt5, self.symbol, MT5_TIMEFRAME_MINUTES, BARS_TO_FETCH + 1
                )
                # Verify data fetcher by attempting to fetch one bar
                test_data = self.data_fetcher.fetch_current_bar(include_history=False)
                if test_data is None:
                    self.logger.error("Failed to verify DataFetcher functionality")
                    return False
            except Exception as e:
                self.logger.error(f"Failed to initialize DataFetcher: {e}")
                return False
            
            # Get symbol parameters from connector
            try:
                contract_size, min_lots, max_lots, volume_step, point_value, digits = self.mt5.get_symbol_info(self.symbol)
            except Exception as e:
                self.logger.error(f"Failed to get symbol info for {self.symbol}: {e}")
                return False
            
            self.logger.info(f"Symbol info - Point: {point_value}, " 
                           f"Contract Size: {contract_size}, Min Lot: {min_lots}, Max Lot: {max_lots}, "
                           f"Volume Step: {volume_step}")
                
            # Initialize components
            self.data_fetcher = DataFetcher(
                self.mt5, self.symbol, MT5_TIMEFRAME_MINUTES, BARS_TO_FETCH + 1
            )
            # Initialize trading model with actual balance
            self.logger.info(f"Loading trading model from: {self.model_path}")
            account_balance = self.mt5.get_account_balance()
            self.model = TradeModel(
                self.model_path,
                balance_per_lot=self.balance_per_lot,
                initial_balance=account_balance,
                point_value=point_value,
                min_lots=min_lots,
                max_lots=max_lots,
                contract_size=contract_size
            )
            if not self.model.model:  # Check if model loaded successfully
                self.logger.error("Failed to load trading model")
                return False

            # Initialize trade executor with balance_per_lot, stop_loss_pips and take_profit_pips
            self.trade_executor = TradeExecutor(
                self.mt5, 
                symbol=self.symbol,
                balance_per_lot=self.balance_per_lot,
                stop_loss_pips=self.stop_loss_pips,
                take_profit_pips=self.take_profit_pips
            )
            
            # Get initial data for warmup
            initial_data = self.data_fetcher.fetch_data()
            if initial_data is None or len(initial_data) < 100:  # Just need enough data to start
                self.logger.error(f"Failed to fetch enough initial data (need at least 100 bars)")
                return False
                
            # Initialize full historical data with all available data (matching backtest approach)
            self.full_historical_data = initial_data.copy()
            self.logger.info(f"Initialized full historical data with {len(self.full_historical_data)} bars")

            # Perform LSTM warmup if model supports it
            if self.model.is_lstm_model():
                warmup_window = min(100, len(initial_data) // 10)  # Use 100 bars or 10% of data
                self.model.warmup_lstm_state(initial_data.iloc[:-warmup_window], warmup_window)
                self.logger.info(f"Warmed up LSTM states using {warmup_window} bars")

            # Get initial bar data
            current_bar = self.data_fetcher.fetch_current_bar()
            if current_bar is None or len(current_bar.index) == 0:
                self.logger.error("Failed to fetch initial bar data")
                return False

            # Check for existing positions
            positions = self.mt5.get_open_positions(self.symbol, MT5_COMMENT)
            if positions:
                position = positions[0]  # Get first position if multiple exist
                self.current_position = {
                    "direction": 1 if position.type == 0 else -1,  # 0=buy, 1=sell in MT5
                    "entry_price": position.price_open,
                    "lot_size": position.volume,
                    "entry_step": len(self.full_historical_data) - 1,  # Entry at the most recent bar
                    "entry_time": str(position.time),
                    "profit": position.profit  # Get profit directly from MT5
                }
                
                # Log the position info
                self.logger.info(
                    f"Recovered existing position: "
                    f"{'LONG' if self.current_position['direction'] == 1 else 'SHORT'} "
                    f"{self.current_position['lot_size']:.2f} lots @ {self.current_position['entry_price']:.5f}"
                )
                
                # Try to convert entry_time to ensure it's in proper format
                try:
                    entry_time_str = self.current_position["entry_time"]
                    self.logger.debug(f"Position entry time (original): {entry_time_str}")
                    
                    # Handle numeric timestamp (seconds since epoch)
                    if isinstance(entry_time_str, (int, float)) or (isinstance(entry_time_str, str) and entry_time_str.isdigit()):
                        try:
                            # Convert to integer if it's a string containing digits
                            timestamp = int(float(entry_time_str))
                            # Check if this is a large timestamp in seconds or milliseconds
                            if timestamp > 1000000000000:  # If in milliseconds
                                timestamp = timestamp / 1000
                            entry_time = pd.to_datetime(timestamp, unit='s')
                            self.logger.debug(f"Converted entry timestamp to: {entry_time}")
                            self.current_position["entry_time"] = str(entry_time)
                        except (ValueError, OverflowError) as e:
                            self.logger.warning(f"Failed to convert entry timestamp: {e}, keeping original value")
                except Exception as e:
                    self.logger.warning(f"Error processing position entry time: {e}")

            self.last_bar_index = current_bar.index[-1]
            self.logger.info("Trading bot initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error during initialization: {e}")
            return False
            
    def process_trading_cycle(self) -> None:
        """Execute a single trading cycle."""
        try:
            # Get current bar first to check for updates
            current_bar = self.data_fetcher.fetch_current_bar(include_history=False)  # Just get latest bar
            
            # Skip if no new data or error
            if current_bar is None:
                return
            
            # Check if we have a new bar
            current_time = current_bar.index[-1]
            if self.last_bar_index is not None and current_time <= self.last_bar_index:
                time.sleep(1)  # Avoid excessive CPU usage
                return
                
            # Check for market gaps and reset LSTM states if needed
            if self.last_bar_index is not None:
                time_diff = (current_time - self.last_bar_index).total_seconds() / 60
                expected_diff = MT5_TIMEFRAME_MINUTES  # Expected time between bars
                
                if time_diff > expected_diff * 2:  # Gap is more than double the expected timeframe
                    self.logger.warning(f"Market gap detected: {time_diff/60:.2f}h - Resetting LSTM states")
                    if self.model.is_lstm_model():
                        self.model.reset_lstm_states()
            
            self.logger.info(f"New bar detected at {current_time}")
            self.last_bar_index = current_time
            
            # Verify position tracking is synchronized with MT5
            self.verify_positions()
            
            # Get fresh market data
            new_data = self.data_fetcher.fetch_data()
            if new_data is None:
                self.logger.warning("Failed to fetch market data")
                return

            # Update the full historical data to include the new bar
            if self.full_historical_data is not None:
                # Get only new data that's not already in our historical dataset
                last_historical_time = self.full_historical_data.index[-1]
                new_bars = new_data[new_data.index > last_historical_time]
                
                if len(new_bars) > 0:
                    self.logger.info(f"Adding {len(new_bars)} new bars to historical dataset")
                    # Append new data to our historical dataset
                    self.full_historical_data = pd.concat([self.full_historical_data, new_bars])
                    self.logger.info(f"Historical dataset now contains {len(self.full_historical_data)} bars")
            else:
                # If somehow historical data is missing, initialize it
                self.full_historical_data = new_data.copy()
                self.logger.info(f"Reinitialized historical dataset with {len(self.full_historical_data)} bars")

            # Try to get USD/ZAR rate, fallback to 1.0 if not available
            try:
                usd_zar_bid, usd_zar_ask = self.mt5.get_symbol_info_tick(MT5_BASE_SYMBOL)
                usd_zar_rate = usd_zar_bid  # Use bid price for conversion
                self.logger.debug(f"Using USD/ZAR rate: {usd_zar_rate}")
            except Exception as e:
                self.logger.warning(f"Failed to get USD/ZAR rate, using 19.0: {str(e)}")
                usd_zar_rate = 19.0

            # Update model's balance before prediction
            current_balance = self.mt5.get_account_balance()
            self.logger.debug(f"Updating model balance: {self.model.initial_balance} -> {current_balance}")
            self.model.initial_balance = current_balance
            
            # Work with the complete dataset for prediction
            data_for_prediction = self.full_historical_data.copy()
            
            # Log historical data size being used for prediction
            self.logger.info(f"Using {len(data_for_prediction)} historical bars for prediction")

            # Create a single trading environment that will be used for both prediction and feature extraction
            env = TradingEnv(
                data=data_for_prediction,
                initial_balance=self.model.initial_balance,
                balance_per_lot=self.model.balance_per_lot,
                random_start=False,
                predict_mode=True,  # Set to True to always use most recent data
                point_value=self.model.point_value,
                min_lots=self.model.min_lots,
                max_lots=self.model.max_lots,
                contract_size=self.model.contract_size,
                currency_conversion=usd_zar_rate
            )
            
            # Reset environment to initialize
            obs, _ = env.reset()

            # Copy current position to environment
            if self.current_position:
                env.current_position = self.current_position.copy()
            
            # Get observation with updated position metrics - predict_mode=True automatically uses the last data point
            obs = env.get_observation()
            
            # Use predict_single to get prediction including LSTM state management
            prediction = self.model.predict_single(data_for_prediction, env)
            if prediction is None:
                self.logger.error("Failed to get prediction from model")
                return

            # Log prediction details
            self.logger.info(
                f"Model prediction: Action={prediction['action']} | "
                f"Description={prediction['description']}"
            )
            
            # Extract current price for trade logging
            current_price = data_for_prediction['close'].iloc[-1]
            
            # Create normalized feature dictionary for trade tracking using our environment's observation
            features_for_tracking = {}
            try:
                # Create feature dictionary if observation is available
                if obs is not None and isinstance(obs, np.ndarray):
                    feature_names = env.feature_processor.get_feature_names()
                    
                    self.logger.debug("\nNormalized features for prediction:")
                    for i, feat in enumerate(obs):
                        feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                        features_for_tracking[feature_name] = float(feat)  # Convert numpy values to Python float
                        self.logger.debug(f"  {feature_name}: {feat:.6f}")
                        
                # Also log raw feature values for debugging
                feature_processor = env.feature_processor
                atr, rsi, (upper_band, lower_band), trend_strength = feature_processor._calculate_indicators(
                    data_for_prediction['high'].values,
                    data_for_prediction['low'].values,
                    data_for_prediction['close'].values
                )
                
                # Log raw feature values for the last bar
                self.logger.debug("\nRaw feature values (last bar):")
                if len(atr) > 0:
                    self.logger.debug(f"ATR: {atr[-1]:.6f}")
                if len(rsi) > 0:
                    self.logger.debug(f"RSI: {rsi[-1]:.6f}")
                if len(upper_band) > 0:
                    self.logger.debug(f"BB Upper: {upper_band[-1]:.6f}")
                if len(lower_band) > 0:
                    self.logger.debug(f"BB Lower: {lower_band[-1]:.6f}")
                if len(trend_strength) > 0:
                    self.logger.debug(f"Trend Strength: {trend_strength[-1]:.6f}")
                    
            except Exception as e:
                self.logger.warning(f"Error extracting features for tracking: {e}")
            
            # Convert prediction to trade action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            action_desc = action_map[prediction['action']]
            
            self.logger.info(
                f"Trade Signal - Action: {action_desc} | "
                f"Description: {prediction['description']}"
            )
            
            # Execute trade and update position tracking
            success = self.trade_executor.execute_trade(prediction)
            
            # Update position tracking and log trade events
            if success:
                if prediction['action'] == 3:  # Close
                    if self.current_position:
                        self.trade_tracker.log_trade_exit(
                            'model_close',
                            current_price,
                            self.current_position.get('profit', 0.0),
                            features_for_tracking
                        )
                    self.current_position = None
                elif prediction['action'] in [1, 2] and not self.current_position:  # New position
                    self.current_position = {
                        "direction": 1 if prediction['action'] == 1 else -1,
                        "entry_price": current_price,
                        "lot_size": self.trade_executor.last_lot_size,
                        "entry_step": len(data_for_prediction) - 1,
                        "entry_time": str(data_for_prediction.index[-1])
                    }
                    # Log trade entry with features
                    self.trade_tracker.log_trade_entry(
                        'buy' if prediction['action'] == 1 else 'sell',
                        features_for_tracking,
                        current_price,
                        self.trade_executor.last_lot_size
                    )
                
                # Log trade update if position exists
                if self.current_position:
                    self.trade_tracker.log_trade_update(
                        features_for_tracking,
                        current_price,
                        self.current_position.get('profit', 0.0)
                    )
            
        except Exception as e:
            self.logger.exception(f"Error in trading cycle: {e}")
    
    def setup_signal_handlers(self) -> None:
        """Set up handlers for termination signals."""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received shutdown signal {signum}, shutting down...")
        self.running = False
        
    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Cleaning up resources...")
        
        if self.mt5:
            self.mt5.disconnect()
        
        self.logger.info("Cleanup complete")
        
    def run(self) -> None:
        """Start the bot's main loop."""
        self.logger.info("Trading Bot starting")
        
        # Initialize components first
        if not self.initialize():
            self.logger.error("Failed to initialize bot components. Shutting down.")
            return
        
        # Main loop - run until stopped
        try:
            self.running = True
            
            # Initialize the update timer for tracking time between updates
            last_update_time = time.time()
            
            while self.running:
                try:
                    # Execute trading cycle
                    self.process_trading_cycle()
                    
                    # Sleep to avoid excessive CPU usage
                    time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    time.sleep(1)  # Add delay before retry to avoid tight loops on persistent errors
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, shutting down")
        finally:
            self.cleanup()


def main() -> int:
    """Main entry point for the trading bot."""
    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning Trading Bot using PPO-LSTM model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the trained model")
    parser.add_argument("--symbol", type=str, default=MT5_SYMBOL, help="Trading symbol")
    parser.add_argument("--max_spread", type=float, default=MAX_SPREAD, help="Maximum allowed spread")
    parser.add_argument("--balance_per_lot", type=float, default=BALANCE_PER_LOT, help="Balance per lot")
    parser.add_argument("--stop_loss_pips", type=int, default=STOP_LOSS_PIPS, help="Stop loss in pips")
    parser.add_argument("--take_profit_pips", type=int, default=TAKE_PROFIT_PIPS, help="Take profit in pips")
    args = parser.parse_args()

    bot = TradingBot(
        model_path=args.model_path,
        symbol=args.symbol,
        max_spread=args.max_spread,
        balance_per_lot=args.balance_per_lot,
        stop_loss_pips=args.stop_loss_pips,
        take_profit_pips=args.take_profit_pips
    )
    try:
        bot.run()
        return 0
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
