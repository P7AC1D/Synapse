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
from datetime import datetime
from typing import Optional
import pandas as pd

# Import specific config values instead of using wildcard imports
from mt5_connector import MT5Connector
from data_fetcher import DataFetcher
from trade_model import TradeModel
from trade_executor import TradeExecutor
from trading.environment import TradingEnv
from config import (
    LOG_FILE_PATH,
    MT5_BASE_SYMBOL,
    MT5_SYMBOL,
    MT5_TIMEFRAME_MINUTES,
    BARS_TO_FETCH,
    MODEL_PATH,
    BALANCE_PER_LOT,
    MT5_COMMENT
)


class TradingBot:
    """Trading bot that uses a PPO-LSTM model to make trading decisions."""
    
    def __init__(self):
        """Initialize the trading bot components."""
        self.setup_logging()
        self.running = True
        self.mt5 = None
        self.data_fetcher = None
        self.model = None
        self.trade_executor = None
        self.last_bar_index = None
        self.lstm_states = None  # Store LSTM states between predictions
        self.current_position = None  # Track current position info
        self.data_window = None  # Store rolling data window
        
    def setup_logging(self) -> None:
        """Configure logging with both console and file output."""
        log_file = datetime.now().strftime("DRL_PPO_LSTM_Bot_%Y-%m-%d.log")
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
            positions = self.mt5.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
            
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
                    "entry_time": str(position.time)
                }
                
            # Case 3: Both have positions - verify details match
            elif self.current_position and positions:
                position = positions[0]
                mt5_direction = 1 if position.type == 0 else -1
                
                if (mt5_direction != self.current_position['direction'] or
                    abs(position.volume - self.current_position['lot_size']) > 1e-8):
                    self.logger.warning(
                        f"Position details mismatch - MT5: {position.type} {position.volume:.2f} lots, "
                        f"Internal: {self.current_position['direction']} {self.current_position['lot_size']:.2f} lots. "
                        f"Updating internal tracking."
                    )
                    self.current_position.update({
                        "direction": mt5_direction,
                        "lot_size": position.volume
                    })
            
        except Exception as e:
            self.logger.error(f"Error verifying positions: {e}")

    def initialize(self) -> bool:
        """Initialize connections and components."""
        try:
            self.logger.info("Initializing trading bot...")
            
            # Connect to MT5
            self.mt5 = MT5Connector()
            if not self.mt5.connect():
                self.logger.error("Failed to connect to MT5")
                return False
                
            # Initialize components
            self.data_fetcher = DataFetcher(
                self.mt5, MT5_SYMBOL, MT5_TIMEFRAME_MINUTES, BARS_TO_FETCH + 1
            )
            # Initialize trading model with actual balance
            self.logger.info(f"Loading trading model from: {MODEL_PATH}")
            account_balance = self.mt5.get_account_balance()
            self.model = TradeModel(
                MODEL_PATH,
                balance_per_lot=BALANCE_PER_LOT,
                initial_balance=account_balance
            )
            if not self.model.model:  # Check if model loaded successfully
                self.logger.error("Failed to load trading model")
                return False

            # Initialize trade executor with balance_per_lot from config
            self.trade_executor = TradeExecutor(
                self.mt5, 
                balance_per_lot=BALANCE_PER_LOT
            )
            
            # Get initial data for warmup
            initial_data = self.data_fetcher.fetch_data()
            if initial_data is None or len(initial_data) < self.model.initial_warmup:
                self.logger.error(f"Failed to fetch enough initial data (need {self.model.initial_warmup} bars)")
                return False
                
            # Initialize data window with warmup data
            self.data_window = initial_data.copy()
            
            # Preload LSTM states
            try:
                self.model.preload_states(initial_data)
                self.logger.info("Successfully preloaded LSTM states")
            except Exception as e:
                self.logger.error(f"Failed to preload LSTM states: {e}")
                return False

            # Get initial bar data
            current_bar = self.data_fetcher.fetch_current_bar()
            if current_bar is None or len(current_bar.index) == 0:
                self.logger.error("Failed to fetch initial bar data")
                return False

            # Check for existing positions
            positions = self.mt5.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
            if positions:
                position = positions[0]  # Get first position if multiple exist
                self.current_position = {
                    "direction": 1 if position.type == 0 else -1,  # 0=buy, 1=sell in MT5
                    "entry_price": position.price_open,
                    "lot_size": position.volume,
                    "entry_step": 0,  # Will be updated in first trading cycle
                    "entry_time": str(position.time)
                }
                self.logger.info(
                    f"Recovered existing position: "
                    f"{'LONG' if self.current_position['direction'] == 1 else 'SHORT'} "
                    f"{self.current_position['lot_size']:.2f} lots @ {self.current_position['entry_price']:.5f}"
                )
                
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
                
            self.logger.info(f"New bar detected at {current_time}")
            self.last_bar_index = current_time
            
            # Verify position tracking is synchronized with MT5
            self.verify_positions()
            
            # Get and preprocess the data for prediction
            data = self.data_fetcher.fetch_data()
            if data is None:
                self.logger.warning("Failed to fetch market data")
                return

            # Reset LSTM states only on significant data gaps
            if self.last_bar_index is not None:
                expected_time = self.last_bar_index + pd.Timedelta(minutes=MT5_TIMEFRAME_MINUTES)
                time_diff = abs((current_bar.index[-1] - expected_time).total_seconds())
                # Only reset if gap is more than 2x the timeframe
                if time_diff > (MT5_TIMEFRAME_MINUTES * 2 * 60):
                    self.logger.info(f"Significant data gap detected ({time_diff/60:.1f} minutes), resetting LSTM states")
                    self.lstm_states = None

            # Get USD/ZAR rate for currency conversion
            usd_zar_rate = self.mt5.get_symbol_info_tick(MT5_BASE_SYMBOL)[0]  # Use bid price
            if usd_zar_rate is None:
                self.logger.warning("Failed to get USD/ZAR rate, using 1.0")
                usd_zar_rate = 1.0

            # Update position info for prediction
            if self.current_position:
                # Update entry step relative to current data window
                self.current_position["entry_step"] = len(data) - 1

            # Update model's balance before prediction
            current_balance = self.mt5.get_account_balance()
            self.model.initial_balance = current_balance

            # Get current price for live P&L calculation
            current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)
            if current_price is None:
                self.logger.warning("Failed to get current price")
                return
                
            # Use bid price for P&L calculation (matches MQL5)
            live_price = current_price[0]
            
            data = data.iloc[:-1]  # Exclude the last row for prediction as its not completed yet
            
            # Make prediction with position context, live price, and currency conversion
            prediction = self.model.predict_single(
                data,
                current_position=self.current_position,
                verbose=True,
                live_price=live_price,
                currency_conversion=usd_zar_rate
            )
            self.lstm_states = self.model.lstm_states  # Update LSTM states
            
            # Convert prediction to trade action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'CLOSE'}
            action_desc = action_map[prediction['action']]
            
            self.logger.debug(
                f"Trade Signal - Action: {action_desc} | "
                f"Description: {prediction['description']} | "
                f"Current Position: {'None' if not self.current_position else ('LONG' if self.current_position['direction'] == 1 else 'SHORT')}"
            )
            
            # Execute trade and update position tracking
            success = self.trade_executor.execute_trade(prediction)
            
            # Update position tracking based on action
            if success:
                if prediction['action'] == 3:  # Close
                    self.current_position = None
                elif prediction['action'] in [1, 2] and not self.current_position:  # New position
                    self.current_position = {
                        "direction": 1 if prediction['action'] == 1 else -1,
                        "entry_price": data['close'].iloc[-1],
                        "lot_size": self.trade_executor.last_lot_size,
                        "entry_step": len(data) - 1,
                        "entry_time": str(data.index[-1])
                    }
            
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
        
        # Reset model states
        if self.model:
            self.model.reset_states()
            self.lstm_states = None
        
        self.logger.info("Cleanup complete")
        
    def run(self) -> None:
        """Run the trading bot main loop."""
        if not self.initialize():
            self.logger.error("Initialization failed")
            return
            
        self.setup_signal_handlers()
        self.logger.info("Starting trading bot main loop...")
        
        try:
            while self.running:
                self.process_trading_cycle()
                time.sleep(1)  # Sleep to avoid excessive CPU usage
                
        except Exception as e:
            self.logger.exception(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()


def main() -> int:
    """Main entry point for the trading bot."""
    bot = TradingBot()
    try:
        bot.run()
        return 0
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
