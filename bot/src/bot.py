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
from config import (
    LOG_FILE_PATH,
    MT5_SYMBOL,
    MT5_TIMEFRAME_MINUTES,
    BARS_TO_FETCH,
    MODEL_PATH
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
        self.current_grid_id = 0  # Track grid IDs
        self.active_grids = {
            'long': {},  # Dict to track long grid trades
            'short': {}  # Dict to track short grid trades
        }
        
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
            # Initialize trading model
            self.logger.info(f"Loading trading model from: {MODEL_PATH}")
            self.model = TradeModel(MODEL_PATH)
            if not self.model.model:  # Check if model loaded successfully
                self.logger.error("Failed to load trading model")
                return False

            self.trade_executor = TradeExecutor(self.mt5)
            
            # Get initial bar data
            current_bar = self.data_fetcher.fetch_current_bar()
            if current_bar is None or len(current_bar.index) == 0:
                self.logger.error("Failed to fetch initial bar data")
                return False
                
            self.last_bar_index = current_bar.index[-1]
            self.logger.info("Trading bot initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error during initialization: {e}")
            return False
            
    def process_trading_cycle(self) -> None:
        """Execute a single trading cycle."""
        try:
            current_bar = self.data_fetcher.fetch_current_bar()
            
            # Check if we have a new bar
            if current_bar is None or self.last_bar_index == current_bar.index[-1]:
                return
                
            self.logger.info(f"New bar detected at {current_bar.index[-1]}")
            self.last_bar_index = current_bar.index[-1]
            
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

            # Make prediction and execute trade
            # Make prediction
            prediction = self.model.predict_single(data)
            self.lstm_states = self.model.lstm_states  # Update LSTM states
            
            # Update prediction with grid ID if new trade
            if prediction['position'] != 0:
                direction = 'long' if prediction['position'] == 1 else 'short'
                if not self.active_grids[direction]:  # No active grid in this direction
                    self.current_grid_id += 1
                    self.active_grids[direction][self.current_grid_id] = {
                        'positions': [],
                        'grid_size': prediction['grid_size_pips'],
                        'created_at': current_bar.index[-1]
                    }
                prediction['grid_id'] = self.current_grid_id
            
            self.logger.debug(
                f"Grid Trade Signal - Direction: {'BUY' if prediction['position'] == 1 else 'SELL' if prediction['position'] == -1 else 'HOLD'} | "
                f"Grid Size: {prediction.get('grid_size_pips', 0):.1f} pips | "
                f"Grid ID: {prediction.get('grid_id', 'None')}"
            )
            
            # Execute trade
            success = self.trade_executor.execute_trade(prediction)
            
            # Update grid tracking on successful trade
            if success and prediction['position'] != 0:
                direction = 'long' if prediction['position'] == 1 else 'short'
                grid_id = prediction['grid_id']
                if grid_id in self.active_grids[direction]:
                    self.active_grids[direction][grid_id]['positions'].append({
                        'entry_time': current_bar.index[-1],
                        'entry_price': current_bar['close'].iloc[-1],
                        'grid_size': prediction['grid_size_pips']
                    })
            
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
        
        # Log final grid statistics
        for direction in ['long', 'short']:
            for grid_id, grid in self.active_grids[direction].items():
                self.logger.info(
                    f"Grid {grid_id} ({direction}) - "
                    f"Positions: {len(grid['positions'])} | "
                    f"Grid Size: {grid['grid_size']:.1f} pips | "
                    f"Active Since: {grid['created_at']}"
                )
        
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
