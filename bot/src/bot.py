#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Trading Bot using DQN model.
"""

import time
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

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
    """Trading bot that uses a DQN model to make trading decisions."""
    
    def __init__(self):
        """Initialize the trading bot components."""
        self.setup_logging()
        self.running = True
        self.mt5 = None
        self.data_fetcher = None
        self.model = None
        self.trade_executor = None
        self.last_bar_index = None
        
    def setup_logging(self) -> None:
        """Configure logging with both console and file output."""
        log_file = datetime.now().strftime("DRL_DQN_Bot_%Y-%m-%d.log")
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
            self.model = TradeModel(MODEL_PATH)
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
            
            # Get the data for prediction
            data = self.data_fetcher.fetch_data()
            if data is None:
                self.logger.warning("Failed to fetch market data")
                return
                
            # Make prediction and execute trade
            prediction = self.model.predict_single(data)
            self.logger.debug(f"Model prediction: {prediction}")
            self.trade_executor.execute_trade(prediction)
            
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