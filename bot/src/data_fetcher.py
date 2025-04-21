"""Market data fetching and processing module."""

import logging
from typing import Optional, Any

import pandas as pd
import MetaTrader5 as mt5

from mt5_connector import MT5Connector


class DataFetcher:
    """Fetches and processes market data from MT5."""
    
    def __init__(self, mt5_connector: MT5Connector, symbol: str, timeframe: int, num_bars: int):
        """
        Initialize the data fetcher.
        
        Args:
            mt5_connector: MT5 connector instance
            symbol: Trading symbol
            timeframe: Timeframe in minutes
            num_bars: Number of bars to fetch
        """
        self.logger = logging.getLogger(__name__)
        self.mt5_connector = mt5_connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars

    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch and process market data.
        
        Returns:
            Processed DataFrame or None if failed
        """
        # Add buffer for historical data requirements
        buffer_bars = 70  # Match environment's lookback requirements
        total_bars = self.num_bars + buffer_bars
        
        rates = self.mt5_connector.fetch_data(
            self.symbol, 
            self.timeframe, 
            total_bars
        )

        if rates is None:
            self.logger.warning(f"No data returned. Error: {mt5.last_error()}")
            return None
            
        if len(rates) < self.num_bars:
            self.logger.error(f"Insufficient data: received {len(rates)} bars, need minimum {self.num_bars}")
            return None
            
        # Format the data
        data = self._format_data(rates)
        if data is None:
            self.logger.error("Data formatting failed")
            return None
            
        if len(data) < self.num_bars:
            self.logger.error(f"Insufficient data after formatting: {len(data)} bars (need {self.num_bars})")
            return None
        
        # Log data range
        self.logger.debug(f"Data collected from {data.index[0]} to {data.index[-1]}")
        self.logger.debug(data.tail())
            
        return data
        
    def fetch_current_bar(self, include_history: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch current bar data with optional historical bars for LSTM preloading.
        
        Args:
            include_history: If True, include historical bars for LSTM state preloading
            
        Returns:
            DataFrame with bar data or None if failed
        """
        try:
            min_bars = 100 if include_history else 1
            extra_bars = 70 if include_history else 0  # Only add buffer for history mode
            total_bars = min_bars + extra_bars
            
            rates = self.mt5_connector.fetch_data(
                self.symbol,
                self.timeframe,
                total_bars
            )

            if rates is None:
                self.logger.warning(f"No bar data returned. Error: {mt5.last_error()}")
                return None

            if include_history and len(rates) < min_bars:
                self.logger.error(f"Insufficient bars: got {len(rates)}, need {min_bars}")
                return None

            df = self._format_data(rates)
            if df is None:
                self.logger.error("Data formatting failed")
                return None
                
            # For single bar requests, extract just the last bar
            if not include_history:
                df = df.tail(1)
                if len(df) == 0:
                    self.logger.error("No data after extracting last bar")
                    return None
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching bar data: {e}")
            return None

    def _format_data(self, data: Any) -> pd.DataFrame:
        """
        Format market data into DataFrame.
        
        Args:
            data: Raw data from MT5
            
        Returns:
            Processed DataFrame with OHLCV data
        """
        try:
            df = pd.DataFrame(data)
            
            # Convert time and set as index
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Handle volume and cleanup
            df['volume'] = df['tick_volume']
            df.drop(columns=['real_volume', 'tick_volume'], inplace=True)
            df.dropna(inplace=True)

            if len(df) == 0:
                self.logger.error("No data available after cleaning")
                return None

            # Return only the required number of bars
            return df.tail(self.num_bars)

        except Exception as e:
            self.logger.error(f"Error formatting data: {e}")
            return None
