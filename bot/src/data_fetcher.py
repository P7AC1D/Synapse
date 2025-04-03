"""Market data fetching and processing module."""

import logging
from datetime import datetime
from typing import Optional, Any

import pandas as pd
import pytz
import ta
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
        # Fetch double the required bars to have enough data for indicators
        rates = self.mt5_connector.fetch_data(
            self.symbol, 
            self.timeframe, 
            self.num_bars + 50
        )

        if rates is None or len(rates) == 0:
            self.logger.warning(f"No data returned. Error: {mt5.last_error()}")
            return None
            
        return self._format_data(rates)
        
    def fetch_current_bar(self, include_history: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch current bar data with optional historical bars for LSTM preloading.
        
        Args:
            include_history: If True, include historical bars for LSTM state preloading
            
        Returns:
            DataFrame with bar data or None if failed
        """
        try:
            # Determine how many bars to fetch
            if include_history:
                # Fetch more bars for LSTM preloading
                preload_bars = min(50, self.num_bars // 4)  # Use 25% of configured bars or 50, whichever is smaller
                total_bars = preload_bars + 1  # +1 for current bar
                rates = self.mt5_connector.fetch_data(self.symbol, self.timeframe, total_bars)
            else:
                rates = self.mt5_connector.fetch_current_bar(self.symbol, self.timeframe)

            if rates is None or len(rates) == 0:
                self.logger.warning(f"No bar data returned. Error: {mt5.last_error()}")
                return None

            df = self._format_data(rates)
            
            # For single bar requests, extract just the last bar
            if not include_history:
                df = df.tail(1)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching bar data: {e}")
            return None

    def _format_current_bar(self, data: Any) -> pd.DataFrame:
        """
        Format current bar data.
        
        Args:
            data: Raw data from MT5
            
        Returns:
            Formatted DataFrame
        """
        df = pd.DataFrame(data)
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df['volume'] = df['tick_volume']
        df.drop(columns=['real_volume', 'tick_volume'], inplace=True)
        df.drop(columns=['spread'], inplace=True)
        df.dropna(inplace=True)
        
        return df

    def _format_data(self, data: Any) -> pd.DataFrame:
        """
        Format and add technical indicators to market data.
        
        Args:
            data: Raw data from MT5
            
        Returns:
            Processed DataFrame with technical indicators
        """
        df = pd.DataFrame(data)
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df['volume'] = df['tick_volume']

        try:
            # Remove the last row as it might be incomplete
            df = df.iloc[:-1]

            if len(df) == 0:
                self.logger.error("No data available after removing incomplete bar")
                return None

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Clean up columns
            df.drop(columns=['real_volume', 'tick_volume'], inplace=True)
            df.dropna(inplace=True)

            if len(df) == 0:
                self.logger.error("No data available after cleaning and calculating indicators")
                return None

            # Log data range
            start_datetime = df.index[0]
            end_datetime = df.index[-1]
            self.logger.debug(f"Data collected from {start_datetime} to {end_datetime}")

            if len(df) < self.num_bars:
                self.logger.warning(f"Insufficient data: only {len(df)} bars available")
                return None

            # Return only the required number of bars
            return df.tail(self.num_bars)

        except Exception as e:
            self.logger.error(f"Error formatting data: {e}")
            return None
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            DataFrame with indicators
        """
        # Trend Indicators
        df['EMA_fast'] = ta.trend.ema_indicator(df['close'], window=9)
        df['EMA_slow'] = ta.trend.ema_indicator(df['close'], window=50)

        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['close'])

        # Volatility Indicators
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['VWAP'] = ta.volume.volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        return df
