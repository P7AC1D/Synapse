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
            self.num_bars * 2
        )

        if rates is None or len(rates) == 0:
            self.logger.warning(f"No data returned. Error: {mt5.last_error()}")
            return None
            
        return self._format_data(rates)
        
    def fetch_current_bar(self) -> Optional[pd.DataFrame]:
        """
        Fetch only the current bar data.
        
        Returns:
            DataFrame with current bar or None if failed
        """
        try:
            rates = self.mt5_connector.fetch_current_bar(self.symbol, self.timeframe)

            if rates is None or len(rates) == 0:
                self.logger.warning(f"No current bar data returned. Error: {mt5.last_error()}")
                return None
                
            return self._format_current_bar(rates)
            
        except Exception as e:
            self.logger.error(f"Error fetching current bar: {e}")
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

        # Remove the last row as it might be incomplete
        df = df.iloc[:-1]

        # Add technical indicators
        df = self._add_technical_indicators(df)

        # Clean up columns
        df.drop(columns=['real_volume', 'tick_volume'], inplace=True)
        df.dropna(inplace=True)

        # Log data range
        start_datetime = df.index[0]
        end_datetime = df.index[-1]
        self.logger.debug(f"Data collected from {start_datetime} to {end_datetime}")

        # Return only the required number of bars
        return df.tail(self.num_bars)
        
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