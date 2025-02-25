import pandas as pd
import ta
import MetaTrader5 as mt5
import pytz
import logging
from mt5_connector import MT5Connector
from datetime import datetime, timezone
from config import *

class DataFetcher:
    def __init__(self, mt5_connector, symbol, timeframe, num_bars):
        self.mt5_connector = mt5_connector
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars

    def fetch_data(self):
        rates = self.mt5_connector.fetch_data(self.symbol, self.timeframe, self.num_bars * 2)

        if rates is None or len(rates) == 0:
            logging.warning(f"No data returned. | {mt5.last_error()}")
            return []
        else:
            return self.format_data(rates)
        
    def fetch_current_bar(self):
        rates = self.mt5_connector.fetch_current_bar(self.symbol, self.timeframe)

        if rates is None or len(rates) == 0:
            logging.warning(f"No data returned. | {mt5.last_error()}")
            raise
        else:
            df = pd.DataFrame(rates)
        
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            df['volume'] = df['tick_volume']
            df.drop(columns=['real_volume', 'tick_volume'], inplace = True)
            df.drop(columns=['spread'], inplace = True)
            df.dropna(inplace=True)
            return df

    def format_data(self, data):
        df = pd.DataFrame(data)
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df['volume'] = df['tick_volume']

        # Remove the last row as it might have not completed yet
        df = df.iloc[:-1]

        # Trend Indicators
        # df['EMA_fast'] = ta.trend.ema_indicator(df['close'], window=9)
        # df['EMA_medium'] = ta.trend.ema_indicator(df['close'], window=21)
        # df['EMA_slow'] = ta.trend.ema_indicator(df['close'], window=50)	
        # df['MACD'] = ta.trend.macd_diff(df['close'])

        # # Momentum Indicators
        # df['RSI'] = ta.momentum.rsi(df['close'])
        # df['Stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])

        # # Volatility Indicators
        # df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
        # df['BB_middle'] = ta.volatility.bollinger_mavg(df['close'])
        # df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
        # df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

        # # Volume Indicators
        # df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        # df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])	

        df.drop(columns=['real_volume', 'tick_volume'], inplace = True)
        df.drop(columns=['spread'], inplace = True)
        df.dropna(inplace=True)

        start_datetime = df.index[0]
        end_datetime = df.index[-1]
        logging.debug(f"Data collected from {start_datetime} to {end_datetime}")

        return df.tail(self.num_bars)