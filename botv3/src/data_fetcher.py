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
        rates = self.mt5_connector.fetch_data(self.symbol, self.timeframe, self.num_bars)

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

        df.drop(columns=['real_volume', 'tick_volume'], inplace = True)
        df.drop(columns=['spread'], inplace = True)
        df.dropna(inplace=True)

        return df