import time
import logging
from mt5_connector import MT5Connector
from data_fetcher import DataFetcher
from ppo_model import PPOModel
from trade_executor import TradeExecutor
from config import *
from datetime import datetime

# Generate log file name based on the current date
log_file = datetime.now().strftime("DRL_PPO_Bot_%Y-%m-%d.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(f"{LOG_FILE_PATH}/{log_file}", mode='a', encoding='utf-8')  # Logs to file with date-based name
    ]
)

def main():
    mt5 = MT5Connector()
    mt5.connect()

    data_fetcher = DataFetcher(mt5, MT5_SYMBOL, MT5_TIMEFRAME_MINUTES, BARS_TO_FETCH)
    model = PPOModel(MODEL_PATH)
    trade_executor = TradeExecutor(mt5)

    current_bar = data_fetcher.fetch_current_bar()
    last_bar_index = current_bar.index[-1]

    while True:
        current_bar = data_fetcher.fetch_current_bar()      
        data = data_fetcher.fetch_data()

        trade_action, sl, tp = model.predict(data)
        trade_executor.execute_trade(trade_action, sl, tp)

        while last_bar_index == current_bar.index[-1]:            
            time.sleep(1)
            current_bar = data_fetcher.fetch_current_bar()

        last_bar_index = current_bar.index[-1]


if __name__ == "__main__":
    main()