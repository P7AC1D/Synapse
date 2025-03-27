import numpy as np

MT5_SYMBOL = "BTCUSDm"
MT5_BASE_SYMBOL = "USDZARm"
MT5_PATH = "C:/Program Files/MetaTrader 5 EXNESS/terminal64.exe"
MT5_TIMEFRAME_MINUTES = 15  # Updated to match training timeframe
MT5_COMMENT = "PPO_LSTM"  # Updated to reflect current model
BARS_TO_FETCH = 10  # Match bar_count used in training
RISK_PERCENTAGE = 2  # More conservative risk setting
LOG_FILE_PATH = f"C:/Users/Admin/Desktop"
MODEL_PATH = f"C:/Code/drl/bot/model/{MT5_SYMBOL}.zip"
SCALER_PATH = f"C:/Code/drl/bot/model/{MT5_SYMBOL}.pkl"
MAX_SPREAD = 35.0
