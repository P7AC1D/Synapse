import numpy as np

MT5_SYMBOL = "BTCUSDm"
MT5_BASE_SYMBOL = "USDZARm"
MT5_PATH = "C:/Program Files/MetaTrader 5 EXNESS/terminal64.exe"
MT5_TIMEFRAME_MINUTES = 60
MT5_COMMENT = f"DQN"
BARS_TO_FETCH = 250
RISK_PERCENTAGE = 5
LOG_FILE_PATH = f"C:/Users/Admin/Desktop"
MODEL_PATH = f"C:/Code/drl/botv3/model/{MT5_SYMBOL}.zip"
SCALER_PATH = f"C:/Code/drl/botv3/model/{MT5_SYMBOL}.pkl"
MAX_SPREAD = 35.0