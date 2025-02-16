import numpy as np

MT5_SYMBOL = "BTCUSDm"
MT5_BASE_SYMBOL = "USDZARm"
MT5_PATH = "C:/Program Files/MetaTrader 5 EXNESS/terminal64.exe"
MT5_TIMEFRAME_MINUTES = 15
MT5_COMMENT = f"PPO-V2"
BARS_TO_FETCH = 60
RISK_PERCENTAGE = 5
SL_TP_LEVELS = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
LOG_FILE_PATH = f"C:/Users/Admin/Desktop"
MODEL_PATH = f"C:/Code/drl/bot/model/{MT5_SYMBOL}.zip"
SCALER_PATH = f"C:/Code/drl/bot/model/{MT5_SYMBOL}.pkl"
MAX_SPREAD = 35.0