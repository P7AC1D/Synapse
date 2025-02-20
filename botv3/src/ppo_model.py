import logging
import pandas as pd
import joblib
import MetaTrader5 as mt5
from sklearn.preprocessing import RobustScaler
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from config import *
from trade_environment import BitcoinTradingEnv
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Custom functions to replace clip_range and lr_schedule
def get_custom_clip_range():
    return 0.2  # Set your desired clipping value here

def get_custom_lr_schedule():
    return get_schedule_fn(0.00001)  # Use your desired learning rate

class PPOModel:
    def __init__(self, model_path, scaler_path):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)

    def load_model(self, model_path):
        try:            
            # Load the model with custom objects
            model = PPO.load(model_path, custom_objects={
                "clip_range": get_custom_clip_range,
                "lr_schedule": get_custom_lr_schedule
            })
            logging.info(f"Model loaded successfully from: {model_path}")

            # Correct logging to handle the action space
            logging.debug(f"Model action space: {model.action_space}")
            logging.debug(f"Model observation space: {model.observation_space}")

            return model
        except Exception as e:
            logging.critical(f"Error loading model: {e}")
            return None
        
    def load_scaler(self, scaler_path):
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            logging.critical(f"Error loading scaler: {e}")
            raise

    def predict(self, df):
        action, _ = self.model.predict(df, deterministic=True)

        trade_action, sl_index, tp_index, close_index = action
        sl = SL_TP_LEVELS[sl_index]
        tp = SL_TP_LEVELS[tp_index]
        
        trade_action_string = "Buy" if trade_action == 1 else "Sell" if trade_action == 2 else "Close" if trade_action == 3 else "Hold"
        logging.debug(f"Action: {trade_action_string} | SL: {sl} | TP: {tp} | Close: {close_index}")
        
        return trade_action, sl, tp, close_index
    
    def scale_data(self, df):
        df_scaled = df.copy()

        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
        df_scaled["unscaled_close"] = df["close"]

        return df_scaled
    
    def get_observation(self, rates, positions):
        env = BitcoinTradingEnv(rates)
        history_obs = env.get_history()
        position_obs = np.zeros((env.max_positions, env.position_features), dtype=np.float32)

        open_positions = []
        for pos in positions:
            open_positions.append({
                "trade_id": pos.ticket,
                "position": 1 if pos.type == mt5.ORDER_TYPE_BUY else -1,
                "entry_price": pos.price_open,
                "sl_price": pos.sl,
                "tp_price": pos.tp,
                "lot_size": pos.volume,
                "entry_step": 0  # Needs to be adjusted based on history
            })

        for i, pos in enumerate(open_positions[:env.max_positions]):
            position_obs[i] = [
                pos["trade_id"],
                pos["position"],
                pos["entry_price"],
                pos["sl_price"],
                pos["tp_price"],
                pos["lot_size"],
                pos["entry_step"]
            ]

        observation = np.concatenate([history_obs, position_obs.flatten()])
        return observation