import logging
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from config import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Custom functions to replace clip_range and lr_schedule
def get_custom_clip_range():
    return 0.2  # Set your desired clipping value here

def get_custom_lr_schedule():
    return get_schedule_fn(0.001)  # Use your desired learning rate

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
        # Check if df is a single observation or a batch
        if df.ndim > 1 and df.shape[0] > 1:
            logging.debug(f"Received a batch of {df.shape[0]} observations. Expecting a single observation.")
            df = df.iloc[0]  # Use only the first observation from the batch, using iloc for DataFrame indexing

        action, _ = self.model.predict(df)        
        if len(action) != 3:
            logging.critical(f"Unexpected action length: {len(action)}. Expected 3.")
            raise ValueError(f"Unexpected action length: {len(action)}")

        trade_action, sl_index, tp_index = action
        sl = SL_TP_LEVELS[sl_index]
        tp = SL_TP_LEVELS[tp_index]
        
        trade_action_string = "Buy" if trade_action == 1 else "Sell" if trade_action == 2 else "Hold"
        logging.debug(f"Action: {trade_action_string} | SL: {sl} | TP: {tp}")
        
        return trade_action, sl, tp
    
    def scale_data(self, df):
        df_scaled = df.copy()

        df_scaled = pd.DataFrame(self.scaler.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
        df_scaled["unscaled_close"] = df["close"]

        return df_scaled