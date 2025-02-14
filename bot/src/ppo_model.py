import logging
from stable_baselines3 import PPO
from config import *
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PPOModel:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        try:
            model = PPO.load(model_path)
            logging.info(f"Model loaded successfully from: {model_path}")

            # Correct logging to handle the action space
            logging.debug(f"Model action space: {model.action_space}")
            logging.debug(f"Model observation space: {model.observation_space}")

            return model
        except Exception as e:
            logging.critical(f"Error loading model: {e}")
            return None

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
