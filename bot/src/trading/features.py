"""Feature calculation and preprocessing for trading environment."""
import numpy as np
import pandas as pd
from gymnasium import spaces

class FeatureProcessor:
    """Handles feature calculation and preprocessing."""
    
    def setup_observation_space(self, feature_count: int = 9) -> spaces.Box:
        """Setup observation space with proper feature bounds.
        
        Args:
            feature_count: Number of features in observation space

        Returns:
            Box space with feature bounds
        """
        return spaces.Box(
            low=-1, high=1, shape=(feature_count,), dtype=np.float32
        )

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data and calculate features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame containing the calculated features
        """
        # Extract raw price data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        opens = data['open'].values
        
        features = {
            'returns': np.clip(np.diff(close, prepend=close[0]) / close, -0.1, 0.1),
            'candle_pattern': np.clip((close - opens) / (high - low + 1e-8) + 
                                    ((high - np.maximum(close, opens)) - (np.minimum(close, opens) - low)) /
                                    (high - low + 1e-8) / 2, -1, 1),
            'sin_time': np.sin(2 * np.pi * (pd.to_datetime(data.index).hour * 60 + 
                                          pd.to_datetime(data.index).minute) / (24 * 60)),
            'cos_time': np.cos(2 * np.pi * (pd.to_datetime(data.index).hour * 60 + 
                                          pd.to_datetime(data.index).minute) / (24 * 60)),
            'volume_change': np.clip(np.diff(data['volume'].values, prepend=data['volume'].values[0]) / 
                                   (data['volume'].values + 1e-8), -1, 1),
            'sin_weekday': np.sin(2 * np.pi * pd.to_datetime(data.index).dayofweek / 7),
            'cos_weekday': np.cos(2 * np.pi * pd.to_datetime(data.index).dayofweek / 7)
        }
        
        features_df = pd.DataFrame(features, index=data.index)
        
        return features_df

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'returns',          # [-0.1, 0.1] Price momentum
            'candle_pattern',   # [-1, 1] Combined price action signal
            'sin_time',        # [-1, 1] Sine encoding of time
            'cos_time',        # [-1, 1] Cosine encoding of time
            'volume_change',    # [-1, 1] Volume percentage change
            'sin_weekday',     # [-1, 1] Sine encoding of weekday
            'cos_weekday',     # [-1, 1] Cosine encoding of weekday
            'position_type',    # [-1, 0, 1] Current position
            'unrealized_pnl'   # [-1, 1] Current position P&L
        ]
