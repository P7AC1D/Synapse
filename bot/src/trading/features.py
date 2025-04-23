"""Feature calculation and preprocessing for trading environment."""
import numpy as np
import pandas as pd
from gymnasium import spaces

class FeatureProcessor:
    """Handles feature calculation and preprocessing."""
    
    def setup_observation_space(self, feature_count: int = 15) -> spaces.Box:
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
            'range_ratio': np.clip((high - low) / (high + low + 1e-8), 0, 1),
            'gap_ratio': np.clip((opens - np.roll(close, 1)) / (np.roll(close, 1) + 1e-8), -0.1, 0.1),
            'high_change': np.clip(np.diff(high, prepend=high[0]) / high, -0.1, 0.1),
            'low_change': np.clip(np.diff(low, prepend=low[0]) / low, -0.1, 0.1),
            'upper_shadow': (high - np.maximum(close, opens)) / (high - low + 1e-8),
            'lower_shadow': (np.minimum(close, opens) - low) / (high - low + 1e-8),
            'body_ratio': (close - opens) / (high - low + 1e-8),
            'high_position': (close - low) / (high - low + 1e-8),
            'pivot_position': (close - np.mean([high, low, opens], axis=0)) / (high - low + 1e-8),
            'sin_time': np.sin(2 * np.pi * (pd.to_datetime(data.index).hour * 60 + 
                                          pd.to_datetime(data.index).minute) / (24 * 60)),
            'cos_time': np.cos(2 * np.pi * (pd.to_datetime(data.index).hour * 60 + 
                                          pd.to_datetime(data.index).minute) / (24 * 60)),
            'volume_change': np.clip(np.diff(data['volume'].values, prepend=data['volume'].values[0]) / 
                                   (data['volume'].values + 1e-8), -1, 1)
        }
        
        features_df = pd.DataFrame(features, index=data.index)
        
        return features_df

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'returns',          # [-0.1, 0.1] Price momentum
            'range_ratio',      # [0, 1] Normalized price range
            'gap_ratio',        # [-0.1, 0.1] Gap between candles
            'high_change',      # [-0.1, 0.1] High price momentum
            'low_change',       # [-0.1, 0.1] Low price momentum
            'upper_shadow',     # [0, 1] Upper wick ratio
            'lower_shadow',     # [0, 1] Lower wick ratio
            'body_ratio',       # [-1, 1] Body to range ratio
            'high_position',    # [0, 1] Close position in range
            'pivot_position',   # [-1, 1] Position relative to pivot
            'sin_time',         # [-1, 1] Sine encoding of time
            'cos_time',         # [-1, 1] Cosine encoding of time
            'volume_change',    # [-1, 1] Volume percentage change
            'position_type',    # [-1, 0, 1] Current position
            'unrealized_pnl'   # [-1, 1] Current position P&L
        ]
