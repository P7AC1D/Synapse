"""Implementation of paper-inspired feature processing using price ratios."""
import numpy as np
import pandas as pd
from typing import Tuple
from gymnasium import spaces
from .features import FeatureProcessor

class PaperInspiredFeatureProcessor(FeatureProcessor):
    """Feature processor implementing paper's price ratio methodology."""
    
    def __init__(self, window_size: int = 10):
        """Initialize paper-inspired feature processor.
        
        Args:
            window_size: Size of the rolling window for temporal alignment
        """
        super().__init__()
        self.window_size = window_size
        self.lookback = window_size  # Required for proper temporal alignment
        
    def setup_observation_space(self, feature_count: int = 6) -> spaces.Box:
        """Setup observation space for price ratio features.
        
        Args:
            feature_count: Number of features (6 base features including volume)
            
        Returns:
            Box space with feature bounds
        """
        # All ratio features are theoretically unbounded but practically within [-1, 1]
        return spaces.Box(
            low=-1, high=1, shape=(feature_count,), dtype=np.float32
        )
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data using price ratio methodology.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features DataFrame, ATR values for position sizing)
        """
        features_df = pd.DataFrame(index=data.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            
            # Calculate price ratios as per paper
            
            # x1t = (Pct - Pct-1) / Pct-1  # Close price change ratio
            close_ratio = np.zeros_like(close)
            close_ratio[1:] = (close[1:] - close[:-1]) / close[:-1]
            
            # x2t = (Pht - Pht-1) / Pht-1  # High price change ratio
            high_ratio = np.zeros_like(high)
            high_ratio[1:] = (high[1:] - high[:-1]) / high[:-1]
            
            # x3t = (Plt - Plt-1) / Plt-1  # Low price change ratio
            low_ratio = np.zeros_like(low)
            low_ratio[1:] = (low[1:] - low[:-1]) / low[:-1]
            
            # x4t = (Pht - Pct) / Pct  # Upper shadow ratio
            upper_shadow = (high - close) / close
            
            # x5t = (Pct - Plt) / Pct  # Lower shadow ratio
            lower_shadow = (close - low) / close
            
            # Calculate volume change
            volume = data['volume'].values.astype(np.float64)
            volume_pct = np.zeros_like(volume, dtype=np.float64)
            volume_pct[1:] = np.divide(
                volume[1:] - volume[:-1],
                volume[:-1],
                out=np.zeros(len(volume)-1, dtype=np.float64),
                where=volume[:-1] != 0
            )
            volume_pct = np.clip(volume_pct, -1, 1)
            
            # Create features DataFrame
            features = {
                'close_ratio': close_ratio,
                'high_ratio': high_ratio,
                'low_ratio': low_ratio,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'volume_change': volume_pct
            }
            features_df = pd.DataFrame(features, index=data.index)
            
            # Calculate ATR for position sizing (maintain compatibility)
            atr = self._calculate_atr(data)
            atr_df = pd.DataFrame({'atr': atr}, index=data.index)
            
            # Clean up NaN values
            features_df = features_df.dropna()
            
            # Apply lookback window
            if len(features_df) > self.lookback:
                features_df = features_df.iloc[self.lookback:]
                common_index = features_df.index.intersection(atr_df.index)
                features_df = features_df.loc[common_index]
                atr_df = atr_df.loc[common_index]
            
            # Normalize features to [-1, 1] range
            for col in features_df.columns:
                features_df[col] = np.clip(features_df[col], -1, 1)
            
            # Validate data
            if len(features_df) < 100:
                raise ValueError("Insufficient data after preprocessing")
            
            return features_df, atr_df.values.reshape(-1)
            
    def _calculate_atr(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range for position sizing.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ATR values as numpy array
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr = np.zeros_like(high)
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])
        
        # Calculate ATR using simple moving average
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values
        return atr

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'close_ratio',    # [-1, 1] Normalized close price change
            'high_ratio',     # [-1, 1] Normalized high price change
            'low_ratio',      # [-1, 1] Normalized low price change
            'upper_shadow',   # [-1, 1] Normalized upper shadow
            'lower_shadow',   # [-1, 1] Normalized lower shadow
            'volume_change',  # [-1, 1] Volume percentage change
            'position_type',  # [-1, 0, 1] Current position (added by env)
            'unrealized_pnl'  # [-1, 1] Current position P&L (added by env)
        ]
