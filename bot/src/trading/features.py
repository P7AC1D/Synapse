"""Feature calculation and preprocessing for trading environment."""
import numpy as np
import pandas as pd
import talib as ta
from typing import Tuple, Dict, Any
from gymnasium import spaces

class FeatureProcessor:
    """Handles feature calculation and preprocessing."""
    
    def __init__(self):
        """Initialize feature processor with default parameters."""
        self.atr_period = 14
        self.rsi_period = 14
        self.boll_period = 20
        self.lookback = max(self.boll_period, self.atr_period)

    def setup_observation_space(self, feature_count: int = 11) -> spaces.Box:
        """Setup observation space with proper feature bounds.
        
        Args:
            feature_count: Number of features in observation space

        Returns:
            Box space with feature bounds
        """
        return spaces.Box(
            low=-1, high=1, shape=(feature_count,), dtype=np.float32
        )

    def _calculate_indicators(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Calculate technical indicators using TA-Lib.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Tuple of (ATR, RSI, (upper band, lower band), trend strength)
        """
        # Calculate ATR
        atr = ta.ATR(high, low, close, timeperiod=self.atr_period)
        atr = pd.Series(atr).bfill().fillna(pd.Series(atr).mean()).values
        
        # Calculate RSI
        rsi = ta.RSI(close, timeperiod=self.rsi_period)
        rsi = pd.Series(rsi).bfill().fillna(50).values  # Default to neutral RSI
        
        # Calculate Bollinger Bands
        upper, middle, lower = ta.BBANDS(close, timeperiod=self.boll_period)
        upper = pd.Series(upper).bfill().fillna(pd.Series(close).iloc[0]).values
        lower = pd.Series(lower).bfill().fillna(pd.Series(close).iloc[0]).values
        
        # Calculate ADX for trend strength
        adx = ta.ADX(high, low, close, timeperiod=self.atr_period)
        adx = pd.Series(adx).bfill().fillna(0).values
        trend_strength = np.clip(adx/25 - 1, -1, 1)  # Same normalization as before
        
        return atr, rsi, (upper, lower), trend_strength

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data and calculate features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features DataFrame, ATR values)
        """
        
        features_df = pd.DataFrame(index=data.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            opens = data['open'].values
            
            # Calculate technical indicators using TA-Lib
            atr, rsi, (upper_band, lower_band), trend_strength = self._calculate_indicators(high, low, close)
            
            # Returns and time features
            returns = np.diff(close) / close[:-1]
            returns = np.insert(returns, 0, 0)
            returns = np.clip(returns, -0.1, 0.1)
            
            minutes_in_day = 24 * 60
            time_index = pd.to_datetime(data.index).hour * 60 + pd.to_datetime(data.index).minute
            sin_time = np.sin(2 * np.pi * time_index / minutes_in_day)
            cos_time = np.cos(2 * np.pi * time_index / minutes_in_day)
            
            # Calculate price action signals
            body = close - opens
            upper_wick = high - np.maximum(close, opens)
            lower_wick = np.minimum(close, opens) - low
            range_ = high - low + 1e-8
            
            candle_pattern = (body/range_ + 
                           (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2
            candle_pattern = np.clip(candle_pattern, -1, 1)
            
            # Calculate volatility breakout
            band_range = upper_band - lower_band
            band_range = np.where(band_range < 1e-8, 1e-8, band_range)
            position = close - lower_band
            volatility_breakout = np.divide(position, band_range, out=np.zeros_like(position), where=band_range!=0)
            volatility_breakout = np.clip(volatility_breakout, 0, 1)
            
            # Create features with preserved index
            features = {
                'returns': returns,
                'rsi': rsi / 50 - 1,  # Normalize to [-1, 1]
                'atr': 2 * (atr / close - np.nanmin(atr / close)) / 
                      (np.nanmax(atr / close) - np.nanmin(atr / close) + 1e-8) - 1,
                'volatility_breakout': volatility_breakout,
                'trend_strength': trend_strength,
                'candle_pattern': candle_pattern,
                'sin_time': sin_time,
                'cos_time': cos_time
            }
            
            # Convert all features to DataFrame at once
            features_df = pd.DataFrame(features, index=data.index)
            
            # Clean up features
            orig_len = len(features_df)
            orig_index = features_df.index
            
            features_df = features_df.dropna()
            post_dropna_len = len(features_df)
            dropped_index = orig_index.difference(features_df.index)
            # Validate lookback size
            if self.lookback > len(features_df) * 0.3:  # Don't allow more than 30% data loss
                raise ValueError(f"Lookback window ({self.lookback}) is too large for data length ({len(features_df)})")
                
            if len(dropped_index) > 0:
                print(f"\nDropped rows due to NaN at indices:")
                print(dropped_index.to_list()[:5], "..." if len(dropped_index) > 5 else "")
            
            features_df = features_df.iloc[self.lookback:]
            post_lookback_len = len(features_df)
            lookback_removed = features_df.index[0] - orig_index[0]
            
            # Validate remaining data
            if len(features_df) < max(100, len(orig_index) * 0.5):
                raise ValueError(f"Too much data lost in preprocessing: {len(features_df)} rows remaining from {len(orig_index)}")
            
            # Check for any remaining NaN values in features
            nan_counts = features_df.isna().sum()
            if nan_counts.any():
                print("\nWarning: NaN values in features:")
                for col in features_df.columns[nan_counts > 0]:
                    print(f"  {col}: {nan_counts[col]} NaN values")
            
            if len(features_df) < 100:
                raise ValueError("Insufficient data after preprocessing: need at least 100 bars")            
            
            # Convert ATR to DataFrame for proper index alignment
            atr_df = pd.DataFrame({'atr': atr}, index=data.index)
            
            # Align ATR with features using index
            atr_aligned = atr_df.loc[features_df.index].values
            
            # Validate alignment
            if len(atr_aligned) != len(features_df):
                raise ValueError(f"Feature and ATR lengths don't match after preprocessing: features={len(features_df)}, atr={len(atr_aligned)}")
            
            # Final validation
            if np.isnan(atr_aligned).any():
                raise ValueError(f"Found {np.isnan(atr_aligned).sum()} NaN values in aligned ATR data")
            
            # Validate feature ranges
            for col, values in features_df.items():
                if col in ['volatility_breakout']:
                    if (values < 0).any() or (values > 1).any():
                        raise ValueError(f"Feature {col} contains values outside [0, 1] range")
                elif col not in ['returns']:  # Returns has special range [-0.1, 0.1]
                    if (values < -1).any() or (values > 1).any():
                        raise ValueError(f"Feature {col} contains values outside [-1, 1] range")                        
                
            return features_df, atr_aligned.reshape(-1)  # Ensure 1D array

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'returns',          # [-0.1, 0.1] Price momentum
            'rsi',             # [-1, 1] Momentum oscillator
            'atr',             # [-1, 1] Volatility indicator
            'volatility_breakout', # [0, 1] Trend with volatility context
            'trend_strength',   # [-1, 1] ADX-based trend quality
            'candle_pattern',   # [-1, 1] Combined price action signal
            'sin_time',        # [-1, 1] Sine encoding of time
            'cos_time',        # [-1, 1] Cosine encoding of time
            'position_type',    # [-1, 0, 1] Current position
            'hold_time',       # [0, 1] Normalized hold time
            'unrealized_pnl'   # [-1, 1] Current position P&L
        ]
