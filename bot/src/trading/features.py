"""Feature calculation and preprocessing for trading environment."""
import numpy as np
import pandas as pd
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

    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            ATR values
        """
        tr = np.maximum(high - low,
                     np.maximum(np.abs(high - np.roll(close, 1)),
                              np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  
        
        # Calculate ATR with proper NaN handling
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean()
        # Fill any remaining NaNs with first valid value
        atr = atr.bfill().fillna(atr.mean())
        return atr.values

    def _calculate_rsi(self, close: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index.
        
        Args:
            close: Close prices
            
        Returns:
            RSI values
        """
        delta = pd.Series(close).diff().fillna(0)
        gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        
        # Fill any initial NaN values
        gain = gain.bfill().fillna(0)
        loss = loss.bfill().fillna(0)
        
        rs = np.zeros_like(gain)
        mask = loss != 0
        rs[mask] = gain[mask] / loss[mask]
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands.
        
        Args:
            close: Close prices
            
        Returns:
            Tuple of (upper band, lower band)
        """
        price_series = pd.Series(close)
        ma20 = price_series.rolling(self.boll_period, min_periods=1).mean()
        boll_std = price_series.rolling(self.boll_period, min_periods=1).std()
        
        # Handle NaN values
        ma20 = ma20.bfill().fillna(price_series.iloc[0])
        boll_std = boll_std.bfill().fillna(boll_std.mean())
        
        upper_band = (ma20 + (boll_std * 2)).values
        lower_band = (ma20 - (boll_std * 2)).values
        return upper_band, lower_band

    def _calculate_trend_strength(self, high: np.ndarray, low: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """Calculate trend strength using ADX.
        
        Args:
            high: High prices
            low: Low prices
            atr: ATR values
            
        Returns:
            Trend strength values
        """
        pdm = np.maximum(high[1:] - high[:-1], 0)
        ndm = np.maximum(low[:-1] - low[1:], 0)
        pdm = np.insert(pdm, 0, 0)
        ndm = np.insert(ndm, 0, 0)
        
        pdm_smooth = pd.Series(pdm).rolling(self.atr_period, min_periods=1).mean().fillna(0)
        ndm_smooth = pd.Series(ndm).rolling(self.atr_period, min_periods=1).mean().fillna(0)
        
        atr_safe = np.where(atr < 1e-8, 1e-8, atr)
        pdi = (pdm_smooth.values / atr_safe) * 100
        ndi = (ndm_smooth.values / atr_safe) * 100
        
        sum_di = pdi + ndi
        sum_di = np.where(sum_di < 1e-8, 1e-8, sum_di)
        dx = np.abs(pdi - ndi) / sum_di * 100
        adx = pd.Series(dx).rolling(self.atr_period, min_periods=1).mean().fillna(0).values
        return np.clip(adx/25 - 1, -1, 1)

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
            
            # Calculate technical indicators
            atr = self._calculate_atr(high, low, close)
            rsi = self._calculate_rsi(close)
            upper_band, lower_band = self._calculate_bollinger_bands(close)
            trend_strength = self._calculate_trend_strength(high, low, atr)
            
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
