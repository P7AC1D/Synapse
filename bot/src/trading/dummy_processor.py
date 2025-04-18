"""Feature processor with relaxed validation for test data generation."""
import numpy as np
import pandas as pd
import ta
from typing import Tuple, Dict, Any
from gymnasium import spaces
from .features import FeatureProcessor

class DummyFeatureProcessor(FeatureProcessor):
    """Feature processor with relaxed validation for test data."""
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data with relaxed validation.
        
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
            
            # Calculate volume percentage change
            volume = data['volume'].values.astype(np.float64)
            volume_pct = np.zeros(len(volume), dtype=np.float64)
            volume_pct[1:] = np.divide(
                np.diff(volume),
                volume[:-1],
                out=np.zeros(len(volume)-1, dtype=np.float64),
                where=volume[:-1] != 0
            )
            volume_pct = np.clip(volume_pct, -1, 1)

            # Calculate ATR relative to its moving average
            window_size = 20
            atr_sma = np.convolve(atr, np.ones(window_size)/window_size, mode='valid')
            atr_sma = np.pad(atr_sma, (window_size-1, 0), mode='edge')
            atr_ratio = atr / (atr_sma + 0.00000001)

            # Map ATR ratio from [0.5, 2.0] to [-1, 1] (same as MQL5)
            min_expected_ratio = 0.5
            max_expected_ratio = 2.0
            expected_range = max_expected_ratio - min_expected_ratio
            norm_atr = 2.0 * (atr_ratio - min_expected_ratio) / expected_range - 1.0
            norm_atr = np.clip(norm_atr, -1, 1)

            # Create features DataFrame in same order as MQL5
            features = {
                'returns': returns,
                'rsi': rsi / 50 - 1,
                'atr': norm_atr,
                'volume_change': volume_pct,
                'volatility_breakout': volatility_breakout,
                'trend_strength': trend_strength,
                'candle_pattern': candle_pattern,
                'sin_time': sin_time,
                'cos_time': cos_time
            }
            
            features_df = pd.DataFrame(features, index=data.index)
            
            # Clean up features with minimal validation
            features_df = features_df.fillna(0)  # Fill any NaN values with 0
            features_df = features_df.iloc[self.lookback:]  # Remove lookback period
            
            # Convert ATR to DataFrame and align
            atr_df = pd.DataFrame({'atr': atr}, index=data.index)
            atr_aligned = atr_df.loc[features_df.index].values
            atr_aligned = np.nan_to_num(atr_aligned, 0)  # Replace NaN with 0
            
            return features_df, atr_aligned.reshape(-1)
