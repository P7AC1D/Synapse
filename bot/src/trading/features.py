"""Feature calculation and preprocessing for trading environment."""
import numpy as np
import pandas as pd
import ta
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
        """Calculate technical indicators using ta package.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Tuple of (ATR, RSI, (upper band, lower band), trend strength)
        """
        # Convert to pandas Series for ta package
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        # Calculate ATR
        atr = ta.volatility.AverageTrueRange(
            high=high_s, low=low_s, close=close_s, 
            window=self.atr_period
        ).average_true_range().values
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(
            close=close_s,
            window=self.rsi_period
        ).rsi().values
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=close_s,
            window=self.boll_period,
            window_dev=2
        )
        upper = bb.bollinger_hband().values
        lower = bb.bollinger_lband().values
        
        # Calculate ADX for trend strength
        adx = ta.trend.ADXIndicator(
            high=high_s, low=low_s, close=close_s,
            window=self.atr_period
        ).adx().values
        
        # Normalize trend strength
        trend_strength = np.zeros_like(adx)
        valid_mask = ~np.isnan(adx)
        trend_strength[valid_mask] = np.clip(adx[valid_mask]/25 - 1, -1, 1)
        
        return atr, rsi, (upper, lower), trend_strength

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data and calculate features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features DataFrame, ATR values)
        """
        
        # Keep track of original data length for logging
        original_length = len(data)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            opens = data['open'].values
            
            # Calculate technical indicators using TA-Lib
            atr, rsi, (upper_band, lower_band), trend_strength = self._calculate_indicators(high, low, close)
            
            # Calculate ATR normalization
            window_size = 20
            atr_series = pd.Series(atr, index=data.index)
            atr_sma = atr_series.rolling(window_size, min_periods=1).mean().values
            atr_ratio = np.divide(atr, atr_sma, where=atr_sma!=0, out=np.zeros_like(atr))
            
            # Scale ATR ratio
            min_expected_ratio = 0.5
            max_expected_ratio = 2.0
            expected_range = max_expected_ratio - min_expected_ratio
            atr_norm = 2 * (atr_ratio - min_expected_ratio) / expected_range - 1
            atr_norm = np.clip(atr_norm, -1, 1)
            
            # Calculate other features
            returns = np.zeros_like(close)
            returns[1:] = np.diff(close) / close[:-1]
            returns = np.clip(returns, -0.1, 0.1)
            minutes_in_day = 24 * 60            # Explicitly use UTC to prevent timezone conversions
            time_index = pd.to_datetime(data.index, utc=True).hour * 60 + pd.to_datetime(data.index, utc=True).minute
            
            # Calculate correct angle and trigonometric values
            angle = 2 * np.pi * time_index / minutes_in_day
            sin_time = np.sin(angle)
            cos_time = np.cos(angle)
            
            # Price action features
            body = close - opens
            upper_wick = high - np.maximum(close, opens)
            lower_wick = np.minimum(close, opens) - low
            range_ = high - low + 1e-8
            candle_pattern = (body/range_ + (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2
            candle_pattern = np.clip(candle_pattern, -1, 1)
            
            # Volatility breakout
            band_range = upper_band - lower_band
            band_range = np.where(band_range < 1e-8, 1e-8, band_range)
            position = close - lower_band
            volatility_breakout = np.divide(position, band_range, out=np.zeros_like(position), where=band_range!=0)
            volatility_breakout = np.clip(volatility_breakout, 0, 1)
            
            # Volume change
            volume = data['volume'].values.astype(np.float64)
            volume_pct = np.zeros_like(volume, dtype=np.float64)
            volume_pct[1:] = np.divide(
                volume[1:] - volume[:-1],
                volume[:-1],
                out=np.zeros(len(volume)-1, dtype=np.float64),
                where=volume[:-1] != 0
            )
            volume_pct = np.clip(volume_pct, -1, 1)            # Debug the last values of sin_time and cos_time
            if len(time_index) > 0:
                print(f"DEBUG: Last index sin_time value: {sin_time[-1]}, cos_time value: {cos_time[-1]}")              # Create features DataFrame with exact ordering to match get_feature_names
            features = {
                'returns': returns,
                'rsi': np.divide(rsi, 50, out=np.zeros_like(rsi), where=~np.isnan(rsi)) - 1,
                'atr': atr_norm,
                'volume_change': volume_pct,  # Moved to match get_feature_names order
                'volatility_breakout': volatility_breakout,
                'trend_strength': trend_strength,
                'candle_pattern': candle_pattern,
                'cos_time': cos_time,
                'sin_time': sin_time,
                # position_type and unrealized_pnl are added by the environment
            }
            features_df = pd.DataFrame(features, index=data.index)
            
            # Clean up NaN values - this will primarily drop rows at the beginning 
            # where technical indicators don't have enough data points
            features_df = features_df.dropna()
            
            # Log dropped data information
            rows_dropped = original_length - len(features_df)
            percentage_dropped = (rows_dropped / original_length) * 100
            print(f"Data preprocessing: Dropped {rows_dropped} rows out of {original_length} " 
                  f"({percentage_dropped:.2f}%) due to NaN values")
            
            # Keep the original ATR values (not normalized) for potential position sizing
            atr_aligned = atr_series.loc[features_df.index].values
            
            # Validation
            if len(features_df) < 100:
                raise ValueError("Insufficient data after preprocessing: need at least 100 bars")            
            
            return features_df, atr_aligned

    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return [
            'returns',          # [-0.1, 0.1] Price momentum
            'rsi',             # [-1, 1] Momentum oscillator
            'atr',             # [-1, 1] Volatility indicator
            'volume_change',    # [-1, 1] Volume percentage change
            'volatility_breakout', # [0, 1] Trend with volatility context
            'trend_strength',   # [-1, 1] ADX-based trend quality
            'candle_pattern',   # [-1, 1] Combined price action signal
            'cos_time',        # [-1, 1] Cosine encoding of time
            'sin_time',        # [-1, 1] Sine encoding of time
            'position_type',    # [-1, 0, 1] Current position
            'unrealized_pnl'   # [-1, 1] Current position P&L
        ]
