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
        return pd.Series(tr).rolling(self.atr_period).mean().values

    def _calculate_rsi(self, close: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index.
        
        Args:
            close: Close prices
            
        Returns:
            RSI values
        """
        delta = pd.Series(close).diff().fillna(0).values
        gain = pd.Series(np.where(delta > 0, delta, 0)).rolling(window=self.rsi_period).mean().values
        loss = pd.Series(np.where(delta < 0, -delta, 0)).rolling(window=self.rsi_period).mean().values
        
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
        boll_std = pd.Series(close).rolling(self.boll_period).std().fillna(0).values
        ma20 = pd.Series(close).rolling(self.boll_period).mean().fillna(close[0]).values
        upper_band = ma20 + (boll_std * 2)
        lower_band = ma20 - (boll_std * 2)
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
            
            # Store features
            features_df['returns'] = returns
            features_df['rsi'] = rsi / 50 - 1  # Normalize to [-1, 1]
            features_df['atr'] = 2 * (atr / close - np.nanmin(atr / close)) / \
                              (np.nanmax(atr / close) - np.nanmin(atr / close) + 1e-8) - 1
            features_df['volatility_breakout'] = volatility_breakout
            features_df['trend_strength'] = trend_strength
            features_df['candle_pattern'] = candle_pattern
            features_df['sin_time'] = sin_time
            features_df['cos_time'] = cos_time
            
            # Clean up features
            features_df = features_df.dropna()
            features_df = features_df.iloc[self.lookback:]
            
            if len(features_df) < 100:
                raise ValueError("Insufficient data after preprocessing: need at least 100 bars")
            
            return features_df, atr[features_df.index]

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
