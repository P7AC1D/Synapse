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
        
        # Calculate ADX
        adx = ta.trend.ADXIndicator(
            high=high_s, low=low_s, close=close_s,
            window=self.atr_period
        ).adx().values
        
        return atr, rsi, (upper, lower), adx

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.Index]:
        """Preprocess market data and calculate features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features DataFrame, ATR values)
        """
        
        with np.errstate(divide='ignore', invalid='ignore'):
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            opens = data['open'].values
            
            # Calculate technical indicators and store in temporary DataFrame
            atr, rsi, (upper_band, lower_band), adx = self._calculate_indicators(high, low, close)
            indicators_df = pd.DataFrame({
                'atr': atr,
                'rsi': rsi,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'adx': adx
            }, index=data.index)
            
            # Log initial row count
            initial_rows = len(indicators_df)
            
            # Clean up NaN values from indicators and log dropped rows
            indicators_df = indicators_df.dropna()
            dropped_rows = initial_rows - len(indicators_df)
            print(f"Dropped {dropped_rows} rows containing NaN values ({(dropped_rows/initial_rows)*100:.2f}% of data)")
            
            # Extract cleaned indicator values
            atr = indicators_df['atr'].values
            rsi = indicators_df['rsi'].values
            upper_band = indicators_df['upper_band'].values
            lower_band = indicators_df['lower_band'].values
            adx = indicators_df['adx'].values
            
            # Get the aligned price data
            aligned_index = indicators_df.index
            close = data.loc[aligned_index, 'close'].values
            high = data.loc[aligned_index, 'high'].values
            low = data.loc[aligned_index, 'low'].values
            opens = data.loc[aligned_index, 'open'].values
            volume = data.loc[aligned_index, 'volume'].values.astype(np.float64)
            
            # Calculate ATR percentage change
            atr_pct = np.zeros_like(atr)
            atr_pct[1:] = np.clip((atr[1:] - atr[:-1]) / (atr[:-1] + 1e-8), -1, 1)
            atr_norm = atr_pct
            
            # Calculate other features
            returns = np.zeros_like(close)
            returns[1:] = np.diff(close) / close[:-1]
            returns = np.clip(returns, -0.1, 0.1)
            minutes_in_day = 24 * 60            # Explicitly use UTC to prevent timezone conversions
            time_index = pd.to_datetime(aligned_index, utc=True).hour * 60 + pd.to_datetime(aligned_index, utc=True).minute
            
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
            
            # Volume change - using aligned volume data
            volume_pct = np.zeros_like(volume, dtype=np.float64)
            volume_pct[1:] = np.divide(
                volume[1:] - volume[:-1],
                volume[:-1],
                out=np.zeros(len(volume)-1, dtype=np.float64),
                where=volume[:-1] != 0
            )
            volume_pct = np.clip(volume_pct, -1, 1)            # Create features DataFrame with exact ordering to match get_feature_names
            features = {
                'returns': returns,
                'rsi': rsi / 50 - 1,
                'atr': atr_norm,
                'volume_change': volume_pct,
                'volatility_breakout': volatility_breakout,
                'trend_strength': np.clip(adx/25 - 1, -1, 1),  # Normalize ADX to [-1, 1] range
                'candle_pattern': candle_pattern,
                'cos_time': cos_time,
                'sin_time': sin_time,
                # position_type and unrealized_pnl are added by the environment
            }
            features_df = pd.DataFrame(features, index=aligned_index)
            
            # Validation
            if len(features_df) < 100:
                raise ValueError("Insufficient data after preprocessing: need at least 100 bars")
            
            return features_df, atr, features_df.index
        
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
