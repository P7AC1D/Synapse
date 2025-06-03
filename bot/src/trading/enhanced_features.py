"""Enhanced feature calculation with advanced features for trading environment."""
import numpy as np
import pandas as pd
import ta
from typing import Tuple, Dict, Any
from gymnasium import spaces
from .fixed_advanced_features import FixedAdvancedFeatureCalculator as AdvancedFeatureCalculator

class EnhancedFeatureProcessor:
    """Enhanced feature processor with 28+ advanced features for professional day trading."""
    
    def __init__(self, use_advanced_features: bool = True):
        """Initialize enhanced feature processor with advanced capabilities."""
        self.atr_period = 14
        self.rsi_period = 14
        self.boll_period = 20
        self.lookback = max(self.boll_period, self.atr_period)
        
        # Advanced features
        self.use_advanced_features = use_advanced_features
        if self.use_advanced_features:
            self.advanced_calculator = AdvancedFeatureCalculator()

    def setup_observation_space(self, feature_count: int = 37) -> spaces.Box:
        """Setup observation space with proper feature bounds.
        
        Args:
            feature_count: Number of features in observation space (37 with advanced features)

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
        ).average_true_range()
        atr = atr.bfill().fillna(atr.mean()).values
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(
            close=close_s,
            window=self.rsi_period
        ).rsi()
        rsi = rsi.bfill().fillna(50).values  # Default to neutral RSI
        
        # Calculate Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close=close_s,
            window=self.boll_period,
            window_dev=2
        )
        upper = bb.bollinger_hband().bfill().fillna(close_s.iloc[0]).values
        lower = bb.bollinger_lband().bfill().fillna(close_s.iloc[0]).values
        
        # Calculate ADX for trend strength
        adx = ta.trend.ADXIndicator(
            high=high_s, low=low_s, close=close_s,
            window=self.atr_period
        ).adx()
        adx = adx.bfill().fillna(0).values
        trend_strength = np.clip(adx/25 - 1, -1, 1)  # Same normalization as before
        
        return atr, rsi, (upper, lower), trend_strength

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Preprocess market data and calculate all features (basic + advanced).
        
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
            
            # Calculate technical indicators using TA-Lib
            atr, rsi, (upper_band, lower_band), trend_strength = self._calculate_indicators(high, low, close)
            
            # Store ATR in DataFrame immediately for proper alignment
            atr_df = pd.DataFrame({'atr': atr}, index=data.index)
            
            # Calculate ATR normalization
            window_size = 20
            atr_sma = pd.Series(atr).rolling(window_size, min_periods=1).mean().values
            atr_ratio = atr / (atr_sma + 1e-8)
            
            # Scale ATR ratio
            min_expected_ratio = 0.5
            max_expected_ratio = 2.0
            expected_range = max_expected_ratio - min_expected_ratio
            atr_norm = 2 * (atr_ratio - min_expected_ratio) / expected_range - 1
            atr_norm = np.clip(atr_norm, -1, 1)
            
            # Calculate basic features
            returns = np.zeros_like(close)
            returns[1:] = np.diff(close) / close[:-1]
            returns = np.clip(returns, -0.1, 0.1)
            
            minutes_in_day = 24 * 60
            time_index = pd.to_datetime(data.index).hour * 60 + pd.to_datetime(data.index).minute
            sin_time = np.sin(2 * np.pi * time_index / minutes_in_day)
            cos_time = np.cos(2 * np.pi * time_index / minutes_in_day)
            
            # Price action features
            body = close - opens
            upper_wick = high - np.maximum(close, opens)
            lower_wick = np.minimum(close, opens) - low
            range_ = high - low + 1e-8
            candle_pattern = (body/range_ + (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2
            candle_pattern = np.clip(candle_pattern, -1, 1)
            
            # Volatility breakout (but remove this since it's 88% correlated with RSI)
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
            volume_pct = np.clip(volume_pct, -1, 1)
            
            # Create basic features DataFrame
            basic_features = {
                'returns': returns,
                'rsi': rsi / 50 - 1,
                'atr': atr_norm,
                'trend_strength': trend_strength,
                'candle_pattern': candle_pattern,
                'sin_time': sin_time,
                'cos_time': cos_time,
                'volume_change': volume_pct
            }
            
            # Note: Removing volatility_breakout due to high correlation with RSI (88.4%)
            # This reduces redundancy and improves model efficiency
            
            features_df = pd.DataFrame(basic_features, index=data.index)
            
            # Add advanced features if enabled
            if self.use_advanced_features:
                advanced_features_df = self.advanced_calculator.calculate_all_advanced_features(data)
                
                # Combine basic and advanced features
                # Ensure both DataFrames have the same index for proper alignment
                common_index = features_df.index.intersection(advanced_features_df.index)
                features_df = features_df.loc[common_index]
                advanced_features_df = advanced_features_df.loc[common_index]
                
                # Concatenate features
                features_df = pd.concat([features_df, advanced_features_df], axis=1)
            
            # Clean up NaN values
            features_df = features_df.dropna()
            
            # Apply lookback and ensure alignment
            if len(features_df) > self.lookback:
                features_df = features_df.iloc[self.lookback:]
                # Ensure exact index alignment between features and ATR
                common_index = features_df.index.intersection(atr_df.index)
                features_df = features_df.loc[common_index]
                atr_df = atr_df.loc[common_index]
            
            # Validation
            if len(features_df) < 100:
                # Adaptive validation - allow smaller datasets for optimization tests
                min_required = max(50, self.lookback + 10)
                if len(features_df) < 30:  # Absolute minimum
                    raise ValueError(f"Insufficient data after preprocessing: need at least {min_required} bars, got {len(features_df)}")
                else:
                    print(f"⚠️ Warning: Limited data ({len(features_df)} bars) - continuing with reduced validation data")
            
            # Convert to array after guaranteed alignment
            atr_aligned = atr_df.values.reshape(-1)
            
            # Double-check alignment (should never fail now)
            if len(atr_aligned) != len(features_df):
                # If lengths still don't match, use minimum length
                min_len = min(len(features_df), len(atr_aligned))
                features_df = features_df.iloc[-min_len:]
                atr_aligned = atr_aligned[-min_len:]
            
            # Validate feature ranges - all features should be in [-1, 1] range
            for col, values in features_df.items():
                if col == 'returns':
                    # Returns can exceed [-1, 1] in extreme market conditions (already clipped to [-0.1, 0.1])
                    continue
                else:
                    # Check for features outside [-1, 1] range with small tolerance for floating-point precision
                    out_of_range_low = (values < -1.01).sum()  # Allow small tolerance
                    out_of_range_high = (values > 1.01).sum()
                    
                    if out_of_range_low > 0 or out_of_range_high > 0:
                        print(f"⚠️ Warning: {col} has {out_of_range_low + out_of_range_high} values outside [-1,1] range")
                        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
                        # Clip to valid range
                        features_df[col] = np.clip(features_df[col], -1, 1)
            
            return features_df, atr_aligned

    def get_feature_names(self) -> list:
        """Get list of all feature names."""
        basic_features = [
            'returns',          # [-0.1, 0.1] Price momentum
            'rsi',             # [-1, 1] Momentum oscillator
            'atr',             # [-1, 1] Volatility indicator
            'trend_strength',   # [-1, 1] ADX-based trend quality
            'candle_pattern',   # [-1, 1] Combined price action signal
            'sin_time',        # [-1, 1] Sine encoding of time
            'cos_time',        # [-1, 1] Cosine encoding of time
            'volume_change',    # [-1, 1] Volume percentage change
            'position_type',    # [-1, 0, 1] Current position
            'unrealized_pnl'   # [-1, 1] Current position P&L
        ]
        
        if self.use_advanced_features:
            advanced_features = [
                # MACD features (5)
                'macd_line', 'macd_signal', 'macd_histogram', 'macd_momentum', 'macd_divergence',
                # Stochastic features (5)
                'stoch_k', 'stoch_d', 'stoch_cross', 'stoch_overbought', 'stoch_oversold',
                # VWAP features (3)
                'vwap_distance', 'vwap_trend', 'vwap_volume_ratio',
                # Session features (5)
                'asian_session', 'london_session', 'ny_session', 'overlap_session', 'session_premium',
                # Psychological level features (2)
                'psych_level_distance', 'psych_level_strength',
                # Multi-timeframe trend features (3)
                'trend_alignment', 'trend_strength_fast', 'trend_strength_medium',
                # Volume profile features (2)
                'volume_profile', 'volume_concentration',
                # Momentum features (3)
                'williams_r', 'roc', 'cci'
            ]
            return basic_features + advanced_features
        else:
            return basic_features

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_feature_names())

# Backward compatibility alias
FeatureProcessor = EnhancedFeatureProcessor
