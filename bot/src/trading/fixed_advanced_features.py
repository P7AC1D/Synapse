"""
Fixed advanced feature engineering with corrected Williams %R and other indicators.
"""

import numpy as np
import pandas as pd
import ta
from typing import Tuple, Dict, Any, Optional
from scipy.signal import find_peaks

class FixedAdvancedFeatureCalculator:
    """Fixed advanced feature calculations for professional day trading."""
    
    def __init__(self):
        """Initialize with XAU/USD-specific parameters."""
        # Multi-timeframe periods
        self.short_period = 12
        self.long_period = 26
        self.signal_period = 9
        
        # Session times (UTC)
        self.sessions = {
            'asian': (23, 8),      # 23:00-08:00 UTC
            'london': (8, 16),     # 08:00-16:00 UTC  
            'ny': (13, 21),        # 13:00-21:00 UTC
            'overlap': (13, 16)    # London/NY overlap
        }
        
        # XAU/USD psychological levels (major round numbers)
        self.psychological_levels = [
            1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150,
            2175, 2200, 2225, 2250, 2275, 2300, 2325, 2350, 2375, 2400
        ]
        
    def calculate_macd_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate MACD and related features."""
        close_series = pd.Series(close)
        
        # Standard MACD
        macd_line = ta.trend.MACD(close_series, window_slow=self.long_period, 
                                 window_fast=self.short_period).macd()
        macd_signal = ta.trend.MACD(close_series, window_slow=self.long_period,
                                   window_fast=self.short_period, 
                                   window_sign=self.signal_period).macd_signal()
        macd_histogram = macd_line - macd_signal
        
        # MACD momentum
        macd_momentum = np.diff(macd_line.values, prepend=macd_line.values[0])
        
        # Normalize features
        macd_line_norm = self._normalize_feature(macd_line.values, method='robust')
        macd_signal_norm = self._normalize_feature(macd_signal.values, method='robust')
        macd_histogram_norm = self._normalize_feature(macd_histogram.values, method='robust')
        macd_momentum_norm = self._normalize_feature(macd_momentum, method='robust')
        
        # Simple divergence detection
        macd_divergence = np.zeros_like(close)
        
        return {
            'macd_line': macd_line_norm,
            'macd_signal': macd_signal_norm,
            'macd_histogram': macd_histogram_norm,
            'macd_momentum': macd_momentum_norm,
            'macd_divergence': macd_divergence
        }
    
    def calculate_stochastic_features(self, high: np.ndarray, low: np.ndarray, 
                                    close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Stochastic oscillator and related features."""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # Stochastic %K and %D
        stoch_k = ta.momentum.StochasticOscillator(
            high_series, low_series, close_series, window=14, smooth_window=3
        ).stoch()
        stoch_d = ta.momentum.StochasticOscillator(
            high_series, low_series, close_series, window=14, smooth_window=3
        ).stoch_signal()
        
        # Stochastic features
        stoch_cross = np.where(stoch_k.values > stoch_d.values, 1, -1)
        stoch_overbought = np.where(stoch_k.values > 80, 1, 0)
        stoch_oversold = np.where(stoch_k.values < 20, 1, 0)
        
        # Normalize
        stoch_k_norm = (stoch_k.values - 50) / 50  # Convert to [-1, 1]
        stoch_d_norm = (stoch_d.values - 50) / 50
        stoch_cross_norm = stoch_cross.astype(np.float32)
        
        return {
            'stoch_k': np.clip(stoch_k_norm, -1, 1),
            'stoch_d': np.clip(stoch_d_norm, -1, 1),
            'stoch_cross': stoch_cross_norm,
            'stoch_overbought': stoch_overbought.astype(np.float32),
            'stoch_oversold': stoch_oversold.astype(np.float32)
        }
    
    def calculate_vwap_features(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, volume: np.ndarray,
                               index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Calculate VWAP and related features."""
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Daily VWAP calculation
        df = pd.DataFrame({
            'typical_price': typical_price,
            'volume': volume,
            'close': close
        }, index=index)
        
        # Group by date for daily VWAP
        df['date'] = df.index.date
        df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()
        df['cumulative_pv'] = df.groupby('date').apply(
            lambda x: (x['typical_price'] * x['volume']).cumsum()
        ).values
        
        # Calculate VWAP
        vwap = df['cumulative_pv'] / df['cumulative_volume']
        vwap = vwap.fillna(method='ffill').fillna(close[0])
        
        # VWAP features
        vwap_distance = (close - vwap.values) / close
        vwap_trend = np.diff(vwap.values, prepend=vwap.values[0])
        vwap_volume_ratio = volume / (df.groupby('date')['volume'].transform('mean'))
        
        # Normalize
        vwap_distance_norm = self._normalize_feature(vwap_distance, method='robust')
        vwap_trend_norm = self._normalize_feature(vwap_trend, method='robust')
        vwap_volume_ratio_norm = self._normalize_feature(vwap_volume_ratio, method='robust')
        
        return {
            'vwap_distance': vwap_distance_norm,
            'vwap_trend': vwap_trend_norm,
            'vwap_volume_ratio': vwap_volume_ratio_norm
        }
    
    def calculate_session_features(self, index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Calculate session-based features for XAU/USD."""
        hours = index.hour
        
        # Session indicators
        asian_session = ((hours >= self.sessions['asian'][0]) | 
                        (hours <= self.sessions['asian'][1])).astype(np.float32)
        london_session = ((hours >= self.sessions['london'][0]) & 
                         (hours <= self.sessions['london'][1])).astype(np.float32)
        ny_session = ((hours >= self.sessions['ny'][0]) & 
                     (hours <= self.sessions['ny'][1])).astype(np.float32)
        overlap_session = ((hours >= self.sessions['overlap'][0]) & 
                          (hours <= self.sessions['overlap'][1])).astype(np.float32)
        
        # Session momentum (premium trading times)
        session_premium = overlap_session * 2 + london_session + ny_session
        session_premium = np.clip(session_premium, 0, 3) / 3  # Normalize to [0,1]
        
        # Convert to [-1, 1] range for consistency
        asian_norm = asian_session * 2 - 1
        london_norm = london_session * 2 - 1
        ny_norm = ny_session * 2 - 1
        overlap_norm = overlap_session * 2 - 1
        session_premium_norm = session_premium * 2 - 1
        
        return {
            'asian_session': asian_norm,
            'london_session': london_norm,
            'ny_session': ny_norm,
            'overlap_session': overlap_norm,
            'session_premium': session_premium_norm
        }
    
    def calculate_psychological_levels(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate psychological level proximity features."""
        # Find nearest psychological level for each price
        nearest_level_distance = np.zeros_like(close)
        level_strength = np.zeros_like(close)
        
        for i, price in enumerate(close):
            # Find nearest level
            distances = [abs(price - level) for level in self.psychological_levels]
            min_distance = min(distances)
            nearest_level = self.psychological_levels[distances.index(min_distance)]
            
            # Distance as percentage of price
            nearest_level_distance[i] = (price - nearest_level) / price
            
            # Level strength (stronger for round numbers)
            if nearest_level % 100 == 0:  # $2000, $2100, etc.
                level_strength[i] = 1.0
            elif nearest_level % 50 == 0:  # $2050, $2150, etc.
                level_strength[i] = 0.7
            elif nearest_level % 25 == 0:  # $2025, $2075, etc.
                level_strength[i] = 0.4
            else:
                level_strength[i] = 0.2
        
        # Normalize
        distance_norm = self._normalize_feature(nearest_level_distance, method='robust')
        
        return {
            'psych_level_distance': distance_norm,
            'psych_level_strength': level_strength * 2 - 1  # Convert to [-1, 1]
        }
    
    def calculate_multi_timeframe_trend(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate multi-timeframe trend alignment."""
        close_series = pd.Series(close)
        
        # Different timeframe SMAs (simulated)
        sma_fast = close_series.rolling(20, min_periods=1).mean()  # ~5 hours
        sma_medium = close_series.rolling(80, min_periods=1).mean()  # ~20 hours (daily)
        sma_slow = close_series.rolling(320, min_periods=1).mean()  # ~80 hours (weekly)
        
        # Trend directions
        trend_fast = np.where(close > sma_fast.values, 1, -1)
        trend_medium = np.where(close > sma_medium.values, 1, -1)
        trend_slow = np.where(close > sma_slow.values, 1, -1)
        
        # Trend alignment
        trend_alignment = (trend_fast + trend_medium + trend_slow) / 3
        
        # Trend strength
        trend_strength_fast = (close - sma_fast.values) / sma_fast.values
        trend_strength_medium = (close - sma_medium.values) / sma_medium.values
        
        # Normalize
        trend_strength_fast_norm = self._normalize_feature(trend_strength_fast, method='robust')
        trend_strength_medium_norm = self._normalize_feature(trend_strength_medium, method='robust')
        
        return {
            'trend_alignment': np.clip(trend_alignment, -1, 1),
            'trend_strength_fast': trend_strength_fast_norm,
            'trend_strength_medium': trend_strength_medium_norm
        }
    
    def calculate_momentum_features(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate momentum indicators with fixed implementations."""
        close_series = pd.Series(close)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        
        # Manual Williams %R calculation (to avoid parameter issues)
        williams_r_values = []
        lookback = 14
        for i in range(len(close)):
            if i < lookback:
                williams_r_values.append(-50)  # Neutral value
            else:
                period_high = np.max(high[i-lookback:i+1])
                period_low = np.min(low[i-lookback:i+1])
                if period_high != period_low:
                    wr = ((period_high - close[i]) / (period_high - period_low)) * -100
                else:
                    wr = -50
                williams_r_values.append(wr)
        
        williams_r = np.array(williams_r_values)
        
        # Rate of Change (ROC)
        roc = ta.momentum.ROCIndicator(close_series, window=12).roc()
        
        # Simple CCI calculation
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price)
        sma_tp = tp_series.rolling(20, min_periods=1).mean()
        mean_deviation = tp_series.rolling(20, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        cci = (typical_price - sma_tp.values) / (0.015 * mean_deviation.values)
        cci = np.nan_to_num(cci, nan=0)
        
        # Normalize features
        williams_r_norm = williams_r / 50  # Convert from [-100, 0] to [-2, 0], then clip
        roc_norm = self._normalize_feature(roc.values, method='robust')
        cci_norm = self._normalize_feature(cci, method='robust')
        
        return {
            'williams_r': np.clip(williams_r_norm, -1, 1),
            'roc': roc_norm,
            'cci': cci_norm
        }
    
    def _normalize_feature(self, values: np.ndarray, method: str = 'robust') -> np.ndarray:
        """Normalize feature values to [-1, 1] range."""
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        if method == 'robust':
            # Use robust scaling (median and IQR)
            median = np.median(values)
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:
                scaled = (values - median) / iqr
                return np.clip(scaled, -3, 3) / 3  # Clip to 3 IQRs and normalize
            else:
                return np.zeros_like(values)
        else:
            # Z-score normalization
            mean, std = np.mean(values), np.std(values)
            if std > 0:
                scaled = (values - mean) / std
                return np.clip(scaled, -3, 3) / 3  # Clip to 3 standard deviations
            else:
                return np.zeros_like(values)

    def calculate_all_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced features for the dataset."""
        print("Calculating advanced features...")
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        index = data.index
        
        advanced_features = {}
        
        # 1. MACD features (5 features)
        print("  - MACD features...")
        macd_features = self.calculate_macd_features(close)
        advanced_features.update(macd_features)
        
        # 2. Stochastic features (5 features)
        print("  - Stochastic features...")
        stoch_features = self.calculate_stochastic_features(high, low, close)
        advanced_features.update(stoch_features)
        
        # 3. VWAP features (3 features)
        print("  - VWAP features...")
        vwap_features = self.calculate_vwap_features(high, low, close, volume, index)
        advanced_features.update(vwap_features)
        
        # 4. Session features (5 features)
        print("  - Session features...")
        session_features = self.calculate_session_features(index)
        advanced_features.update(session_features)
        
        # 5. Psychological level features (2 features)
        print("  - Psychological level features...")
        psych_features = self.calculate_psychological_levels(close)
        advanced_features.update(psych_features)
        
        # 6. Multi-timeframe trend features (3 features)
        print("  - Multi-timeframe trend features...")
        trend_features = self.calculate_multi_timeframe_trend(close)
        advanced_features.update(trend_features)
        
        # 7. Momentum features (3 features)
        print("  - Momentum features...")
        momentum_features = self.calculate_momentum_features(high, low, close)
        advanced_features.update(momentum_features)
        
        # Create DataFrame
        advanced_df = pd.DataFrame(advanced_features, index=index)
        
        # Fill any remaining NaN values
        advanced_df = advanced_df.fillna(method='bfill').fillna(0)
        
        print(f"✓ Advanced features calculated: {len(advanced_df.columns)} new features")
        return advanced_df
