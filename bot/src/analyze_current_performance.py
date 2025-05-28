"""
Comprehensive diagnostic analysis of current trading system performance.
This script identifies the key weaknesses that need to be addressed in Phase 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trade_model import TradeModel
from trading.environment import TradingEnv
from trading.features import FeatureProcessor
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class PerformanceDiagnostics:
    """Comprehensive analysis of current trading system performance."""
    
    def __init__(self, model_path: str, data_path: str):
        """Initialize diagnostics with model and data."""
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.env = None
        self.results = {}
        
    def load_data_and_model(self):
        """Load the trading data and model."""
        print("Loading data and model...")
        
        # Load data
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data.set_index('time', inplace=True)
            print(f"Data loaded: {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        else:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Load model
        if os.path.exists(self.model_path):
            self.model = TradeModel(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
    def analyze_current_features(self):
        """Analyze the effectiveness of current features."""
        print("\n=== FEATURE ANALYSIS ===")
        
        # Create environment to get features
        self.env = TradingEnv(self.data.tail(1000), initial_balance=10000)
        
        # Get feature names and sample data
        feature_names = self.env.feature_processor.get_feature_names()
        features_df = self.env.raw_data
        
        print(f"Current feature count: {len(features_df.columns)}")
        print(f"Feature names: {list(features_df.columns)}")
        
        # Feature statistics
        print("\n--- Feature Statistics ---")
        feature_stats = features_df.describe()
        for col in features_df.columns:
            values = features_df[col]
            print(f"{col:18s}: min={values.min():6.3f}, max={values.max():6.3f}, "
                  f"mean={values.mean():6.3f}, std={values.std():6.3f}")
            
        # Feature correlation analysis
        print("\n--- Feature Correlations ---")
        correlation_matrix = features_df.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = abs(correlation_matrix.iloc[i, j])
                if corr > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))
        
        if high_corr_pairs:
            print("Highly correlated features (>0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print("No highly correlated features found.")
            
        # Feature range compliance
        print("\n--- Feature Range Validation ---")
        for col in features_df.columns:
            values = features_df[col]
            if col == 'volatility_breakout':
                out_of_range = (values < 0).sum() + (values > 1).sum()
                if out_of_range > 0:
                    print(f"⚠️  {col}: {out_of_range} values outside [0,1] range")
                else:
                    print(f"✓  {col}: All values in [0,1] range")
            elif col not in ['returns']:
                out_of_range = (values < -1).sum() + (values > 1).sum()
                if out_of_range > 0:
                    print(f"⚠️  {col}: {out_of_range} values outside [-1,1] range")
                else:
                    print(f"✓  {col}: All values in [-1,1] range")
                    
        self.results['feature_analysis'] = {
            'feature_count': len(features_df.columns),
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_corr_pairs,
            'feature_stats': feature_stats
        }
        
    def analyze_trade_patterns(self):
        """Analyze trade patterns to identify weaknesses."""
        print("\n=== TRADE PATTERN ANALYSIS ===")
        
        # Run evaluation to get trade data
        results = self.model.evaluate(self.data.tail(2000), initial_balance=10000)
        
        print(f"Total trades analyzed: {len(results.get('trades', []))}")
        
        if not results.get('trades'):
            print("No trades found in evaluation results.")
            return
            
        trades_df = pd.DataFrame(results['trades'])
        
        # Basic trade statistics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        print(f"\n--- Trade Distribution ---")
        print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
        print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
        
        if len(winning_trades) > 0:
            print(f"Average win: ${winning_trades['pnl'].mean():.2f}")
            print(f"Max win: ${winning_trades['pnl'].max():.2f}")
            
        if len(losing_trades) > 0:
            print(f"Average loss: ${losing_trades['pnl'].mean():.2f}")
            print(f"Max loss: ${losing_trades['pnl'].min():.2f}")
            
        # Hold time analysis
        if 'duration' in trades_df.columns:
            print(f"\n--- Hold Time Analysis ---")
            print(f"Average hold time: {trades_df['duration'].mean():.1f} bars")
            print(f"Winners avg hold: {winning_trades['duration'].mean():.1f} bars")
            print(f"Losers avg hold: {losing_trades['duration'].mean():.1f} bars")
            
        # Directional analysis
        if 'direction' in trades_df.columns:
            print(f"\n--- Directional Analysis ---")
            long_trades = trades_df[trades_df['direction'] == 1]
            short_trades = trades_df[trades_df['direction'] == 2]
            
            if len(long_trades) > 0:
                long_wins = long_trades[long_trades['pnl'] > 0]
                print(f"Long trades: {len(long_trades)}, Win rate: {len(long_wins)/len(long_trades)*100:.1f}%")
                
            if len(short_trades) > 0:
                short_wins = short_trades[short_trades['pnl'] > 0]
                print(f"Short trades: {len(short_trades)}, Win rate: {len(short_wins)/len(short_trades)*100:.1f}%")
                
        # Time-of-day analysis
        if 'entry_time' in trades_df.columns:
            print(f"\n--- Time-of-Day Analysis ---")
            trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
            hourly_performance = trades_df.groupby('hour').agg({
                'pnl': ['count', 'mean', 'sum']
            }).round(2)
            
            best_hours = hourly_performance.sort_values(('pnl', 'mean'), ascending=False).head(3)
            worst_hours = hourly_performance.sort_values(('pnl', 'mean'), ascending=True).head(3)
            
            print("Best performing hours:")
            print(best_hours)
            print("\nWorst performing hours:")
            print(worst_hours)
            
        self.results['trade_patterns'] = {
            'total_trades': len(trades_df),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'trades_df': trades_df
        }
        
    def analyze_market_conditions(self):
        """Analyze performance across different market conditions."""
        print("\n=== MARKET CONDITIONS ANALYSIS ===")
        
        # Calculate market regime indicators
        features_df = self.env.raw_data
        
        # Volatility regimes
        atr_values = features_df['atr']
        high_vol_threshold = atr_values.quantile(0.7)
        low_vol_threshold = atr_values.quantile(0.3)
        
        high_vol_periods = atr_values > high_vol_threshold
        low_vol_periods = atr_values < low_vol_threshold
        
        print(f"High volatility periods: {high_vol_periods.sum()} bars ({high_vol_periods.mean()*100:.1f}%)")
        print(f"Low volatility periods: {low_vol_periods.sum()} bars ({low_vol_periods.mean()*100:.1f}%)")
        
        # Trend vs ranging markets
        trend_strength = features_df['trend_strength']
        strong_trend = abs(trend_strength) > 0.5
        ranging_market = abs(trend_strength) <= 0.2
        
        print(f"Strong trend periods: {strong_trend.sum()} bars ({strong_trend.mean()*100:.1f}%)")
        print(f"Ranging market periods: {ranging_market.sum()} bars ({ranging_market.mean()*100:.1f}%)")
        
        # Session analysis (for XAU/USD)
        hours = pd.to_datetime(features_df.index).hour
        
        # Define trading sessions (UTC times for XAU/USD)
        asian_session = ((hours >= 23) | (hours <= 8))  # 23:00-08:00 UTC
        london_session = ((hours >= 8) & (hours <= 16))  # 08:00-16:00 UTC
        ny_session = ((hours >= 13) & (hours <= 21))     # 13:00-21:00 UTC
        overlap_session = ((hours >= 13) & (hours <= 16))  # London/NY overlap
        
        print(f"\n--- Session Distribution ---")
        print(f"Asian session: {asian_session.sum()} bars ({asian_session.mean()*100:.1f}%)")
        print(f"London session: {london_session.sum()} bars ({london_session.mean()*100:.1f}%)")
        print(f"NY session: {ny_session.sum()} bars ({ny_session.mean()*100:.1f}%)")
        print(f"London/NY overlap: {overlap_session.sum()} bars ({overlap_session.mean()*100:.1f}%)")
        
        self.results['market_conditions'] = {
            'volatility_regimes': {
                'high_vol_periods': high_vol_periods.sum(),
                'low_vol_periods': low_vol_periods.sum()
            },
            'trend_analysis': {
                'strong_trend_periods': strong_trend.sum(),
                'ranging_periods': ranging_market.sum()
            },
            'session_analysis': {
                'asian': asian_session.sum(),
                'london': london_session.sum(),
                'ny': ny_session.sum(),
                'overlap': overlap_session.sum()
            }
        }
        
    def identify_improvement_opportunities(self):
        """Identify specific areas for improvement."""
        print("\n=== IMPROVEMENT OPPORTUNITIES ===")
        
        # Feature improvement opportunities
        print("\n--- Feature Engineering Priorities ---")
        
        current_features = list(self.env.raw_data.columns)
        missing_critical_features = []
        
        # Check for missing critical features
        critical_features = [
            'macd', 'stochastic', 'williams_r', 'vwap_relative',
            'support_resistance', 'session_indicator', 'psychological_levels',
            'multi_timeframe_trend', 'volume_profile'
        ]
        
        for feature in critical_features:
            if not any(feature in existing for existing in current_features):
                missing_critical_features.append(feature)
                
        print(f"Missing critical features: {missing_critical_features}")
        
        # Performance improvement priorities
        print("\n--- Performance Improvement Priorities ---")
        
        trade_data = self.results.get('trade_patterns', {})
        win_rate = trade_data.get('win_rate', 0)
        avg_win = trade_data.get('avg_win', 0)
        avg_loss = trade_data.get('avg_loss', 0)
        
        if win_rate < 0.5:
            print(f"🔴 CRITICAL: Win rate too low ({win_rate*100:.1f}%) - Need better entry signals")
            
        if avg_win > 0 and avg_loss < 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            if win_loss_ratio < 1.5:
                print(f"🔴 CRITICAL: Poor win/loss ratio ({win_loss_ratio:.2f}) - Need better risk management")
                
        print("\n--- Recommended Action Plan ---")
        print("1. 🎯 Add multi-timeframe analysis features")
        print("2. 🎯 Implement VWAP and volume profile analysis")
        print("3. 🎯 Add session-specific indicators for XAU/USD")
        print("4. 🎯 Implement psychological level proximity features")
        print("5. 🎯 Add advanced technical indicators (MACD, Stochastic)")
        print("6. 🔧 Redesign reward function for quality over quantity")
        print("7. 🔧 Add risk-adjusted position sizing")
        print("8. 🔧 Implement drawdown-aware training")
        
    def generate_diagnostic_report(self):
        """Generate a comprehensive diagnostic report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE DIAGNOSTIC REPORT")
        print("="*60)
        
        # Summary of current state
        feature_count = self.results.get('feature_analysis', {}).get('feature_count', 0)
        trade_count = self.results.get('trade_patterns', {}).get('total_trades', 0)
        win_rate = self.results.get('trade_patterns', {}).get('win_rate', 0)
        
        print(f"\n📊 CURRENT SYSTEM STATUS:")
        print(f"   Features: {feature_count} (Target: 25-35)")
        print(f"   Total Trades: {trade_count}")
        print(f"   Win Rate: {win_rate*100:.1f}% (Target: 55%+)")
        
        # Key weaknesses identified
        print(f"\n🚨 KEY WEAKNESSES IDENTIFIED:")
        print(f"   1. Limited feature set - only {feature_count} basic features")
        print(f"   2. No multi-timeframe analysis")
        print(f"   3. No XAU/USD-specific features")
        print(f"   4. Basic reward function encouraging quantity over quality")
        print(f"   5. No advanced risk management")
        
        # Phase 1 priorities
        print(f"\n🎯 PHASE 1 PRIORITIES:")
        print(f"   1. Add 15+ new features (targeting 25+ total)")
        print(f"   2. Implement multi-timeframe analysis")
        print(f"   3. Add XAU/USD session-specific features")
        print(f"   4. Include volume profile and VWAP analysis")
        print(f"   5. Add psychological level proximity")
        
        print(f"\n✅ NEXT STEPS:")
        print(f"   → Begin implementing advanced features in features.py")
        print(f"   → Test each feature for effectiveness")
        print(f"   → Prepare for reward function redesign")
        
        return self.results

def main():
    """Run comprehensive diagnostic analysis."""
    
    # Configuration
    model_path = "../model/XAUUSDm.zip"
    data_path = "../data/XAUUSDm_15min.csv"
    
    try:
        # Initialize diagnostics
        diagnostics = PerformanceDiagnostics(model_path, data_path)
        
        # Run analysis
        diagnostics.load_data_and_model()
        diagnostics.analyze_current_features()
        diagnostics.analyze_trade_patterns()
        diagnostics.analyze_market_conditions()
        diagnostics.identify_improvement_opportunities()
        
        # Generate final report
        results = diagnostics.generate_diagnostic_report()
        
        print(f"\n🎉 Diagnostic analysis complete!")
        print(f"📁 Results saved in diagnostics object")
        print(f"📋 Ready to begin Phase 1 implementation")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
