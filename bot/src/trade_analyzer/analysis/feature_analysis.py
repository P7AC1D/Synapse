"""
Feature analysis module for analyzing feature importance and patterns.

This module provides functionality to analyze feature distributions,
correlations, and importance in trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureAnalyzer:
    """Analyzes feature patterns and importance in trading decisions."""
    
    def __init__(self):
        """Initialize the feature analyzer."""
        pass
        
    def extract_feature_columns(self, df: pd.DataFrame, prefix: str = 'feature_') -> List[str]:
        """
        Extract column names that contain feature data.
        
        Args:
            df: DataFrame containing feature columns
            prefix: Prefix that identifies feature columns
            
        Returns:
            List of feature column names
        """
        return [col for col in df.columns if col.startswith(prefix)]
        
    def compare_feature_distributions(self, 
                                     backtest_df: pd.DataFrame, 
                                     live_df: pd.DataFrame,
                                     feature_prefix: str = 'feature_',
                                     profitable_only: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compare feature distributions between backtest and live trading.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            feature_prefix: Prefix that identifies feature columns
            profitable_only: Whether to only consider profitable trades
            
        Returns:
            Dictionary containing statistical comparison results
        """
        comparison_results = {}
        
        # Filter for profitable trades if requested
        if profitable_only and 'is_profitable' in backtest_df.columns:
            backtest_df = backtest_df[backtest_df['is_profitable']]
        if profitable_only and 'is_profitable' in live_df.columns:
            live_df = live_df[live_df['is_profitable']]
            
        # Get feature columns
        backtest_features = self.extract_feature_columns(backtest_df, feature_prefix)
        live_features = self.extract_feature_columns(live_df, feature_prefix)
        
        # Find common features
        common_features = list(set(backtest_features).intersection(set(live_features)))
        
        # Compare distributions for each feature
        for feature in common_features:
            # Get values, dropping NaN
            backtest_values = backtest_df[feature].dropna().values
            live_values = live_df[feature].dropna().values
            
            # Skip if not enough data
            if len(backtest_values) < 5 or len(live_values) < 5:
                continue
                
            # Calculate basic statistics
            backtest_mean = np.mean(backtest_values)
            live_mean = np.mean(live_values)
            backtest_std = np.std(backtest_values)
            live_std = np.std(live_values)
            
            # Perform statistical tests
            try:
                # Kolmogorov-Smirnov test for distribution similarity
                ks_stat, ks_pvalue = stats.ks_2samp(backtest_values, live_values)
                
                # T-test for mean comparison
                t_stat, t_pvalue = stats.ttest_ind(backtest_values, live_values, equal_var=False)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((backtest_std**2 + live_std**2) / 2)
                effect_size = abs(backtest_mean - live_mean) / pooled_std if pooled_std > 0 else 0
                
                # Store results
                comparison_results[feature] = {
                    'backtest_mean': backtest_mean,
                    'live_mean': live_mean,
                    'backtest_std': backtest_std,
                    'live_std': live_std,
                    'ks_stat': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    't_stat': t_stat,
                    't_pvalue': t_pvalue,
                    'effect_size': effect_size,
                    'sample_sizes': (len(backtest_values), len(live_values))
                }
            except Exception as e:
                print(f"Error comparing distributions for feature {feature}: {e}")
                continue
                
        return comparison_results
    
    def identify_important_features(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Identify features that are most important for profitable trades.
        
        Args:
            trades_df: DataFrame containing trade data
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if trades_df.empty or 'is_profitable' not in trades_df.columns:
            return {}
            
        # Get entry feature columns
        feature_cols = [col for col in trades_df.columns if col.startswith('entry_feature_')]
        
        if not feature_cols:
            return {}
            
        # Create a feature importance dictionary
        feature_importance = {}
        
        # Calculate point-biserial correlation between each feature and profitability
        for feature in feature_cols:
            # Drop NaN values
            valid_data = trades_df[[feature, 'is_profitable']].dropna()
            if len(valid_data) < 5:
                continue
                
            # Calculate correlation
            try:
                # Convert boolean to int
                if valid_data['is_profitable'].dtype == bool:
                    valid_data['is_profitable'] = valid_data['is_profitable'].astype(int)
                    
                corr, pvalue = stats.pointbiserialr(
                    valid_data[feature], 
                    valid_data['is_profitable']
                )
                
                # Store results
                feature_importance[feature] = {
                    'correlation': corr,
                    'pvalue': pvalue,
                    'importance_score': abs(corr) if pvalue < 0.05 else 0
                }
            except Exception as e:
                print(f"Error calculating importance for feature {feature}: {e}")
                continue
                
        return feature_importance
    
    def find_feature_thresholds(self, 
                              trades_df: pd.DataFrame, 
                              feature: str,
                              bins: int = 10) -> Dict[str, Any]:
        """
        Find optimal thresholds for a feature that maximize win rate.
        
        Args:
            trades_df: DataFrame containing trade data
            feature: Feature to analyze
            bins: Number of bins for histogram
            
        Returns:
            Dictionary containing threshold analysis results
        """
        if trades_df.empty or 'is_profitable' not in trades_df.columns or feature not in trades_df.columns:
            return {}
            
        # Filter to valid data
        valid_data = trades_df[[feature, 'is_profitable']].dropna()
        if len(valid_data) < 10:  # Need sufficient data
            return {}
            
        try:
            # Calculate min and max values
            min_val = valid_data[feature].min()
            max_val = valid_data[feature].max()
            
            # Create bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            bin_width = (max_val - min_val) / bins
            
            # Create bins and calculate win rate for each bin
            bin_results = []
            
            for i in range(bins):
                lower = bin_edges[i]
                upper = bin_edges[i + 1]
                
                # Get trades in this bin
                bin_trades = valid_data[(valid_data[feature] >= lower) & 
                                      (valid_data[feature] <= upper)]
                
                if len(bin_trades) > 0:
                    win_rate = bin_trades['is_profitable'].mean() * 100
                    bin_results.append({
                        'lower_bound': lower,
                        'upper_bound': upper,
                        'mid_point': (lower + upper) / 2,
                        'trade_count': len(bin_trades),
                        'win_rate': win_rate,
                        'bin_width': bin_width
                    })
            
            # Analyze above/below thresholds
            thresholds = []
            feature_values = valid_data[feature].values
            
            # Check various percentiles
            for percentile in [10, 25, 50, 75, 90]:
                threshold = np.percentile(feature_values, percentile)
                
                # Trades above and below threshold
                above = valid_data[valid_data[feature] > threshold]
                below = valid_data[valid_data[feature] <= threshold]
                
                # Calculate win rates
                if len(above) > 0 and len(below) > 0:
                    above_win_rate = above['is_profitable'].mean() * 100
                    below_win_rate = below['is_profitable'].mean() * 100
                    
                    thresholds.append({
                        'percentile': percentile,
                        'threshold': threshold,
                        'above_count': len(above),
                        'below_count': len(below),
                        'above_win_rate': above_win_rate,
                        'below_win_rate': below_win_rate,
                        'win_rate_difference': above_win_rate - below_win_rate
                    })
            
            return {
                'feature': feature,
                'bin_results': bin_results,
                'threshold_analysis': thresholds,
                'optimal_threshold': self._find_optimal_threshold(valid_data, feature)
            }
            
        except Exception as e:
            print(f"Error analyzing thresholds for feature {feature}: {e}")
            return {}
    
    def _find_optimal_threshold(self, data: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """
        Find the optimal threshold that maximizes win rate difference.
        
        Args:
            data: DataFrame containing feature and is_profitable columns
            feature: Feature to analyze
            
        Returns:
            Dictionary containing optimal threshold information
        """
        # Sort by feature value
        sorted_data = data.sort_values(feature)
        
        best_threshold = None
        best_difference = 0
        best_metrics = {}
        
        # Try each value as a threshold
        feature_values = sorted_data[feature].unique()
        
        for threshold in feature_values:
            above = sorted_data[sorted_data[feature] > threshold]
            below = sorted_data[sorted_data[feature] <= threshold]
            
            # Need at least 3 samples in each group
            if len(above) < 3 or len(below) < 3:
                continue
                
            above_win_rate = above['is_profitable'].mean() * 100
            below_win_rate = below['is_profitable'].mean() * 100
            difference = abs(above_win_rate - below_win_rate)
            
            if difference > best_difference:
                best_difference = difference
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'above_count': len(above),
                    'below_count': len(below),
                    'above_win_rate': above_win_rate,
                    'below_win_rate': below_win_rate,
                    'win_rate_difference': difference
                }
        
        return best_metrics
    
    def analyze_feature_correlation(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlation between features.
        
        Args:
            trades_df: DataFrame containing trade data
            
        Returns:
            DataFrame containing feature correlation matrix
        """
        # Extract feature columns
        feature_cols = [col for col in trades_df.columns if col.startswith('entry_feature_')]
        
        if not feature_cols or len(feature_cols) < 2:
            return pd.DataFrame()
            
        # Calculate correlation matrix
        try:
            corr_matrix = trades_df[feature_cols].corr()
            return corr_matrix
        except Exception as e:
            print(f"Error calculating feature correlation: {e}")
            return pd.DataFrame()
    
    def plot_feature_distribution(self, 
                                backtest_df: pd.DataFrame,
                                live_df: pd.DataFrame,
                                feature: str,
                                bins: int = 20,
                                figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot feature distribution comparison between backtest and live trades.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            feature: Feature to plot
            bins: Number of bins for histogram
            figsize: Figure size (width, height)
        """
        # Check if feature exists in both DataFrames
        if feature not in backtest_df.columns or feature not in live_df.columns:
            print(f"Feature {feature} not found in both datasets")
            return
            
        # Get feature values, dropping NaN
        backtest_values = backtest_df[feature].dropna()
        live_values = live_df[feature].dropna()
        
        # Skip if not enough data
        if len(backtest_values) < 5 or len(live_values) < 5:
            print(f"Not enough data for feature {feature}")
            return
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Calculate optimal bins using Freedman-Diaconis rule for each dataset
        def calc_bins(x):
            if len(x) < 2:
                return bins
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            if iqr == 0:
                return bins
            bin_width = 2 * iqr / (len(x) ** (1/3))
            if bin_width == 0:
                return bins
            return int(np.ceil((max(x) - min(x)) / bin_width))
        
        backtest_bins = min(calc_bins(backtest_values), 50)  # Limit bins
        live_bins = min(calc_bins(live_values), 50)
        
        # Use the smaller number of bins for both
        hist_bins = min(backtest_bins, live_bins)
        if hist_bins < 5:
            hist_bins = bins  # Fallback to default
        
        # Plot distributions
        sns.histplot(backtest_values, bins=hist_bins, alpha=0.5, 
                   label=f'Backtest (n={len(backtest_values)})', kde=True)
        sns.histplot(live_values, bins=hist_bins, alpha=0.5, 
                   label=f'Live (n={len(live_values)})', kde=True)
        
        # Calculate means
        backtest_mean = backtest_values.mean()
        live_mean = live_values.mean()
        
        # Add mean lines
        plt.axvline(backtest_mean, color='blue', linestyle='--', 
                  label=f'Backtest Mean: {backtest_mean:.4f}')
        plt.axvline(live_mean, color='orange', linestyle='--', 
                  label=f'Live Mean: {live_mean:.4f}')
        
        # Statistical test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(backtest_values, live_values)
            t_stat, t_pvalue = stats.ttest_ind(backtest_values, live_values, equal_var=False)
        except:
            ks_stat, ks_pvalue, t_stat, t_pvalue = 0, 1, 0, 1
            
        # Title and labels
        plt.title(f'Distribution Comparison for {feature}\n' +
                f'KS Test: p={ks_pvalue:.4f} | T-Test: p={t_pvalue:.4f}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.show()