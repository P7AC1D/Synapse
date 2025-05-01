"""
Main module for the trade analyzer.

This module provides a simple interface for analyzing and comparing
backtest and live trading data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .data_loading.trade_loader import TradeLoader
from .data_loading.trade_aggregator import TradeAggregator
from .analysis.feature_analysis import FeatureAnalyzer
from .visualization.comparison_plots import ComparisonPlotter


class TradeAnalyzer:
    """Main class for analyzing and comparing trade data."""
    
    def __init__(self):
        """Initialize the trade analyzer with its component modules."""
        self.trade_loader = TradeLoader()
        self.trade_aggregator = TradeAggregator()
        self.feature_analyzer = FeatureAnalyzer()
        self.comparison_plotter = ComparisonPlotter()
        
        # Data containers
        self.backtest_raw_data = {}
        self.live_raw_data = {}
        self.backtest_trades = None
        self.live_trades = None
        
    def load_data(self, backtest_path: str, live_path: str) -> None:
        """
        Load trade data from backtest and live sources.
        
        Args:
            backtest_path: Path to backtest trade logs
            live_path: Path to live trade logs
        """
        print(f"Loading backtest data from: {backtest_path}")
        self.backtest_raw_data = self.trade_loader.load_trade_logs(backtest_path, source_type="backtest")
        
        print(f"Loading live data from: {live_path}")
        self.live_raw_data = self.trade_loader.load_trade_logs(live_path, source_type="live")
        
        # Process raw data
        backtest_processed = self.trade_loader.process_trade_data(self.backtest_raw_data)
        live_processed = self.trade_loader.process_trade_data(self.live_raw_data)
        
        # Reconstruct complete trades
        self.backtest_trades = self.trade_aggregator.reconstruct_trades(
            backtest_processed.get('entries', pd.DataFrame()),
            backtest_processed.get('updates', pd.DataFrame()),
            backtest_processed.get('exits', pd.DataFrame())
        )
        
        self.live_trades = self.trade_aggregator.reconstruct_trades(
            live_processed.get('entries', pd.DataFrame()),
            live_processed.get('updates', pd.DataFrame()),
            live_processed.get('exits', pd.DataFrame())
        )
        
        # Display basic statistics
        self._print_data_summary()
    
    def _print_data_summary(self) -> None:
        """Print a summary of the loaded data."""
        print("\n=== Data Summary ===")
        
        print(f"\nBacktest Trades: {len(self.backtest_trades) if self.backtest_trades is not None else 0}")
        if self.backtest_trades is not None and not self.backtest_trades.empty:
            complete_count = self.backtest_trades['is_complete'].sum()
            print(f" - Complete trades: {complete_count}")
            print(f" - Incomplete trades: {len(self.backtest_trades) - complete_count}")
            
            if 'is_profitable' in self.backtest_trades.columns:
                win_rate = self.backtest_trades['is_profitable'].mean() * 100
                print(f" - Win rate: {win_rate:.2f}%")
        
        print(f"\nLive Trades: {len(self.live_trades) if self.live_trades is not None else 0}")
        if self.live_trades is not None and not self.live_trades.empty:
            complete_count = self.live_trades['is_complete'].sum()
            print(f" - Complete trades: {complete_count}")
            print(f" - Incomplete trades: {len(self.live_trades) - complete_count}")
            
            if 'is_profitable' in self.live_trades.columns:
                win_rate = self.live_trades['is_profitable'].mean() * 100
                print(f" - Win rate: {win_rate:.2f}%")
    
    def analyze_model_behavior(self) -> Dict[str, Any]:
        """
        Analyze model behavior based on trade data.
        
        Returns:
            Dictionary containing model behavior analysis results
        """
        results = {}
        
        # Combine backtest and live data for overall analysis
        combined_trades = pd.concat([
            self.backtest_trades, self.live_trades
        ], ignore_index=True) if (self.backtest_trades is not None and self.live_trades is not None) else (
            self.backtest_trades if self.backtest_trades is not None else self.live_trades
        )
        
        if combined_trades is None or combined_trades.empty:
            print("No trade data available for model behavior analysis")
            return results
            
        # Find important features
        print("\nAnalyzing feature importance...")
        feature_importance = self.feature_analyzer.identify_important_features(combined_trades)
        
        # Sort features by importance score
        sorted_features = sorted(
            [(f, d['importance_score']) for f, d in feature_importance.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Print top features
        print("\n=== Top Important Features ===")
        for feature, score in sorted_features[:10]:  # Top 10
            clean_name = feature.replace('entry_feature_', '')
            print(f"{clean_name}: {score:.4f}")
            
            # Analyze feature thresholds
            threshold_analysis = self.feature_analyzer.find_feature_thresholds(
                combined_trades, feature
            )
            
            if threshold_analysis and 'optimal_threshold' in threshold_analysis:
                optimal = threshold_analysis['optimal_threshold']
                if optimal:
                    print(f"  - Optimal threshold: {optimal.get('threshold', 'N/A'):.4f}")
                    print(f"  - Above win rate: {optimal.get('above_win_rate', 0):.2f}%")
                    print(f"  - Below win rate: {optimal.get('below_win_rate', 0):.2f}%")
                    print(f"  - Win rate difference: {optimal.get('win_rate_difference', 0):.2f}%")
        
        # Store results
        results['feature_importance'] = feature_importance
        
        # Analyze entry patterns
        print("\n=== Entry Patterns Analysis ===")
        
        # Action distribution
        action_counts = combined_trades['action'].value_counts()
        print("\nAction Distribution:")
        for action, count in action_counts.items():
            print(f"{action}: {count} trades ({count/len(combined_trades)*100:.1f}%)")
        
        # Profitable vs Unprofitable trades analysis
        if 'is_profitable' in combined_trades.columns:
            profit_trades = combined_trades[combined_trades['is_profitable']]
            loss_trades = combined_trades[~combined_trades['is_profitable']]
            
            print(f"\nProfitable trades: {len(profit_trades)} ({len(profit_trades)/len(combined_trades)*100:.1f}%)")
            print(f"Unprofitable trades: {len(loss_trades)} ({len(loss_trades)/len(combined_trades)*100:.1f}%)")
            
            # Compare mean feature values for profit vs loss trades
            print("\nFeature comparison for profitable vs unprofitable trades:")
            
            feature_cols = [col for col in combined_trades.columns if col.startswith('entry_feature_')]
            feature_comparison = {}
            
            for col in feature_cols:
                if profit_trades[col].count() > 0 and loss_trades[col].count() > 0:
                    profit_mean = profit_trades[col].mean()
                    loss_mean = loss_trades[col].mean()
                    diff = profit_mean - loss_mean
                    diff_pct = abs(diff / loss_mean * 100) if loss_mean != 0 else float('inf')
                    
                    feature_comparison[col] = {
                        'profit_mean': profit_mean,
                        'loss_mean': loss_mean,
                        'difference': diff,
                        'difference_pct': diff_pct
                    }
            
            # Sort and display top differences
            sorted_diffs = sorted(
                [(f, d['difference_pct']) for f, d in feature_comparison.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for feature, diff_pct in sorted_diffs[:5]:  # Top 5
                data = feature_comparison[feature]
                clean_name = feature.replace('entry_feature_', '')
                print(f"{clean_name}:")
                print(f"  - Profit trades avg: {data['profit_mean']:.4f}")
                print(f"  - Loss trades avg: {data['loss_mean']:.4f}")
                print(f"  - Difference: {data['difference']:.4f} ({diff_pct:.1f}%)")
        
        results['feature_comparison'] = feature_comparison if 'feature_comparison' in locals() else {}
        
        return results
    
    def compare_backtest_vs_live(self) -> Dict[str, Any]:
        """
        Compare backtest and live trading behavior.
        
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        if self.backtest_trades is None or self.live_trades is None:
            print("Both backtest and live trade data are required for comparison")
            return results
            
        if self.backtest_trades.empty or self.live_trades.empty:
            print("Both backtest and live trade data are required for comparison")
            return results
        
        # Compare feature distributions
        print("\nComparing feature distributions...")
        feature_comparison = self.feature_analyzer.compare_feature_distributions(
            self.backtest_trades, self.live_trades
        )
        
        # Sort by effect size
        sorted_features = sorted(
            [(f, d['effect_size']) for f, d in feature_comparison.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Print top differences
        print("\n=== Top Feature Distribution Differences ===")
        for feature, effect_size in sorted_features[:10]:  # Top 10
            data = feature_comparison[feature]
            clean_name = feature.replace('feature_', '')
            
            print(f"{clean_name}:")
            print(f"  - Backtest mean: {data['backtest_mean']:.4f}")
            print(f"  - Live mean: {data['live_mean']:.4f}")
            print(f"  - Effect size: {effect_size:.4f}")
            print(f"  - KS p-value: {data['ks_pvalue']:.4f}")
            print(f"  - T-test p-value: {data['t_pvalue']:.4f}")
            
        results['feature_comparison'] = feature_comparison
        
        # Compare trading metrics
        print("\n=== Trading Metrics Comparison ===")
        
        # Win rate
        backtest_win_rate = self.backtest_trades['is_profitable'].mean() * 100 if 'is_profitable' in self.backtest_trades.columns else 0
        live_win_rate = self.live_trades['is_profitable'].mean() * 100 if 'is_profitable' in self.live_trades.columns else 0
        
        print(f"Win Rate:")
        print(f"  - Backtest: {backtest_win_rate:.2f}%")
        print(f"  - Live: {live_win_rate:.2f}%")
        print(f"  - Difference: {abs(backtest_win_rate - live_win_rate):.2f}%")
        
        # Trade duration
        if 'trade_duration' in self.backtest_trades.columns and 'trade_duration' in self.live_trades.columns:
            backtest_duration = self.backtest_trades['trade_duration'].mean()
            live_duration = self.live_trades['trade_duration'].mean()
            
            print(f"\nTrade Duration (minutes):")
            print(f"  - Backtest mean: {backtest_duration:.2f}")
            print(f"  - Live mean: {live_duration:.2f}")
            print(f"  - Difference: {abs(backtest_duration - live_duration):.2f}")
        
        # Profit points
        if 'final_profit' in self.backtest_trades.columns and 'final_profit' in self.live_trades.columns:
            backtest_profit = self.backtest_trades['final_profit'].mean()
            live_profit = self.live_trades['final_profit'].mean()
            
            print(f"\nAverage Profit Points:")
            print(f"  - Backtest: {backtest_profit:.2f}")
            print(f"  - Live: {live_profit:.2f}")
            print(f"  - Difference: {abs(backtest_profit - live_profit):.2f}")
        
        results['metrics_comparison'] = {
            'win_rate': {'backtest': backtest_win_rate, 'live': live_win_rate},
            'trade_duration': {'backtest': backtest_duration, 'live': live_duration} if 'backtest_duration' in locals() else {},
            'profit': {'backtest': backtest_profit, 'live': live_profit} if 'backtest_profit' in locals() else {}
        }
        
        return results
    
    def visualize_comparisons(self, output_dir: Optional[str] = None) -> None:
        """
        Generate comparative visualizations.
        
        Args:
            output_dir: Directory to save plots, if None, plots are displayed but not saved
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        if self.backtest_trades is None or self.live_trades is None:
            print("Both backtest and live trade data are required for visualizations")
            return
            
        if self.backtest_trades.empty or self.live_trades.empty:
            print("Both backtest and live trade data are required for visualizations")
            return
            
        print("\nGenerating visualizations...")
        
        # Create comprehensive dashboard
        dashboard = self.comparison_plotter.create_dashboard(
            self.backtest_trades, self.live_trades
        )
        
        if output_dir:
            dashboard.savefig(os.path.join(output_dir, 'comparison_dashboard.png'), dpi=150)
            print(f"Saved dashboard to {os.path.join(output_dir, 'comparison_dashboard.png')}")
        else:
            plt.show()
            
        # Generate feature distribution plots for top features
        feature_comparison = self.feature_analyzer.compare_feature_distributions(
            self.backtest_trades, self.live_trades
        )
        
        # Sort by effect size
        sorted_features = sorted(
            [(f, d['effect_size']) for f, d in feature_comparison.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Plot top differences
        for i, (feature, _) in enumerate(sorted_features[:5]):  # Top 5
            print(f"Generating plot for feature: {feature}")
            
            # Generate plot
            plt.figure()
            self.feature_analyzer.plot_feature_distribution(
                self.backtest_trades, self.live_trades, feature
            )
            
            if output_dir:
                clean_name = feature.replace('feature_', '').replace('_', '-')
                plt.savefig(os.path.join(output_dir, f'feature_{clean_name}.png'), dpi=150)
                print(f"Saved plot to {os.path.join(output_dir, f'feature_{clean_name}.png')}")
            else:
                plt.show()
    
    def generate_report(self, output_path: str) -> None:
        """
        Generate comprehensive HTML report.
        
        Args:
            output_path: Path to save HTML report
        """
        # This is a placeholder for future implementation
        # In a complete implementation, this would generate an HTML report
        # with all the analysis results and visualizations
        print(f"Report generation not implemented yet. Will save to {output_path}")


# Command-line interface
def main():
    """Command-line interface for the trade analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and compare backtest and live trading data")
    parser.add_argument('--backtest', type=str, required=True,
                      help='Path to backtest trade logs')
    parser.add_argument('--live', type=str, required=True,
                      help='Path to live trade logs')
    parser.add_argument('--output', type=str, default=None,
                      help='Directory to save output files')
    
    args = parser.parse_args()
    
    analyzer = TradeAnalyzer()
    analyzer.load_data(args.backtest, args.live)
    analyzer.analyze_model_behavior()
    analyzer.compare_backtest_vs_live()
    analyzer.visualize_comparisons(args.output)
    
    if args.output:
        report_path = os.path.join(args.output, 'analysis_report.html')
        analyzer.generate_report(report_path)


if __name__ == "__main__":
    main()