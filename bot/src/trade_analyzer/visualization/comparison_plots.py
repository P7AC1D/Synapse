"""
Comparison plots module for visualizing differences between backtest and live trading.

This module provides functions to create side-by-side visualizations comparing
backtest and live trading behavior across various metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


class ComparisonPlotter:
    """Creates comparative visualizations between backtest and live trading data."""
    
    def __init__(self):
        """Initialize the comparison plotter."""
        # Set default style
        sns.set_style("whitegrid")
        self.default_colors = {
            'backtest': '#1f77b4',  # Blue
            'live': '#ff7f0e'       # Orange
        }
    
    def plot_win_rate_comparison(self,
                               backtest_df: pd.DataFrame,
                               live_df: pd.DataFrame,
                               figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot win rate comparison between backtest and live trading.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate win rates
        win_rates = []
        
        if not backtest_df.empty and 'is_profitable' in backtest_df.columns:
            backtest_win_rate = backtest_df['is_profitable'].mean() * 100
            win_rates.append({
                'source': 'Backtest',
                'win_rate': backtest_win_rate,
                'count': len(backtest_df)
            })
            
        if not live_df.empty and 'is_profitable' in live_df.columns:
            live_win_rate = live_df['is_profitable'].mean() * 100
            win_rates.append({
                'source': 'Live',
                'win_rate': live_win_rate,
                'count': len(live_df)
            })
            
        if not win_rates:
            plt.title("No win rate data available")
            return fig
            
        # Create plot
        df = pd.DataFrame(win_rates)
        sns.barplot(x='source', y='win_rate', data=df, palette=[
            self.default_colors['backtest'],
            self.default_colors['live']
        ])
        
        # Add value labels on top of bars
        for i, row in enumerate(win_rates):
            ax.text(i, row['win_rate'] + 1, f"{row['win_rate']:.1f}%\n(n={row['count']})",
                  ha='center', va='bottom')
        
        # Add labels and title
        plt.title('Win Rate Comparison')
        plt.ylabel('Win Rate (%)')
        plt.xlabel('')
        
        # Y axis starts at 0
        plt.ylim(0, max([r['win_rate'] for r in win_rates]) * 1.15)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_hold_time_comparison(self,
                                backtest_df: pd.DataFrame,
                                live_df: pd.DataFrame,
                                figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot hold time comparison between backtest and live trading.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Check for hold time column
        if 'trade_duration' not in backtest_df.columns or 'trade_duration' not in live_df.columns:
            plt.title("Hold time data not available")
            return fig
            
        # Create boxplot data
        backtest_hold = backtest_df['trade_duration'].dropna()
        live_hold = live_df['trade_duration'].dropna()
        
        data = []
        if not backtest_hold.empty:
            for val in backtest_hold:
                data.append({'source': 'Backtest', 'duration': val})
                
        if not live_hold.empty:
            for val in live_hold:
                data.append({'source': 'Live', 'duration': val})
        
        if not data:
            plt.title("No hold time data available")
            return fig
            
        # Create plot
        df = pd.DataFrame(data)
        sns.boxplot(x='source', y='duration', data=df, palette=[
            self.default_colors['backtest'],
            self.default_colors['live']
        ])
        
        # Add labels and title
        plt.title('Trade Duration Comparison')
        plt.ylabel('Duration (minutes)')
        plt.xlabel('')
        
        # Add statistics as text
        stats_text = []
        
        if not backtest_hold.empty:
            stats_text.append(f"Backtest (n={len(backtest_hold)}):\n" + 
                            f"Mean: {backtest_hold.mean():.1f} min\n" +
                            f"Median: {backtest_hold.median():.1f} min")
        
        if not live_hold.empty:
            stats_text.append(f"Live (n={len(live_hold)}):\n" + 
                            f"Mean: {live_hold.mean():.1f} min\n" +
                            f"Median: {live_hold.median():.1f} min")
        
        plt.figtext(0.95, 0.5, '\n\n'.join(stats_text),
                  ha='right', va='center', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    def plot_profit_distribution_comparison(self,
                                          backtest_df: pd.DataFrame,
                                          live_df: pd.DataFrame,
                                          figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot profit distribution comparison between backtest and live trading.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        # Check for profit column
        if 'final_profit' not in backtest_df.columns or 'final_profit' not in live_df.columns:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, "Profit data not available", ha='center', va='center')
            return fig
            
        # Prepare data
        backtest_profit = backtest_df['final_profit'].dropna()
        live_profit = live_df['final_profit'].dropna()
        
        if backtest_profit.empty and live_profit.empty:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, "No profit data available", ha='center', va='center')
            return fig
            
        # Plot histograms
        ax1 = fig.add_subplot(gs[0, 0])
        if not backtest_profit.empty:
            sns.histplot(backtest_profit, bins=20, kde=True, ax=ax1, color=self.default_colors['backtest'])
        ax1.set_title(f'Backtest Profit Distribution (n={len(backtest_profit)})')
        ax1.set_xlabel('Profit Points')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if not live_profit.empty:
            sns.histplot(live_profit, bins=20, kde=True, ax=ax2, color=self.default_colors['live'])
        ax2.set_title(f'Live Profit Distribution (n={len(live_profit)})')
        ax2.set_xlabel('Profit Points')
        
        # Plot statistics
        ax3 = fig.add_subplot(gs[1, :])
        
        # Prepare statistics
        stats = []
        
        if not backtest_profit.empty:
            stats.append({
                'source': 'Backtest',
                'mean': backtest_profit.mean(),
                'median': backtest_profit.median(),
                'std': backtest_profit.std(),
                'min': backtest_profit.min(),
                'max': backtest_profit.max(),
                'count': len(backtest_profit)
            })
            
        if not live_profit.empty:
            stats.append({
                'source': 'Live',
                'mean': live_profit.mean(),
                'median': live_profit.median(),
                'std': live_profit.std(),
                'min': live_profit.min(),
                'max': live_profit.max(),
                'count': len(live_profit)
            })
            
        stats_df = pd.DataFrame(stats)
        
        # Create a table
        if not stats_df.empty:
            col_labels = ['Source', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count']
            cell_data = []
            
            for _, row in stats_df.iterrows():
                cell_data.append([
                    row['source'],
                    f"{row['mean']:.2f}",
                    f"{row['median']:.2f}",
                    f"{row['std']:.2f}",
                    f"{row['min']:.2f}",
                    f"{row['max']:.2f}",
                    f"{row['count']}"
                ])
                
            ax3.axis('tight')
            ax3.axis('off')
            table = ax3.table(cellText=cell_data, colLabels=col_labels, loc='center',
                            cellLoc='center', colColours=['#f2f2f2']*len(col_labels))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance_comparison(self,
                                         backtest_features: Dict[str, Dict],
                                         live_features: Dict[str, Dict],
                                         top_n: int = 10,
                                         figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot feature importance comparison between backtest and live trading.
        
        Args:
            backtest_features: Dictionary of feature importance for backtest
            live_features: Dictionary of feature importance for live
            top_n: Number of top features to display
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if not backtest_features and not live_features:
            plt.title("No feature importance data available")
            return fig
            
        # Prepare data for plotting
        features = set(backtest_features.keys()) | set(live_features.keys())
        data = []
        
        for feature in features:
            # Get importance scores, default to 0 if not present
            backtest_score = 0
            live_score = 0
            
            if feature in backtest_features:
                backtest_score = backtest_features[feature].get('importance_score', 0)
                
            if feature in live_features:
                live_score = live_features[feature].get('importance_score', 0)
                
            # Calculate average and absolute difference
            avg_score = (backtest_score + live_score) / 2
            diff_score = abs(backtest_score - live_score)
            
            # Clean up feature name for display
            display_name = feature.replace('entry_feature_', '').replace('_', ' ')
            
            data.append({
                'feature': display_name,
                'backtest_score': backtest_score,
                'live_score': live_score,
                'avg_score': avg_score,
                'diff_score': diff_score
            })
            
        # Convert to DataFrame and sort
        df = pd.DataFrame(data)
        
        if df.empty:
            plt.title("No feature importance data available")
            return fig
            
        # Sort by average importance and take top N
        df = df.sort_values('avg_score', ascending=False).head(top_n)
        
        # Set up plot
        x = np.arange(len(df))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, df['backtest_score'], width, 
             label='Backtest', color=self.default_colors['backtest'])
        ax.bar(x + width/2, df['live_score'], width,
             label='Live', color=self.default_colors['live'])
        
        # Add labels and title
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance Score')
        ax.set_title('Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['feature'], rotation=45, ha='right')
        ax.legend()
        
        # Add correlation value
        if len(df) > 1:
            corr = np.corrcoef(df['backtest_score'], df['live_score'])[0, 1]
            plt.figtext(0.95, 0.05, f"Correlation: {corr:.2f}",
                      ha='right', va='bottom', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
        
    def create_dashboard(self,
                       backtest_df: pd.DataFrame,
                       live_df: pd.DataFrame,
                       feature_importance: Dict = None,
                       figsize: Tuple[int, int] = (18, 12)) -> Figure:
        """
        Create a comprehensive dashboard comparing backtest and live trading.
        
        Args:
            backtest_df: DataFrame containing backtest trades
            live_df: DataFrame containing live trades
            feature_importance: Dictionary containing feature importance data
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib Figure object with dashboard
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Plot win rate comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_win_rate_in_axes(backtest_df, live_df, ax1)
        
        # Plot hold time comparison
        ax2 = fig.add_subplot(gs[0, 1:])
        self._plot_hold_time_in_axes(backtest_df, live_df, ax2)
        
        # Plot profit distribution
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_profit_dist_in_axes(backtest_df, live_df, ax3)
        
        # Plot trade count over time
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_trade_timeline_in_axes(backtest_df, live_df, ax4)
        
        # Add title
        plt.suptitle('Backtest vs Live Trading Comparison Dashboard', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
        
        return fig
    
    def _plot_win_rate_in_axes(self, backtest_df: pd.DataFrame, live_df: pd.DataFrame, ax) -> None:
        """Plot win rate comparison in the given axes."""
        # Calculate win rates
        win_rates = []
        
        if not backtest_df.empty and 'is_profitable' in backtest_df.columns:
            backtest_win_rate = backtest_df['is_profitable'].mean() * 100
            win_rates.append({
                'source': 'Backtest',
                'win_rate': backtest_win_rate,
                'count': len(backtest_df)
            })
            
        if not live_df.empty and 'is_profitable' in live_df.columns:
            live_win_rate = live_df['is_profitable'].mean() * 100
            win_rates.append({
                'source': 'Live',
                'win_rate': live_win_rate,
                'count': len(live_df)
            })
            
        if not win_rates:
            ax.text(0.5, 0.5, "No win rate data available", ha='center', va='center')
            return
            
        # Create plot
        df = pd.DataFrame(win_rates)
        sns.barplot(x='source', y='win_rate', data=df, palette=[
            self.default_colors['backtest'],
            self.default_colors['live']
        ], ax=ax)
        
        # Add value labels on top of bars
        for i, row in enumerate(win_rates):
            ax.text(i, row['win_rate'] + 1, f"{row['win_rate']:.1f}%\n(n={row['count']})",
                  ha='center', va='bottom')
        
        # Add labels and title
        ax.set_title('Win Rate Comparison')
        ax.set_ylabel('Win Rate (%)')
        ax.set_xlabel('')
        
        # Y axis starts at 0
        y_max = max([r['win_rate'] for r in win_rates])
        ax.set_ylim(0, y_max * 1.15)
    
    def _plot_hold_time_in_axes(self, backtest_df: pd.DataFrame, live_df: pd.DataFrame, ax) -> None:
        """Plot hold time comparison in the given axes."""
        # Check for hold time column
        if 'trade_duration' not in backtest_df.columns or 'trade_duration' not in live_df.columns:
            ax.text(0.5, 0.5, "Hold time data not available", ha='center', va='center')
            return
            
        # Create boxplot data
        backtest_hold = backtest_df['trade_duration'].dropna()
        live_hold = live_df['trade_duration'].dropna()
        
        data = []
        if not backtest_hold.empty:
            for val in backtest_hold:
                data.append({'source': 'Backtest', 'duration': val})
                
        if not live_hold.empty:
            for val in live_hold:
                data.append({'source': 'Live', 'duration': val})
        
        if not data:
            ax.text(0.5, 0.5, "No hold time data available", ha='center', va='center')
            return
            
        # Create plot
        df = pd.DataFrame(data)
        sns.boxplot(x='source', y='duration', data=df, palette=[
            self.default_colors['backtest'],
            self.default_colors['live']
        ], ax=ax)
        
        # Add labels and title
        ax.set_title('Trade Duration Comparison')
        ax.set_ylabel('Duration (minutes)')
        ax.set_xlabel('')
        
        # Add statistics as text
        stats_text = []
        
        if not backtest_hold.empty:
            stats_text.append(f"Backtest (n={len(backtest_hold)}):\n" + 
                            f"Mean: {backtest_hold.mean():.1f} min\n" +
                            f"Median: {backtest_hold.median():.1f} min")
        
        if not live_hold.empty:
            stats_text.append(f"Live (n={len(live_hold)}):\n" + 
                            f"Mean: {live_hold.mean():.1f} min\n" +
                            f"Median: {live_hold.median():.1f} min")
        
        ax.text(0.95, 0.95, '\n\n'.join(stats_text), transform=ax.transAxes,
              ha='right', va='top', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
    
    def _plot_profit_dist_in_axes(self, backtest_df: pd.DataFrame, live_df: pd.DataFrame, ax) -> None:
        """Plot profit distribution comparison in the given axes."""
        # Check for profit column
        if 'final_profit' not in backtest_df.columns or 'final_profit' not in live_df.columns:
            ax.text(0.5, 0.5, "Profit data not available", ha='center', va='center')
            return
            
        # Prepare data
        backtest_profit = backtest_df['final_profit'].dropna()
        live_profit = live_df['final_profit'].dropna()
        
        if backtest_profit.empty and live_profit.empty:
            ax.text(0.5, 0.5, "No profit data available", ha='center', va='center')
            return
        
        # Plot distributions
        if not backtest_profit.empty:
            sns.kdeplot(backtest_profit, ax=ax, label=f'Backtest (n={len(backtest_profit)})', 
                       color=self.default_colors['backtest'])
            ax.axvline(backtest_profit.mean(), color=self.default_colors['backtest'], 
                     linestyle='--', label=f'Backtest Mean: {backtest_profit.mean():.2f}')
            
        if not live_profit.empty:
            sns.kdeplot(live_profit, ax=ax, label=f'Live (n={len(live_profit)})', 
                       color=self.default_colors['live'])
            ax.axvline(live_profit.mean(), color=self.default_colors['live'], 
                     linestyle='--', label=f'Live Mean: {live_profit.mean():.2f}')
        
        # Add labels and title
        ax.set_title('Profit Distribution Comparison')
        ax.set_xlabel('Profit Points')
        ax.set_ylabel('Density')
        ax.legend()
    
    def _plot_trade_timeline_in_axes(self, backtest_df: pd.DataFrame, live_df: pd.DataFrame, ax) -> None:
        """Plot trade count over time in the given axes."""
        # Check for timestamp column
        if 'entry_timestamp' not in backtest_df.columns or 'entry_timestamp' not in live_df.columns:
            ax.text(0.5, 0.5, "Timeline data not available", ha='center', va='center')
            return
            
        # Count trades by date
        try:
            backtest_counts = backtest_df.set_index('entry_timestamp').resample('D').size()
            live_counts = live_df.set_index('entry_timestamp').resample('D').size()
            
            # Plot trade counts
            if not backtest_counts.empty:
                backtest_counts.plot(ax=ax, label='Backtest', color=self.default_colors['backtest'])
                
            if not live_counts.empty:
                live_counts.plot(ax=ax, label='Live', color=self.default_colors['live'])
                
            # Add labels and title
            ax.set_title('Trade Count Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Trade Count')
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting timeline: {str(e)}", ha='center', va='center')