#!/usr/bin/env python3
"""
Test to specifically compare data preprocessing and feature generation between
bot and backtest environments.

This script focuses on the data preparation aspects to identify discrepancies
in feature calculation that might lead to different model predictions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import bot and backtest components
from trade_model import TradeModel
from trading.environment import TradingEnv
from data_fetcher import DataFetcher
from trading.features import FeatureProcessor
from config import (
    MODEL_PATH,
    MT5_SYMBOL,
    MT5_TIMEFRAME_MINUTES,
    BALANCE_PER_LOT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"feature_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class FeatureComparisonTester:
    """Tests feature calculation consistency between bot and backtest."""
    
    def __init__(self, data_path: str, output_dir: str = './feature_comparison'):
        """
        Initialize the feature comparison tester.
        
        Args:
            data_path: Path to the CSV data file
            output_dir: Directory to save output files and visualizations
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Will be initialized in setup methods
        self.data = None
        self.bot_features = None
        self.backtest_features = None
        self.feature_differences = None
        self.feature_names = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load data from CSV."""
        logger.info(f"Loading data from CSV: {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path)
            # Convert time column to datetime and set as index
            self.data['time'] = pd.to_datetime(self.data['time'], utc=True)
            self.data.set_index('time', inplace=True)
            
            # Basic validation of data
            required_columns = ['open', 'high', 'low', 'close', 'spread']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                sys.exit(1)
                
            logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
    
    def process_data_backtest_style(self):
        """Process data using the backtest approach."""
        logger.info("Processing data using backtest approach...")
        
        # Create feature processor - same as used in backtest
        feature_processor = FeatureProcessor()
        
        # Process the full dataset at once (the backtest approach)
        processed_data, atr_values = feature_processor.preprocess_data(self.data.copy())
        
        # Store the processed features
        self.backtest_features = processed_data
        
        # Get feature names for later reference
        self.feature_names = feature_processor.get_feature_names()
        
        logger.info(f"Processed {len(processed_data)} bars with backtest approach")
        
    def process_data_bot_style(self):
        """Process data sequentially to simulate bot's approach."""
        logger.info("Processing data using bot approach (sequential)...")
        
        # In the bot, data is often processed incrementally or in windows
        # We'll simulate this by processing data in chunks
        self.bot_features = []
        feature_processor = FeatureProcessor()
        
        # Instead of one-shot processing, we'll simulate incremental updates
        # by moving a window through the data and processing it piece by piece
        window_size = 500  # Typical rolling window size in the bot
        
        for i in range(len(self.data) - window_size + 1):
            # Get window of data (similar to what bot would see)
            window_data = self.data.iloc[i:i+window_size].copy()
            
            # Process window data
            processed_window, _ = feature_processor.preprocess_data(window_data)
            
            # Store only the last row, which would be what the bot uses for prediction
            if i == 0:
                # For the first window, we'll store all processed features to match backtest length
                self.bot_features.append(processed_window.values)
            else:
                # For subsequent windows, just add the latest processed row
                self.bot_features.append(processed_window.values[-1:])
        
        # Convert list of arrays to a single array
        self.bot_features = np.vstack(self.bot_features)
        
        logger.info(f"Processed {len(self.bot_features)} bars with bot approach")
    
    def compare_features(self):
        """Compare features between bot and backtest approaches."""
        logger.info("Comparing features between bot and backtest approaches...")
        
        # Ensure both feature arrays have the same length
        min_length = min(len(self.backtest_features), len(self.bot_features))
        
        # Trim to same length for comparison
        backtest_features = self.backtest_features[:min_length]
        bot_features = self.bot_features[:min_length]
        
        # Calculate differences
        self.feature_differences = bot_features - backtest_features
        
        # Calculate statistics
        mean_diffs = np.mean(np.abs(self.feature_differences), axis=0)
        max_diffs = np.max(np.abs(self.feature_differences), axis=0)
        
        # Log summary statistics
        logger.info(f"Feature comparison summary (across {min_length} bars):")
        logger.info(f"Overall mean absolute difference: {np.mean(mean_diffs):.6f}")
        logger.info(f"Overall max absolute difference: {np.max(max_diffs):.6f}")
        
        # Log per-feature statistics
        logger.info("\nPer-feature statistics:")
        for i, feature_name in enumerate(self.feature_names):
            if i < len(mean_diffs):
                logger.info(f"{feature_name}:")
                logger.info(f"  Mean abs diff: {mean_diffs[i]:.6f}")
                logger.info(f"  Max abs diff: {max_diffs[i]:.6f}")
    
    def visualize_differences(self):
        """Create visualizations of feature differences."""
        logger.info("Creating visualizations of feature differences...")
        
        # Ensure output directory exists
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        
        # Create heatmap of all feature differences
        plt.figure(figsize=(12, 8))
        plt.imshow(self.feature_differences.T, aspect='auto', cmap='coolwarm')
        plt.colorbar(label='Difference')
        plt.xlabel('Bar Index')
        plt.ylabel('Feature Index')
        plt.title('Feature Differences Between Bot and Backtest')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'feature_differences_heatmap.png'))
        
        # Create time series plots for features with largest differences
        mean_abs_diffs = np.mean(np.abs(self.feature_differences), axis=0)
        feature_indices = np.argsort(mean_abs_diffs)[-5:]  # Top 5 features with largest differences
        
        for idx in feature_indices:
            if idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
                plt.figure(figsize=(12, 6))
                
                # Plot both bot and backtest versions
                plt.plot(self.backtest_features[:, idx], label='Backtest', alpha=0.7)
                plt.plot(self.bot_features[:, idx], label='Bot', alpha=0.7)
                
                plt.title(f'Feature: {feature_name}')
                plt.xlabel('Bar Index')
                plt.ylabel('Feature Value')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'plots', f'feature_{idx}_{feature_name}.png'))
                
                # Plot the difference
                plt.figure(figsize=(12, 6))
                plt.plot(self.feature_differences[:, idx])
                plt.title(f'Difference in Feature: {feature_name}')
                plt.xlabel('Bar Index')
                plt.ylabel('Difference')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'plots', f'feature_{idx}_{feature_name}_diff.png'))
        
        logger.info(f"Visualizations saved to {os.path.join(self.output_dir, 'plots')}")
    
    def export_results(self):
        """Export feature comparison results to CSV."""
        logger.info("Exporting feature comparison results...")
        
        # Create a DataFrame with feature differences
        diff_data = pd.DataFrame(self.feature_differences, columns=self.feature_names)
        diff_data.index = self.data.index[:len(diff_data)]
        
        # Add statistics
        diff_data.loc['mean_abs_diff'] = np.mean(np.abs(self.feature_differences), axis=0)
        diff_data.loc['max_abs_diff'] = np.max(np.abs(self.feature_differences), axis=0)
        
        # Export to CSV
        diff_file = os.path.join(self.output_dir, 'feature_differences.csv')
        diff_data.to_csv(diff_file)
        
        # Export both feature sets for reference
        backtest_file = os.path.join(self.output_dir, 'backtest_features.csv')
        bot_file = os.path.join(self.output_dir, 'bot_features.csv')
        
        pd.DataFrame(self.backtest_features, columns=self.feature_names).to_csv(backtest_file)
        pd.DataFrame(self.bot_features, columns=self.feature_names).to_csv(bot_file)
        
        logger.info(f"Results exported to {self.output_dir}")
    
    def run_test(self, visualize=True, export=True):
        """Run the complete feature comparison test."""
        # Load and process data
        self.load_data()
        
        # Process data using both approaches
        self.process_data_backtest_style()
        self.process_data_bot_style()
        
        # Compare features
        self.compare_features()
        
        # Optional steps
        if visualize:
            self.visualize_differences()
            
        if export:
            self.export_results()
            
        logger.info("Feature comparison test completed")


def main():
    """Main function to run the feature comparison test."""
    parser = argparse.ArgumentParser(description='Compare feature calculation between bot and backtest environments')
    
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='./feature_comparison',
                      help='Directory to save output files and visualizations')
    parser.add_argument('--no_visualize', action='store_true',
                      help='Skip visualization of feature differences')
    parser.add_argument('--no_export', action='store_true',
                      help='Skip exporting results to CSV')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = FeatureComparisonTester(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    tester.run_test(
        visualize=not args.no_visualize,
        export=not args.no_export
    )


if __name__ == "__main__":
    main()