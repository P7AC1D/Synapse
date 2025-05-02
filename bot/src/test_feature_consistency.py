#!/usr/bin/env python3
"""
Test script to verify feature consistency between different environments.

This test helps confirm that feature preprocessing and generation is identical
across different parts of the trading system.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Import necessary components
from trading.features import FeatureProcessor
from config import MT5_SYMBOL

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

class FeatureConsistencyTester:
    """Tests consistency of feature generation across environments."""
    
    def __init__(self, data_path: str):
        """
        Initialize the feature consistency tester.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        
        # Will be initialized in setup methods
        self.data = None
        self.feature_processor_1 = None
        self.feature_processor_2 = None
        self.processed_data_1 = None
        self.processed_data_2 = None
        
        # Track feature differences
        self.feature_differences = []
        
    def load_data(self):
        """Load data from CSV."""
        logger.info(f"Loading data from CSV: {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path)
            # Convert time column to datetime and set as index if it exists
            if 'time' in self.data.columns:
                self.data['time'] = pd.to_datetime(self.data['time'], utc=True)
                self.data.set_index('time', inplace=True)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
            
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'spread']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"Data is missing required columns: {missing_columns}")
            sys.exit(1)
            
        # Add volume if not present (common in forex data)
        if 'volume' not in self.data.columns:
            logger.warning("No 'volume' column found in data, adding synthetic volume")
            self.data['volume'] = 1.0
            
    def setup_feature_processors(self):
        """Set up two independent feature processors for comparison."""
        logger.info("Setting up feature processors...")
        
        # Create two separate instances of FeatureProcessor
        self.feature_processor_1 = FeatureProcessor()
        self.feature_processor_2 = FeatureProcessor()
        
        # Verify they are independent instances
        if id(self.feature_processor_1) == id(self.feature_processor_2):
            logger.error("Failed to create independent feature processor instances")
            sys.exit(1)
            
        logger.info("Feature processors set up successfully")
        
    def process_features(self):
        """Process features with both processors."""
        if self.data is None:
            logger.error("No data loaded")
            return
            
        logger.info("Processing features with both processors...")
        
        # Process with first processor
        try:
            self.processed_data_1, self.atr_values_1 = self.feature_processor_1.preprocess_data(self.data.copy())
            logger.info(f"Processor 1: Generated {self.processed_data_1.shape[1]} features for {len(self.processed_data_1)} bars")
        except Exception as e:
            logger.error(f"Error processing features with processor 1: {e}")
            sys.exit(1)
            
        # Process with second processor
        try:
            self.processed_data_2, self.atr_values_2 = self.feature_processor_2.preprocess_data(self.data.copy())
            logger.info(f"Processor 2: Generated {self.processed_data_2.shape[1]} features for {len(self.processed_data_2)} bars")
        except Exception as e:
            logger.error(f"Error processing features with processor 2: {e}")
            sys.exit(1)
            
        # Initial shape check
        if self.processed_data_1.shape != self.processed_data_2.shape:
            logger.error(f"Feature shape mismatch: {self.processed_data_1.shape} vs {self.processed_data_2.shape}")
            sys.exit(1)
            
        # Initial ATR check
        atr_diff = np.abs(self.atr_values_1 - self.atr_values_2)
        max_atr_diff = np.max(atr_diff) if len(atr_diff) > 0 else 0
        if max_atr_diff > 1e-6:
            logger.error(f"ATR values differ: Max difference = {max_atr_diff}")
            sys.exit(1)
            
        logger.info("Features processed successfully")
        
    def compare_features(self):
        """Compare features between the two processors."""
        if self.processed_data_1 is None or self.processed_data_2 is None:
            logger.error("Features not processed")
            return
            
        logger.info("Comparing features between processors...")
        
        # Get feature names
        feature_names = self.feature_processor_1.get_feature_names()
        
        # Calculate differences
        differences = np.abs(self.processed_data_1 - self.processed_data_2)
        
        # Check for significant differences
        max_diff = np.max(differences)
        avg_diff = np.mean(differences)
        
        if max_diff > 1e-6:
            logger.warning(f"Found feature differences: Max = {max_diff}, Avg = {avg_diff}")
            
            # Analyze differences per feature
            for i in range(differences.shape[1]):
                feature_diff = differences[:, i]
                max_feature_diff = np.max(feature_diff)
                avg_feature_diff = np.mean(feature_diff)
                
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                
                if max_feature_diff > 1e-6:
                    # Find locations of significant differences
                    sig_indices = np.where(feature_diff > 1e-6)[0]
                    examples = []
                    
                    for idx in sig_indices[:min(5, len(sig_indices))]:
                        examples.append({
                            "index": idx,
                            "value1": float(self.processed_data_1[idx, i]),
                            "value2": float(self.processed_data_2[idx, i]),
                            "difference": float(feature_diff[idx])
                        })
                    
                    self.feature_differences.append({
                        "feature": feature_name,
                        "max_diff": float(max_feature_diff),
                        "avg_diff": float(avg_feature_diff),
                        "examples": examples
                    })
        else:
            logger.info("No significant feature differences found (max diff < 1e-6)")
        
        # Summarize findings    
        if self.feature_differences:
            logger.warning(f"Found differences in {len(self.feature_differences)} features")
            
            # Sort by max difference
            sorted_diffs = sorted(self.feature_differences, key=lambda x: x["max_diff"], reverse=True)
            
            # Show top differences
            for i, diff in enumerate(sorted_diffs[:5]):
                logger.warning(f"Difference {i+1}: {diff['feature']} - Max: {diff['max_diff']:.8f}, Avg: {diff['avg_diff']:.8f}")
                logger.warning("Examples:")
                for ex in diff["examples"]:
                    logger.warning(f"  Index {ex['index']}: {ex['value1']:.8f} vs {ex['value2']:.8f}, Diff: {ex['difference']:.8f}")
        else:
            logger.info("All features are identical between processors")
            
    def export_results(self, output_file=None):
        """Export test results to CSV."""
        if not self.feature_differences:
            logger.info("No differences to export")
            return
            
        if output_file is None:
            output_file = f"feature_differences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # Flatten differences for CSV export
        rows = []
        for diff in self.feature_differences:
            feature = diff["feature"]
            for ex in diff["examples"]:
                rows.append({
                    "feature": feature,
                    "index": ex["index"],
                    "value1": ex["value1"],
                    "value2": ex["value2"],
                    "difference": ex["difference"],
                    "max_diff_for_feature": diff["max_diff"],
                    "avg_diff_for_feature": diff["avg_diff"]
                })
                
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(rows)} difference records to {output_file}")
            
    def run_test(self, export=True):
        """Run the complete feature consistency test."""
        self.load_data()
        self.setup_feature_processors()
        self.process_features()
        self.compare_features()
        
        if export:
            self.export_results()
            

def main():
    """Main function to run the feature consistency test."""
    parser = argparse.ArgumentParser(description='Test feature consistency across different processors')
    
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV data file')
    parser.add_argument('--no_export', action='store_true',
                      help='Don\'t export results to CSV')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = FeatureConsistencyTester(data_path=args.data_path)
    tester.run_test(export=not args.no_export)


if __name__ == "__main__":
    main()