#!/usr/bin/env python3
"""
Utility script to generate test data for prediction consistency testing.

This script extracts a sample of data from the existing datasets
and creates a CSV file suitable for the prediction consistency test.
"""

import os
import sys
import logging
import pandas as pd
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_test_data(
    source_data_path: str,
    output_path: str,
    sample_size: int = 1000,
    random_sample: bool = False,
    padding: int = 20
):
    """
    Generates a test data file by sampling from an existing dataset.
    
    Args:
        source_data_path: Path to the source data CSV file
        output_path: Path to write the output CSV file
        sample_size: Number of rows to include in the sample
        random_sample: Whether to take a random sample (if False, takes the last N rows)
        padding: Extra rows to add to prevent index out of bounds errors
    """
    # Validate source path
    if not os.path.exists(source_data_path):
        logger.error(f"Source data file not found: {source_data_path}")
        sys.exit(1)
        
    # Load source data
    try:
        logger.info(f"Loading source data from: {source_data_path}")
        data = pd.read_csv(source_data_path)
        logger.info(f"Loaded {len(data)} rows from source data")
    except Exception as e:
        logger.error(f"Failed to load source data: {e}")
        sys.exit(1)
        
    # Check if 'time' column exists
    if 'time' not in data.columns:
        logger.warning("No 'time' column found in source data. This may cause issues with testing.")
        
    # Sample the data with extra padding
    total_size = sample_size + padding
    
    if len(data) <= total_size:
        logger.info(f"Source data contains fewer rows ({len(data)}) than requested sample size ({total_size}). Using all rows.")
        sample = data
    else:
        if random_sample:
            logger.info(f"Selecting random sample of {total_size} rows")
            sample = data.sample(n=total_size, random_state=42)
        else:
            logger.info(f"Selecting last {total_size} rows")
            sample = data.tail(total_size)
            
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    # Write output file
    try:
        sample.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote {len(sample)} rows to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        sys.exit(1)
        
    return len(sample)
        

def main():
    """Main function to parse arguments and generate test data."""
    parser = argparse.ArgumentParser(description='Generate test data for prediction consistency testing')
    
    parser.add_argument('--source', type=str, default='../data/XAUUSDm_15min.csv',
                      help='Path to source data CSV file (default: ../data/XAUUSDm_15min.csv)')
    parser.add_argument('--output', type=str, default='../results/prediction_test.csv',
                      help='Path to output CSV file (default: ../results/prediction_test.csv)')
    parser.add_argument('--sample_size', type=int, default=1000,
                      help='Number of rows to include in the sample (default: 1000)')
    parser.add_argument('--random_sample', action='store_true',
                      help='Take a random sample instead of the last N rows')
    
    args = parser.parse_args()
    
    # Generate test data
    generate_test_data(
        source_data_path=args.source,
        output_path=args.output,
        sample_size=args.sample_size,
        random_sample=args.random_sample
    )
    
    # Print instructions for running the test
    logger.info("\nTo run the prediction consistency test with this data, use:")
    logger.info(f"python test_prediction_consistency.py --data_path {args.output} --model_path ../model/XAUUSDm.zip")


if __name__ == "__main__":
    main()