#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest script for evaluating trading model performance on historical data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from trade_model import TradeModel


def setup_logging() -> None:
    """Configure logging with console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load and preprocess historical market data.
    
    Args:
        file_path: Path to CSV data file
        
    Returns:
        Preprocessed DataFrame or None if loading failed
    """
    try:
        data = pd.read_csv(file_path)
        data.set_index('time', inplace=True)
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower']
        existing_columns = [col for col in columns_to_drop if col in data.columns]
        if existing_columns:
            data.drop(columns=existing_columns, inplace=True)
            
        # Remove rows with NaN values
        data.dropna(inplace=True)
        
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None


def print_data_info(data: pd.DataFrame) -> None:
    """
    Print information about the loaded data.
    
    Args:
        data: DataFrame with market data
    """
    start_datetime = data.index[0]
    end_datetime = data.index[-1]
    logging.info(f"Data range: {start_datetime} to {end_datetime}")
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Last 5 rows:\n{data.tail()}")


def run_backtest(data: pd.DataFrame, model_path: str, bar_count: int, 
               normalization_window: int, test_size: int) -> Dict[str, Any]:
    """
    Run backtest on historical data using the specified model.
    
    Args:
        data: DataFrame with market data
        model_path: Path to the saved model file
        bar_count: Number of bars in each observation
        normalization_window: Window size for normalization
        test_size: Number of bars to use for testing
        
    Returns:
        Dictionary with backtest results
    """
    # Create trade model with specified hyperparameters
    model = TradeModel(
        model_path=model_path,
        bar_count=bar_count,
        normalization_window=normalization_window
    )
    
    # Use last test_size bars for backtest
    backtest_data = data.iloc[-test_size:]
    logging.info(f"Running backtest on {len(backtest_data)} bars...")
    
    # Run backtest
    return model.backtest(backtest_data)


def print_results(results: Dict[str, Any]) -> None:
    """
    Print backtest results.
    
    Args:
        results: Dictionary with backtest results
    """
    logging.info(f"\nBacktest Results:")
    logging.info(f"Final Balance: ${results['final_balance']:.2f}")
    logging.info(f"Return: {results['return_pct']:.2f}%")
    logging.info(f"Total Trades: {results['total_trades']}")
    logging.info(f"Win Rate: {results['win_rate']:.2f}%")
    logging.info(f"Profit Factor: {results['profit_factor']:.2f}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run backtest on trading model")
    
    parser.add_argument(
        "--data", 
        type=str, 
        default="../data/BTCUSDm_60min.csv",
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="../results/65478/best_balance_model.zip",
        help="Path to model file"
    )
    parser.add_argument(
        "--bars", 
        type=int, 
        default=50,
        help="Number of bars in each observation"
    )
    parser.add_argument(
        "--norm-window", 
        type=int, 
        default=100,
        help="Window size for normalization"
    )
    parser.add_argument(
        "--test-size", 
        type=int, 
        default=10000,
        help="Number of bars to use for testing"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the backtest script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load data
    data = load_data(args.data)
    if data is None:
        return 1
    
    # Print data information
    print_data_info(data)
    
    try:
        # Run backtest
        results = run_backtest(
            data, 
            args.model, 
            args.bars, 
            args.norm_window, 
            args.test_size
        )
        
        # Print results
        print_results(results)
        return 0
        
    except Exception as e:
        logging.error(f"Error during backtest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())