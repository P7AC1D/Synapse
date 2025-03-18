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


def run_backtest(data: pd.DataFrame, model_path: str, test_size: int,
                initial_balance: float, risk_percentage: float) -> Dict[str, Any]:
    """
    Run backtest on historical data using the specified model.
    
    Args:
        data: DataFrame with market data
        model_path: Path to the saved model file
        test_size: Number of bars to use for testing
        initial_balance: Starting account balance
        risk_percentage: Risk percentage per trade
        
    Returns:
        Dictionary with backtest results
    """
    # Create trade model with specified hyperparameters
    model = TradeModel(model_path=model_path)
    
    # Use last test_size bars for backtest
    backtest_data = data.iloc[-test_size:]
    logging.info(f"Running backtest on {len(backtest_data)} bars...")
    
    # Run backtest with specified initial balance and risk
    return model.backtest(backtest_data, initial_balance, risk_percentage)


def print_results(results: Dict[str, Any]) -> None:
    """
    Print backtest results in a detailed format matching the evaluation output.
    
    Args:
        results: Dictionary with backtest results
    """
    # Get total trades (to check if anything happened)
    total_trades = results.get('total_trades', 0)
    
    logging.info("\n===== EVALUATION METRICS =====")
    logging.info(f"Backtest complete. Total reward: {results.get('total_reward', 0.0):.2f}")
    
    logging.info(f"\n===== Environment State at Step {results.get('total_steps', 0)} =====")
    logging.info(f"Open Positions: {results.get('open_positions', 0)}")
    
    logging.info("\n===== Trading Performance Metrics =====")
    logging.info(f"Current Balance: {results['final_balance']:.2f}")
    logging.info(f"Initial Balance: {results['initial_balance']:.2f}")
    logging.info(f"Total Return: {results['return_pct']:.2f}%")
    logging.info(f"Total Trades: {total_trades}")
    logging.info(f"Total Win Rate: {results['win_rate']:.2f}%")
    
    if 'long_trades' in results and 'short_trades' in results:
        logging.info(f"Long Trades: {results['long_trades']}")
        logging.info(f"Long Win Rate: {results.get('long_win_rate', 0.0):.2f}%")
        logging.info(f"Short Trades: {results['short_trades']}")
        logging.info(f"Short Win Rate: {results.get('short_win_rate', 0.0):.2f}%")
        
    if 'avg_profit' in results and 'avg_loss' in results:
        logging.info(f"Average Win: {results['avg_profit']:.2f}")
        logging.info(f"Average Loss: {results['avg_loss']:.2f}")
        
    if 'risk_reward_ratio' in results:
        logging.info(f"Average RRR: {results['risk_reward_ratio']:.2f}")
        
    if 'expected_value' in results:
        logging.info(f"Expected Value: {results['expected_value']:.2f}")
        
    if 'kelly_criterion' in results:
        logging.info(f"Kelly Criterion: {results['kelly_criterion']:.2f}")
        
    if 'sharpe_ratio' in results:
        logging.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
    if 'profit_factor' in results:
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
        "--test-size", 
        type=int, 
        default=150000,
        help="Number of bars to use for testing"
    )
    parser.add_argument(
        "--initial-balance", 
        type=float, 
        default=10000.0,
        help="Initial account balance"
    )
    parser.add_argument(
        "--risk-percentage", 
        type=float, 
        default=0.01,
        help="Risk percentage per trade (0.01 = 1%)"
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
            args.test_size,
            args.initial_balance,
            args.risk_percentage
        )
        
        # Print results
        print_results(results)
        return 0
        
    except Exception as e:
        logging.error(f"Error during backtest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())