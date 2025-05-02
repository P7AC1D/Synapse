#!/usr/bin/env python3
"""
Integration test to compare model predictions between bot and backtest environments.

This script helps identify discrepancies in how predictions are made between the
bot and backtest execution paths, which can lead to different trading decisions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Import bot and backtest components
from trade_model import TradeModel
from trading.environment import TradingEnv
from mt5_connector import MT5Connector
from data_fetcher import DataFetcher
from trade_executor import TradeExecutor
from config import (
    MODEL_PATH,
    MT5_SYMBOL,
    MT5_TIMEFRAME_MINUTES,
    BALANCE_PER_LOT,
    STOP_LOSS_PIPS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"prediction_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class PredictionConsistencyTester:
    """Tests consistency between bot and backtest prediction paths."""
    
    def __init__(self, model_path: str, data_path: str, initial_balance: float = 10000.0,
                 balance_per_lot: float = 1000.0, currency_conversion: float = 19.0,
                 use_mt5: bool = False, symbol: str = MT5_SYMBOL):
        """
        Initialize the prediction consistency tester.
        
        Args:
            model_path: Path to the model file
            data_path: Path to the CSV data file
            initial_balance: Initial account balance
            balance_per_lot: Balance per lot
            currency_conversion: Currency conversion rate (e.g., USD/ZAR)
            use_mt5: Whether to use MT5 for data (if False, use CSV)
            symbol: Trading symbol (for MT5)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.balance_per_lot = balance_per_lot
        self.currency_conversion = currency_conversion
        self.use_mt5 = use_mt5
        self.symbol = symbol
        
        # Will be initialized in setup methods
        self.data = None
        self.bot_model = None
        self.backtest_model = None
        self.mt5 = None
        self.data_fetcher = None
        
        # Track predictions for comparison
        self.backtest_actions = []
        self.bot_actions = []
        self.feature_snapshots = []
        self.discrepancies = []
        
        # Environment parameters
        self.point_value = 0.01
        self.min_lots = 0.01
        self.max_lots = 200.0
        self.contract_size = 100.0
        
    def setup_backtest(self):
        """Set up the backtest model and environment."""
        logger.info("Setting up backtest environment...")
        
        # Ensure model path is correct (remove duplicate .zip extension if present)
        if self.model_path.endswith('.zip.zip'):
            logger.warning(f"Detected duplicate .zip extension in model path: {self.model_path}")
            self.model_path = self.model_path.replace('.zip.zip', '.zip')
            logger.info(f"Using corrected model path: {self.model_path}")
            
        # Verify model file exists before attempting to load it
        if not os.path.isfile(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            
            # Try to find the model in the model directory
            model_dir = os.path.dirname(self.model_path)
            if os.path.isdir(model_dir):
                logger.info(f"Searching for models in directory: {model_dir}")
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
                if model_files:
                    alternative_model = os.path.join(model_dir, model_files[0])
                    logger.info(f"Found alternative model: {alternative_model}")
                    self.model_path = alternative_model
                    logger.info(f"Using alternative model: {self.model_path}")
                else:
                    logger.error(f"No model files found in {model_dir}")
                    sys.exit(1)
            else:
                logger.error(f"Model directory does not exist: {model_dir}")
                sys.exit(1)
        
        # Now attempt to load the model with the verified path
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            self.backtest_model = TradeModel(
                model_path=self.model_path,
                balance_per_lot=self.balance_per_lot,
                initial_balance=self.initial_balance,
                point_value=self.point_value,
                min_lots=self.min_lots,
                max_lots=self.max_lots,
                contract_size=self.contract_size
            )
            
            # Ensure model loaded correctly
            if not self.backtest_model.model:
                logger.error("Failed to load backtest model")
                sys.exit(1)
                
            logger.info("Backtest environment set up successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            sys.exit(1)
        
    def setup_bot(self):
        """Set up the bot model and environments."""
        logger.info("Setting up bot environment...")
        
        # Load model for bot
        self.bot_model = TradeModel(
            model_path=self.model_path,
            balance_per_lot=self.balance_per_lot,
            initial_balance=self.initial_balance,
            point_value=self.point_value,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
            contract_size=self.contract_size
        )
        
        # Set up MT5 connection if needed
        if self.use_mt5:
            self.mt5 = MT5Connector()
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5")
                sys.exit(1)
                
            self.data_fetcher = DataFetcher(
                self.mt5, self.symbol, MT5_TIMEFRAME_MINUTES, 500
            )
            
            # Get symbol parameters from connector
            try:
                self.contract_size, self.min_lots, self.max_lots, volume_step, self.point_value, digits = self.mt5.get_symbol_info(self.symbol)
                logger.info(f"Symbol info - Point: {self.point_value}, " 
                           f"Contract Size: {self.contract_size}, Min Lot: {self.min_lots}, "
                           f"Max Lot: {self.max_lots}")
            except Exception as e:
                logger.error(f"Failed to get symbol info for {self.symbol}: {e}")
                sys.exit(1)
        
        # Ensure model loaded correctly
        if not self.bot_model.model:
            logger.error("Failed to load bot model")
            sys.exit(1)
            
        logger.info("Bot environment set up successfully")
        
    def load_data(self):
        """Load data from CSV or MT5."""
        if self.use_mt5:
            logger.info("Fetching data from MT5...")
            self.data = self.data_fetcher.fetch_data()
            if self.data is None:
                logger.error("Failed to fetch data from MT5")
                sys.exit(1)
        else:
            logger.info(f"Loading data from CSV: {self.data_path}")
            try:
                self.data = pd.read_csv(self.data_path)
                # Convert time column to datetime and set as index
                self.data['time'] = pd.to_datetime(self.data['time'], utc=True)
                self.data.set_index('time', inplace=True)
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                sys.exit(1)
                
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        
    def compare_features(self, bot_obs, backtest_obs, step):
        """Compare features between bot and backtest environments."""
        if bot_obs is None or backtest_obs is None:
            return {"error": "One or both observations are None"}
            
        # Ensure both are numpy arrays with same shape
        if not isinstance(bot_obs, np.ndarray) or not isinstance(backtest_obs, np.ndarray):
            return {"error": "Observations are not numpy arrays"}
            
        if bot_obs.shape != backtest_obs.shape:
            return {"error": f"Observations have different shapes: {bot_obs.shape} vs {backtest_obs.shape}"}
            
        # Calculate differences
        differences = bot_obs - backtest_obs
        max_diff = np.max(np.abs(differences))
        avg_diff = np.mean(np.abs(differences))
        
        # Find features with significant differences
        significant_diffs = []
        for i, diff in enumerate(differences):
            if abs(diff) > 1e-6:  # Threshold for significant difference
                significant_diffs.append({
                    "feature_index": i, 
                    "bot_value": float(bot_obs[i]), 
                    "backtest_value": float(backtest_obs[i]),
                    "difference": float(diff)
                })
                
        return {
            "step": step,
            "max_difference": float(max_diff),
            "avg_difference": float(avg_diff),
            "significant_differences": significant_diffs
        }
    
    def test_prediction_consistency(self, num_steps=None):
        """
        Test prediction consistency between bot and backtest.
        
        Args:
            num_steps: Number of steps to test (defaults to all available data)
        """
        if self.data is None:
            logger.error("No data loaded")
            return
            
        max_steps = len(self.data) - 1
        if num_steps is None or num_steps > max_steps:
            num_steps = max_steps
            
        logger.info(f"Testing prediction consistency over {num_steps} steps")
        
        # Reset models
        self.bot_model.reset_states()
        self.backtest_model.reset_states()
        
        # Create environments for both paths
        backtest_env = TradingEnv(
            data=self.data.copy(),
            initial_balance=self.initial_balance,
            balance_per_lot=self.balance_per_lot,
            random_start=False,
            point_value=self.point_value,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
            contract_size=self.contract_size,
            currency_conversion=self.currency_conversion
        )
        bot_env = TradingEnv(
            data=self.data.copy(),
            initial_balance=self.initial_balance,
            balance_per_lot=self.balance_per_lot,
            random_start=False,
            point_value=self.point_value,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
            contract_size=self.contract_size,
            currency_conversion=self.currency_conversion
        )
        
        # Reset environments
        backtest_obs, _ = backtest_env.reset()
        bot_obs, _ = bot_env.reset()
        
        # Track testing progress
        progress_interval = max(1, num_steps // 20)  # Report progress at 5% intervals
        
        # Set same initial LSTM states for both models
        self.bot_model.lstm_states = None
        self.backtest_model.lstm_states = None
        
        # Snapshot of initial observations
        initial_comparison = self.compare_features(bot_obs, backtest_obs, 0)
        if initial_comparison.get("significant_differences"):
            logger.warning(f"Initial observation differences: {initial_comparison}")
            self.discrepancies.append(initial_comparison)
            
        # Main prediction loop
        for step in range(num_steps):
            if step % progress_interval == 0:
                logger.info(f"Testing step {step}/{num_steps} ({step/num_steps*100:.1f}%)")
                
            # Get predictions from both models
            backtest_action, backtest_lstm_states = self.backtest_model.model.predict(
                backtest_obs,
                state=self.backtest_model.lstm_states,
                deterministic=True
            )
            self.backtest_model.lstm_states = backtest_lstm_states
            
            bot_action, bot_lstm_states = self.bot_model.model.predict(
                bot_obs,
                state=self.bot_model.lstm_states,
                deterministic=True
            )
            self.bot_model.lstm_states = bot_lstm_states
            
            # Convert actions to discrete values (same as in bot.py)
            if isinstance(backtest_action, np.ndarray):
                backtest_action_value = int(backtest_action.item())
            else:
                backtest_action_value = int(backtest_action)
            backtest_discrete_action = backtest_action_value % 4
            
            if isinstance(bot_action, np.ndarray):
                bot_action_value = int(bot_action.item())
            else:
                bot_action_value = int(bot_action)
            bot_discrete_action = bot_action_value % 4
            
            # Check if actions match
            if backtest_discrete_action != bot_discrete_action:
                logger.warning(f"Step {step}: Action mismatch - Backtest: {backtest_discrete_action}, Bot: {bot_discrete_action}")
                
                # Capture feature snapshot for analysis
                feature_comparison = self.compare_features(bot_obs, backtest_obs, step)
                self.discrepancies.append({
                    "step": step,
                    "backtest_action": int(backtest_discrete_action),
                    "bot_action": int(bot_discrete_action),
                    "feature_comparison": feature_comparison
                })
            
            # Store predictions for later analysis
            self.backtest_actions.append(backtest_discrete_action)
            self.bot_actions.append(bot_discrete_action)
            
            # Capture feature snapshots at regular intervals or on discrepancies
            if step % 100 == 0 or backtest_discrete_action != bot_discrete_action:
                feature_names = backtest_env.feature_processor.get_feature_names()
                feature_snapshot = {
                    "step": step,
                    "features": {}
                }
                
                for i, (bot_val, backtest_val) in enumerate(zip(bot_obs, backtest_obs)):
                    feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    feature_snapshot["features"][feature_name] = {
                        "bot": float(bot_val),
                        "backtest": float(backtest_val),
                        "diff": float(bot_val - backtest_val)
                    }
                    
                self.feature_snapshots.append(feature_snapshot)
            
            # Execute step in both environments
            backtest_obs, _, _, _, _ = backtest_env.step(backtest_discrete_action)
            bot_obs, _, _, _, _ = bot_env.step(bot_discrete_action)
            
            # Check for feature discrepancies after step
            post_step_comparison = self.compare_features(bot_obs, backtest_obs, step+1)
            if post_step_comparison.get("significant_differences"):
                self.discrepancies.append({
                    "step": step+1,
                    "stage": "post_step",
                    "feature_comparison": post_step_comparison
                })
                
        logger.info("Prediction consistency test completed")
        self.analyze_results()
        
    def analyze_results(self):
        """Analyze test results and show statistics."""
        total_steps = len(self.backtest_actions)
        if total_steps == 0:
            logger.warning("No predictions captured for analysis")
            return
            
        # Calculate action agreement rate
        matching_actions = sum(1 for b, t in zip(self.backtest_actions, self.bot_actions) if b == t)
        agreement_rate = matching_actions / total_steps
        
        # Analyze action distributions
        backtest_action_counts = {i: self.backtest_actions.count(i) for i in range(4)}
        bot_action_counts = {i: self.bot_actions.count(i) for i in range(4)}
        
        # Calculate feature discrepancy statistics
        feature_diffs = {}
        significant_discrepancies = 0
        
        for snapshot in self.feature_snapshots:
            for feat_name, values in snapshot["features"].items():
                if feat_name not in feature_diffs:
                    feature_diffs[feat_name] = []
                feature_diffs[feat_name].append(abs(values["diff"]))
        
        # Identify features with largest discrepancies
        feature_avg_diffs = {}
        for feat_name, diffs in feature_diffs.items():
            if diffs:
                feature_avg_diffs[feat_name] = sum(diffs) / len(diffs)
        
        sorted_features = sorted(feature_avg_diffs.items(), key=lambda x: x[1], reverse=True)
        
        # Report results
        logger.info(f"\nPrediction Consistency Analysis:")
        logger.info(f"Total steps analyzed: {total_steps}")
        logger.info(f"Action agreement rate: {agreement_rate:.4f} ({matching_actions}/{total_steps})")
        
        logger.info("\nAction distributions:")
        logger.info(f"Backtest: {backtest_action_counts}")
        logger.info(f"Bot: {bot_action_counts}")
        
        logger.info(f"\nFeatures with largest discrepancies:")
        for feat_name, avg_diff in sorted_features[:5]:
            logger.info(f"{feat_name}: avg diff {avg_diff:.6f}")
            
        logger.info(f"\nDiscrepancies detected: {len(self.discrepancies)}")
        
        # Show most significant discrepancies
        if self.discrepancies:
            logger.info("\nTop discrepancies:")
            for i, discrepancy in enumerate(self.discrepancies[:5]):
                logger.info(f"Discrepancy {i+1} at step {discrepancy.get('step')}:")
                if 'backtest_action' in discrepancy and 'bot_action' in discrepancy:
                    logger.info(f"  Actions: Backtest={discrepancy['backtest_action']}, Bot={discrepancy['bot_action']}")
                
                feature_comparison = discrepancy.get('feature_comparison', {})
                sig_diffs = feature_comparison.get('significant_differences', [])
                if sig_diffs:
                    logger.info(f"  Features with significant differences:")
                    for diff in sig_diffs[:3]:  # Show top 3 significant differences
                        logger.info(f"    Feature {diff['feature_index']}: "
                                   f"Bot={diff['bot_value']:.6f}, "
                                   f"Backtest={diff['backtest_value']:.6f}, "
                                   f"Diff={diff['difference']:.6f}")
        
    def export_results(self, output_file=None):
        """Export test results to CSV."""
        if output_file is None:
            output_file = f"prediction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        result_rows = []
        for step in range(len(self.backtest_actions)):
            result_rows.append({
                "step": step,
                "backtest_action": self.backtest_actions[step],
                "bot_action": self.bot_actions[step],
                "match": self.backtest_actions[step] == self.bot_actions[step]
            })
            
        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(output_file, index=False)
        logger.info(f"Results exported to {output_file}")
        
        # Export feature snapshots if available
        if self.feature_snapshots:
            feature_file = f"feature_snapshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Flatten feature snapshots for CSV export
            flattened_rows = []
            for snapshot in self.feature_snapshots:
                step = snapshot["step"]
                for feat_name, values in snapshot["features"].items():
                    flattened_rows.append({
                        "step": step,
                        "feature": feat_name,
                        "bot_value": values["bot"],
                        "backtest_value": values["backtest"],
                        "difference": values["diff"]
                    })
                    
            feature_df = pd.DataFrame(flattened_rows)
            feature_df.to_csv(feature_file, index=False)
            logger.info(f"Feature snapshots exported to {feature_file}")
            
    def run_test(self, num_steps=None, export=True):
        """Run the complete test procedure."""
        # Set up environments
        self.setup_backtest()
        self.setup_bot()
        self.load_data()
        
        # Run consistency test
        self.test_prediction_consistency(num_steps)
        
        # Export results if requested
        if export:
            self.export_results()


def main():
    """Main function to run the prediction consistency test."""
    parser = argparse.ArgumentParser(description='Test prediction consistency between bot and backtest environments')
    
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                      help=f'Path to the model file (default: {MODEL_PATH})')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV data file')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial account balance (default: 10000.0)')
    parser.add_argument('--balance_per_lot', type=float, default=BALANCE_PER_LOT,
                      help=f'Balance per lot (default: {BALANCE_PER_LOT})')
    parser.add_argument('--currency_conversion', type=float, default=19.0,
                      help='Currency conversion rate, e.g., USD/ZAR (default: 19.0)')
    parser.add_argument('--num_steps', type=int, default=None,
                      help='Number of steps to test (default: all available data)')
    parser.add_argument('--use_mt5', action='store_true',
                      help='Use MT5 for data instead of CSV')
    parser.add_argument('--symbol', type=str, default=MT5_SYMBOL,
                      help=f'Trading symbol (default: {MT5_SYMBOL})')
    parser.add_argument('--no_export', action='store_true',
                      help='Don\'t export results to CSV')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = PredictionConsistencyTester(
        model_path=args.model_path,
        data_path=args.data_path,
        initial_balance=args.initial_balance,
        balance_per_lot=args.balance_per_lot,
        currency_conversion=args.currency_conversion,
        use_mt5=args.use_mt5,
        symbol=args.symbol
    )
    
    tester.run_test(num_steps=args.num_steps, export=not args.no_export)


if __name__ == "__main__":
    main()