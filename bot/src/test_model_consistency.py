#!/usr/bin/env python3
"""
Model prediction consistency test between live bot and backtest environments.

This script simulates the exact data flow and prediction path used in both
the live trading bot and backtest to identify inconsistencies in model output
given the same input data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

# Import bot and backtest components
from trade_model import TradeModel
from trading.environment import TradingEnv
from config import MODEL_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class ModelConsistencyTester:
    """Tests model prediction consistency between bot and backtest environments."""
    
    def __init__(self, model_path: str, data_path: str, output_dir: str = './model_comparison'):
        """
        Initialize the model consistency tester.
        
        Args:
            model_path: Path to the model file
            data_path: Path to the CSV data file
            output_dir: Directory to save output files
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Internal variables
        self.data = None
        self.model = None
        self.discrepancies = []
        
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
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'spread']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                sys.exit(1)
                
            logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)
    
    def load_model(self):
        """Load the model."""
        logger.info(f"Loading model from: {self.model_path}")
        
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
        
        try:
            self.model = TradeModel(
                model_path=self.model_path,
                balance_per_lot=1000.0,
                initial_balance=10000.0,
                point_value=0.01,
                min_lots=0.01, 
                max_lots=200.0,
                contract_size=100.0
            )
            
            if not self.model.model:
                logger.error("Failed to load model")
                sys.exit(1)
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
    
    def simulate_bot_predictions(self, warmup_size=200):
        """
        Simulate the live bot prediction process using the same full-history approach
        that we've implemented in the bot.py file.
        
        Args:
            warmup_size: Number of bars to use for warmup (default: 200)
        
        Returns:
            List of prediction dictionaries containing actions and LSTM states
        """
        logger.info("Simulating bot predictions using full historical data approach...")
        
        # Reset model states for clean test
        self.model.reset_states()
        
        # Add a dictionary to track bot predictions
        bot_predictions = []
        
        try:
            # First, process all historical data to warm up LSTM states,
            # exactly as the live bot does after our changes
            
            # Make a copy of full data for safety
            data_copy = self.data.copy()
            
            # Add volume if it doesn't exist (common in forex data)
            if 'volume' not in data_copy.columns:
                data_copy['volume'] = 1.0
            
            # Create environment with full data like bot does
            env = TradingEnv(
                data=data_copy,
                initial_balance=10000.0,
                balance_per_lot=1000.0,
                random_start=False
            )
            
            # Initialize environment
            obs, _ = env.reset()
            
            # Process all data to warm up LSTM states (matching bot.py)
            logger.info(f"Warming up LSTM states with {env.data_length} historical bars")
            for i in range(env.data_length - 1):  # -1 to avoid the last potentially incomplete bar
                # Run observation through model to update LSTM states
                action, lstm_states = self.model.model.predict(obs, state=self.model.lstm_states, deterministic=True)
                self.model.lstm_states = lstm_states
                
                # Record prediction for each warm-up step
                if i >= (env.data_length - min(env.data_length, 1000)):  # Only store last 1000 predictions
                    discrete_action = int(action) % 4
                    bot_predictions.append({
                        'time': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                        'index': i,
                        'action': int(discrete_action),
                        'lstm_states_hash': hash(str(lstm_states))  # Use hash to track state changes
                    })
                
                # Get next observation
                obs, _, done, _, _ = env.step(int(action))
                if done:
                    break
                
                if i % 100 == 0:
                    logger.info(f"Warmed up through step {i}/{env.data_length} ({i/env.data_length*100:.1f}%)")
            
            logger.info(f"Completed bot simulation with {len(bot_predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error in bot simulation: {e}")
            # Fall back to a simpler approach if there's an error
            logger.warning("Falling back to simplified simulation")
            
            for i in range(min(100, len(self.data))):
                obs = np.zeros(64)  # Default observation shape
                action, lstm_states = self.model.model.predict(obs, state=self.model.lstm_states, deterministic=True)
                self.model.lstm_states = lstm_states
                
                discrete_action = int(action) % 4
                
                bot_predictions.append({
                    'time': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'index': i,
                    'action': int(discrete_action),
                    'lstm_states_hash': hash(str(lstm_states))
                })
        
        return bot_predictions
    
    def simulate_backtest_predictions(self):
        """
        Simulate the backtest prediction process.
        
        Returns:
            List of prediction dictionaries containing actions and LSTM states
        """
        logger.info("Simulating backtest predictions...")
        
        # Reset model states for clean test
        self.model.reset_states()
        
        # Add a dictionary to track backtest predictions
        backtest_predictions = []
        
        try:
            # Add volume if it doesn't exist (common in forex data)
            data_copy = self.data.copy()
            if 'volume' not in data_copy.columns:
                data_copy['volume'] = 1.0
            
            # Create environment with full data (as backtest does)
            env = TradingEnv(
                data=data_copy,
                initial_balance=10000.0,
                balance_per_lot=1000.0,
                random_start=False
            )
            
            # Initialize environment
            obs, _ = env.reset()
            
            # Make predictions sequentially throughout the dataset
            for i in range(env.data_length):  # Use environment's data_length instead of original data length
                # Get prediction using current observation and LSTM states
                action, lstm_states = self.model.model.predict(obs, state=self.model.lstm_states, deterministic=True)
                
                # Store LSTM states for next prediction
                self.model.lstm_states = lstm_states
                
                # Convert action to discrete value (as done in backtest.py)
                if isinstance(action, np.ndarray):
                    action_value = int(action.item())
                else:
                    action_value = int(action)
                discrete_action = action_value % 4
                
                # Record prediction
                current_time = self.data.index[min(i, len(self.data) - 1)]  # Safety check for index
                backtest_predictions.append({
                    'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'index': i,
                    'action': int(discrete_action),
                    'lstm_states_hash': hash(str(lstm_states))  # Use hash to track state changes
                })
                
                # Execute step in environment to get next observation
                try:
                    obs, _, done, _, _ = env.step(discrete_action)
                    if done:
                        logger.info(f"Environment signaled done at step {i}/{env.data_length}")
                        break
                except Exception as e:
                    logger.error(f"Error executing step {i} in backtest: {e}")
                    break
                
                if i % 100 == 0:
                    logger.info(f"Simulated backtest prediction for step {i}/{env.data_length} ({i/env.data_length*100:.1f}%)")
        except ValueError as e:
            logger.error(f"Feature preprocessing error in backtest: {str(e)}")
            logger.warning("Not enough data available after preprocessing. Using simpler simulation approach.")
            
            # Fallback to direct prediction without environment
            for i in range(10, len(self.data), 10):  # Sample every 10th point
                try:
                    # Create a basic observation
                    obs = np.zeros(64)  # Adjust size based on your model's input requirements
                    
                    # Get prediction
                    action, lstm_states = self.model.model.predict(obs, state=self.model.lstm_states, deterministic=True)
                    
                    # Store LSTM states
                    self.model.lstm_states = lstm_states
                    
                    # Convert action
                    if isinstance(action, np.ndarray):
                        action_value = int(action.item())
                    else:
                        action_value = int(action)
                    discrete_action = action_value % 4
                    
                    # Record prediction
                    backtest_predictions.append({
                        'time': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                        'index': i,
                        'action': int(discrete_action),
                        'lstm_states_hash': hash(str(lstm_states))
                    })
                except Exception as e:
                    logger.error(f"Error in fallback backtest simulation at index {i}: {e}")
                    continue
        
        logger.info(f"Completed backtest simulation with {len(backtest_predictions)} predictions")
        return backtest_predictions
    
    def compare_predictions(self, bot_predictions, backtest_predictions):
        """
        Compare bot and backtest predictions.
        
        Args:
            bot_predictions: List of bot prediction dictionaries
            backtest_predictions: List of backtest prediction dictionaries
        """
        logger.info("Comparing bot and backtest predictions...")
        
        # Create a mapping from time to index for easier comparison
        bot_pred_dict = {pred['index']: pred for pred in bot_predictions}
        backtest_pred_dict = {pred['index']: pred for pred in backtest_predictions}
        
        # Calculate total comparison points
        comparison_points = set(bot_pred_dict.keys()).intersection(set(backtest_pred_dict.keys()))
        logger.info(f"Found {len(comparison_points)} common prediction points for comparison")
        
        # Count matching and different predictions
        matching_count = 0
        different_count = 0
        
        # Track discrepancies
        self.discrepancies = []
        
        # Compare each prediction
        for idx in sorted(comparison_points):
            bot_pred = bot_pred_dict[idx]
            backtest_pred = backtest_pred_dict[idx]
            
            if bot_pred['action'] == backtest_pred['action']:
                matching_count += 1
            else:
                different_count += 1
                # Record discrepancy for analysis
                self.discrepancies.append({
                    'index': idx,
                    'time': bot_pred['time'],
                    'bot_action': bot_pred['action'],
                    'backtest_action': backtest_pred['action'],
                    'bot_states_hash': bot_pred['lstm_states_hash'],
                    'backtest_states_hash': backtest_pred['lstm_states_hash'],
                    'states_match': bot_pred['lstm_states_hash'] == backtest_pred['lstm_states_hash']
                })
        
        # Calculate agreement rate
        total_compared = matching_count + different_count
        agreement_rate = matching_count / total_compared if total_compared > 0 else 0.0
        
        # Log results
        logger.info("\nPrediction Comparison Results:")
        logger.info(f"Total compared: {total_compared}")
        logger.info(f"Matching predictions: {matching_count} ({agreement_rate*100:.2f}%)")
        logger.info(f"Different predictions: {different_count} ({(1-agreement_rate)*100:.2f}%)")
        
        # Count distribution of discrepancies for each prediction action
        if self.discrepancies:
            bot_actions = {}
            backtest_actions = {}
            
            for disc in self.discrepancies:
                bot_act = disc['bot_action']
                backtest_act = disc['backtest_action']
                
                bot_actions[bot_act] = bot_actions.get(bot_act, 0) + 1
                backtest_actions[backtest_act] = backtest_actions.get(backtest_act, 0) + 1
            
            logger.info("\nDiscrepancy Action Distribution:")
            logger.info(f"Bot actions: {dict(sorted(bot_actions.items()))}")
            logger.info(f"Backtest actions: {dict(sorted(backtest_actions.items()))}")
            
            # Check if the LSTM states are diverging
            states_match_count = sum(1 for d in self.discrepancies if d['states_match'])
            logger.info(f"\nLSTM state consistency: {states_match_count}/{len(self.discrepancies)} discrepancies have matching states")
            
            # Show sample of discrepancies
            logger.info("\nSample Discrepancies:")
            for i, disc in enumerate(self.discrepancies[:5]):
                logger.info(f"  {i+1}. Time: {disc['time']} | Bot: {disc['bot_action']} | Backtest: {disc['backtest_action']} | States match: {disc['states_match']}")
    
    def export_results(self):
        """Export test results to files."""
        logger.info("Exporting test results...")
        
        # Export discrepancies
        if self.discrepancies:
            # Convert to serializable format
            serializable_discrepancies = []
            for disc in self.discrepancies:
                # Convert any non-serializable types
                serializable_disc = {
                    k: str(v) if k.endswith('_hash') else v
                    for k, v in disc.items()
                }
                serializable_discrepancies.append(serializable_disc)
            
            # Save to file
            discrepancies_file = os.path.join(self.output_dir, 'prediction_discrepancies.json')
            with open(discrepancies_file, 'w') as f:
                json.dump(serializable_discrepancies, f, indent=2)
            
            # Create CSV version for easier analysis
            discrepancies_df = pd.DataFrame(serializable_discrepancies)
            discrepancies_csv = os.path.join(self.output_dir, 'prediction_discrepancies.csv')
            discrepancies_df.to_csv(discrepancies_csv, index=False)
            
            logger.info(f"Saved {len(self.discrepancies)} discrepancies to {discrepancies_file} and {discrepancies_csv}")
        else:
            logger.info("No discrepancies to export")
    
    def run_test(self, export=True):
        """Run the complete model consistency test."""
        # Load data and model
        self.load_data()
        self.load_model()
        
        # Simulate bot and backtest predictions
        bot_predictions = self.simulate_bot_predictions()
        backtest_predictions = self.simulate_backtest_predictions()
        
        # Compare predictions
        self.compare_predictions(bot_predictions, backtest_predictions)
        
        # Export results if requested
        if export:
            self.export_results()
            
        logger.info("Model consistency test completed")


def main():
    """Main function to run the model consistency test."""
    parser = argparse.ArgumentParser(description='Test model prediction consistency between bot and backtest environments')
    
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                      help=f'Path to the model file (default: {MODEL_PATH})')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='./model_comparison',
                      help='Directory to save output files')
    parser.add_argument('--no_export', action='store_true',
                      help='Skip exporting results to files')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = ModelConsistencyTester(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    tester.run_test(export=not args.no_export)


if __name__ == "__main__":
    main()