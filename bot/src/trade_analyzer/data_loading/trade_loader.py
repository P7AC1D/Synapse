"""
Trade loader module for loading and parsing trade logs.

This module handles loading trade log entries from JSON files created
by the TradeTracker during both backtest and live trading.
"""

import os
import json
import glob
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class TradeLoader:
    """Loads and parses trade logs from JSON files."""
    
    def __init__(self):
        """Initialize the trade loader."""
        self.entry_data = {}
        self.update_data = {}
        self.exit_data = {}
        
    def load_trade_logs(self, log_path: str, source_type: str = "live") -> Dict[str, pd.DataFrame]:
        """
        Load all trade logs from a specific directory.
        
        Args:
            log_path: Path to the directory containing trade logs
            source_type: Tag to identify the source ('live' or 'backtest')
            
        Returns:
            Dictionary containing DataFrames for entries, updates, and exits
        """
        print(f"Loading {source_type} trade logs from {log_path}...")
        
        # Dictionary to store DataFrames
        dfs = {
            'entries': [],
            'updates': [],
            'exits': []
        }
        
        # Get all date directories
        date_dirs = [d for d in glob.glob(os.path.join(log_path, "*")) 
                     if os.path.isdir(d)]
        
        if not date_dirs:
            print(f"No date directories found in {log_path}")
            return {k: pd.DataFrame() for k in dfs.keys()}
            
        for date_dir in date_dirs:
            date_name = os.path.basename(date_dir)
            
            # Process entries
            entries_dir = os.path.join(date_dir, "entrys")  
            if os.path.exists(entries_dir):
                entry_files = glob.glob(os.path.join(entries_dir, "*.json"))
                for file in entry_files:
                    try:
                        with open(file, 'r') as f:
                            entry_data = json.load(f)
                            entry_data['file_path'] = file
                            entry_data['date'] = date_name
                            entry_data['source'] = source_type
                            dfs['entries'].append(entry_data)
                    except Exception as e:
                        print(f"Error loading entry file {file}: {e}")
            
            # Process updates
            updates_dir = os.path.join(date_dir, "updates")
            if os.path.exists(updates_dir):
                update_files = glob.glob(os.path.join(updates_dir, "*.json"))
                for file in update_files:
                    try:
                        with open(file, 'r') as f:
                            update_data = json.load(f)
                            update_data['file_path'] = file
                            update_data['date'] = date_name
                            update_data['source'] = source_type
                            dfs['updates'].append(update_data)
                    except Exception as e:
                        print(f"Error loading update file {file}: {e}")
            
            # Process exits
            exits_dir = os.path.join(date_dir, "exits")
            if os.path.exists(exits_dir):
                exit_files = glob.glob(os.path.join(exits_dir, "*.json"))
                for file in exit_files:
                    try:
                        with open(file, 'r') as f:
                            exit_data = json.load(f)
                            exit_data['file_path'] = file
                            exit_data['date'] = date_name
                            exit_data['source'] = source_type
                            dfs['exits'].append(exit_data)
                    except Exception as e:
                        print(f"Error loading exit file {file}: {e}")
        
        # Convert to DataFrames
        result = {}
        for key, data_list in dfs.items():
            if data_list:
                result[key] = pd.DataFrame(data_list)
                print(f"Loaded {len(data_list)} {key} records")
            else:
                result[key] = pd.DataFrame()
                print(f"No {key} records found")
        
        return result

    def extract_features(self, df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
        """
        Extract features from a nested JSON column into separate columns.
        
        Args:
            df: DataFrame containing the nested features
            feature_column: Column name containing the feature dictionary
            
        Returns:
            DataFrame with features extracted into separate columns
        """
        if df.empty or feature_column not in df.columns:
            return df
            
        # Normalize the JSON in the feature column
        features_df = pd.json_normalize(df[feature_column])
        
        # Prefix column names with 'feature_'
        features_df = features_df.add_prefix('feature_')
        
        # Join with original DataFrame
        result = pd.concat([df.drop(columns=[feature_column]), features_df], axis=1)
        
        return result
    
    def process_trade_data(self, trade_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process trade data to extract features and normalize timestamps.
        
        Args:
            trade_data: Dictionary containing DataFrames for entries, updates, and exits
            
        Returns:
            Processed DataFrames
        """
        processed_data = {}
        
        # Process entries
        if 'entries' in trade_data and not trade_data['entries'].empty:
            entries_df = trade_data['entries'].copy()
            # Convert timestamps
            if 'timestamp' in entries_df:
                entries_df['timestamp'] = pd.to_datetime(entries_df['timestamp'])
            # Extract entry features
            if 'entry_features' in entries_df:
                entries_df = self.extract_features(entries_df, 'entry_features')
            processed_data['entries'] = entries_df
        else:
            processed_data['entries'] = pd.DataFrame()
            
        # Process updates
        if 'updates' in trade_data and not trade_data['updates'].empty:
            updates_df = trade_data['updates'].copy()
            # Convert timestamps
            if 'timestamp' in updates_df:
                updates_df['timestamp'] = pd.to_datetime(updates_df['timestamp'])
            # Extract update features
            if 'features' in updates_df:
                updates_df = self.extract_features(updates_df, 'features')
            processed_data['updates'] = updates_df
        else:
            processed_data['updates'] = pd.DataFrame()
            
        # Process exits
        if 'exits' in trade_data and not trade_data['exits'].empty:
            exits_df = trade_data['exits'].copy()
            # Convert timestamps
            if 'timestamp' in exits_df:
                exits_df['timestamp'] = pd.to_datetime(exits_df['timestamp'])
            # Extract exit features
            if 'exit_features' in exits_df:
                exits_df = self.extract_features(exits_df, 'exit_features')
            processed_data['exits'] = exits_df
        else:
            processed_data['exits'] = pd.DataFrame()
            
        return processed_data