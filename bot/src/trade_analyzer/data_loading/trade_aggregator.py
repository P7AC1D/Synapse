"""
Trade aggregator module for combining entry, update, and exit events into complete trades.

This module reconstructs complete trade lifecycles from individual event logs,
enabling analysis of trade evolution from entry to exit.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class TradeAggregator:
    """Aggregates trade events into complete trade lifecycles."""
    
    def __init__(self):
        """Initialize the trade aggregator."""
        pass
        
    def reconstruct_trades(self, 
                          entries_df: pd.DataFrame, 
                          updates_df: pd.DataFrame, 
                          exits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct complete trades by linking entries, updates, and exits.
        
        Args:
            entries_df: DataFrame containing trade entry events
            updates_df: DataFrame containing trade update events
            exits_df: DataFrame containing trade exit events
            
        Returns:
            DataFrame containing complete trade lifecycles
        """
        if entries_df.empty:
            return pd.DataFrame()

        # Create a list to store complete trades
        complete_trades = []
        
        # Process each entry
        for _, entry in entries_df.iterrows():
            trade_data = {
                'trade_id': self._extract_trade_id_from_path(entry.get('file_path', '')),
                'entry_timestamp': entry.get('timestamp'),
                'action': entry.get('action'),
                'entry_price': entry.get('entry_price'),
                'lot_size': entry.get('lot_size'),
                'source': entry.get('source', 'unknown')
            }
            
            # Add all entry features (those starting with 'feature_')
            for col in entry.index:
                if col.startswith('feature_'):
                    trade_data[f'entry_{col}'] = entry[col]
            
            # Find matching updates
            if not updates_df.empty:
                matching_updates = self._find_matching_updates(entry, updates_df)
                if not matching_updates.empty:
                    # Add update statistics
                    trade_data['update_count'] = len(matching_updates)
                    trade_data['max_profit'] = matching_updates['profit_points'].max()
                    trade_data['min_profit'] = matching_updates['profit_points'].min()
                    trade_data['last_update_timestamp'] = matching_updates['timestamp'].max()
                    
                    # Find the update with max profit for feature extraction
                    max_profit_update = matching_updates.loc[matching_updates['profit_points'].idxmax()]
                    for col in max_profit_update.index:
                        if col.startswith('feature_'):
                            trade_data[f'max_profit_{col}'] = max_profit_update[col]
            else:
                trade_data['update_count'] = 0
            
            # Find matching exit
            if not exits_df.empty:
                matching_exit = self._find_matching_exit(entry, exits_df)
                if not matching_exit.empty:
                    # Add exit data
                    exit_row = matching_exit.iloc[0]
                    trade_data['exit_timestamp'] = exit_row.get('timestamp')
                    trade_data['exit_type'] = exit_row.get('exit_type')
                    trade_data['exit_price'] = exit_row.get('exit_price')
                    trade_data['final_profit'] = exit_row.get('final_profit_points')
                    trade_data['trade_duration'] = (
                        exit_row.get('timestamp') - entry.get('timestamp')
                    ).total_seconds() / 60  # Duration in minutes
                    
                    # Add exit features
                    for col in exit_row.index:
                        if col.startswith('feature_'):
                            trade_data[f'exit_{col}'] = exit_row[col]
                    
                    # Mark as complete
                    trade_data['is_complete'] = True
                else:
                    # No matching exit found
                    trade_data['is_complete'] = False
            else:
                # No exits dataframe
                trade_data['is_complete'] = False
            
            complete_trades.append(trade_data)
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(complete_trades)
        
        # Calculate additional metrics
        if not trades_df.empty and 'final_profit' in trades_df.columns:
            trades_df['is_profitable'] = trades_df['final_profit'] > 0
            
        return trades_df
    
    def _extract_trade_id_from_path(self, file_path: str) -> str:
        """
        Extract a unique trade ID from the file path.
        
        Args:
            file_path: Path to the trade event file
            
        Returns:
            Unique trade identifier based on filename
        """
        if not file_path:
            return 'unknown'
            
        # Extract filename without extension
        filename = file_path.split('/')[-1].split('\\')[-1]
        if filename.endswith('.json'):
            filename = filename[:-5]  # Remove .json extension
            
        return filename
    
    def _find_matching_updates(self, entry: pd.Series, updates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find updates that match a specific entry based on entry_data reference.
        
        Args:
            entry: Entry row
            updates_df: DataFrame containing update events
            
        Returns:
            DataFrame containing matching update events
        """
        # Extract timestamp to match against entry_data in updates
        entry_timestamp = entry.get('timestamp')
        if not entry_timestamp or updates_df.empty:
            return pd.DataFrame()
            
        # Check if updates have entry_data column
        if 'entry_data' not in updates_df.columns:
            return pd.DataFrame()
            
        # Extract entry timestamps from the nested entry_data
        updates_with_entry = updates_df[updates_df['entry_data'].notna()]
        
        # Match by comparing timestamps
        if entry_timestamp and not updates_with_entry.empty:
            matching_updates = []
            
            for _, update in updates_with_entry.iterrows():
                entry_data = update.get('entry_data', {})
                if entry_data and entry_data.get('timestamp') == str(entry_timestamp):
                    matching_updates.append(update)
            
            if matching_updates:
                return pd.DataFrame(matching_updates)
        
        return pd.DataFrame()
    
    def _find_matching_exit(self, entry: pd.Series, exits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find exit that matches a specific entry based on entry_data reference.
        
        Args:
            entry: Entry row
            exits_df: DataFrame containing exit events
            
        Returns:
            DataFrame containing matching exit event
        """
        # Extract timestamp to match against entry_data in exits
        entry_timestamp = entry.get('timestamp')
        if not entry_timestamp or exits_df.empty:
            return pd.DataFrame()
            
        # Check if exits have entry_data column
        if 'entry_data' not in exits_df.columns:
            return pd.DataFrame()
            
        # Extract entry timestamps from the nested entry_data
        exits_with_entry = exits_df[exits_df['entry_data'].notna()]
        
        # Match by comparing timestamps
        if entry_timestamp and not exits_with_entry.empty:
            matching_exits = []
            
            for _, exit_row in exits_with_entry.iterrows():
                entry_data = exit_row.get('entry_data', {})
                if entry_data and entry_data.get('timestamp') == str(entry_timestamp):
                    matching_exits.append(exit_row)
            
            if matching_exits:
                return pd.DataFrame(matching_exits)
        
        return pd.DataFrame()
        
    def compute_trade_evolution(self, 
                              entry: pd.Series, 
                              updates: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the evolution of a trade over time based on updates.
        
        Args:
            entry: Entry event data
            updates: DataFrame containing update events for this trade
            
        Returns:
            DataFrame with trade evolution data
        """
        if updates.empty:
            return pd.DataFrame()
            
        # Sort updates by timestamp
        updates_sorted = updates.sort_values('timestamp')
        
        # Create evolution dataframe
        evolution = updates_sorted.copy()
        
        # Add entry price for comparison
        entry_price = entry.get('entry_price')
        if entry_price:
            evolution['entry_price'] = entry_price
            evolution['price_change'] = evolution['price'] - entry_price
            
            # Calculate percentage change
            if entry.get('action') == 'buy':
                evolution['price_change_pct'] = evolution['price_change'] / entry_price * 100
            else:  # Sell
                evolution['price_change_pct'] = -evolution['price_change'] / entry_price * 100
        
        return evolution