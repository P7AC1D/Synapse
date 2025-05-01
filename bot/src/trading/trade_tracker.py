"""Trade tracking system for logging and analyzing trade lifecycle."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class TradeTracker:
    """Tracks and logs trade lifecycle including feature states."""
    
    def __init__(self, log_path: str):
        """
        Initialize trade tracker.
        
        Args:
            log_path: Base path for log files
        """
        self.log_path = log_path
        self.current_trade = None
        
    def _ensure_directory(self, path: str) -> None:
        """Ensure directory exists, create if not."""
        os.makedirs(path, exist_ok=True)
        
    def _get_date_dir(self) -> str:
        """Get date-based directory path."""
        date = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.log_path, date)
        
    def _save_trade_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Save trade event data to JSON file.
        
        Args:
            event_type: Type of event ('entry', 'update', 'exit')
            data: Event data to log
        """
        # Create date and event type directories
        date_dir = self._get_date_dir()
        event_dir = os.path.join(date_dir, f"{event_type}s")
        self._ensure_directory(event_dir)
        
        # Generate filename with timestamp and trade ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{timestamp}.json"
        
        # Write data to JSON file
        filepath = os.path.join(event_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
    def log_trade_entry(self, action: str, features: dict, price: float, lot_size: float) -> None:
        """
        Log entry conditions and features.
        
        Args:
            action: Trade action ('buy' or 'sell')
            features: Current feature values
            price: Entry price
            lot_size: Position size in lots
        """
        entry_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'entry_price': price,
            'lot_size': lot_size,
            'entry_features': features
        }
        self.current_trade = entry_data
        self._save_trade_log('entry', entry_data)
        
    def log_trade_update(self, features: dict, current_price: float, profit_points: float) -> None:
        """
        Log feature evolution during trade.
        
        Args:
            features: Current feature values
            current_price: Current market price
            profit_points: Current unrealized profit/loss
        """
        if not self.current_trade:
            return
            
        update_data = {
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'profit_points': profit_points,
            'features': features,
            'entry_data': self.current_trade  # Include original entry data for reference
        }
        self._save_trade_log('update', update_data)
        
    def log_trade_exit(self, exit_type: str, exit_price: float, profit_points: float, features: dict) -> None:
        """
        Log exit conditions and final state.
        
        Args:
            exit_type: Type of exit ('stop_loss', 'take_profit', or 'model_close')
            exit_price: Exit price
            profit_points: Final profit/loss
            features: Feature values at exit
        """
        if not self.current_trade:
            return
            
        exit_data = {
            'timestamp': datetime.now().isoformat(),
            'entry_data': self.current_trade,
            'exit_type': exit_type,
            'exit_price': exit_price,
            'final_profit_points': profit_points,
            'exit_features': features
        }
        self._save_trade_log('exit', exit_data)
        self.current_trade = None  # Reset current trade after exit
