import logging
from typing import Dict, Any, Optional

import MetaTrader5 as mt5
from mt5_connector import MT5Connector
from config import MT5_SYMBOL, MT5_COMMENT

class TradeExecutor:
    """Class for executing trades based on model predictions."""
    
    def __init__(self, mt5: MT5Connector, balance_per_lot: float = 1000.0):
        """
        Initialize the trade executor.
        
        Args:
            mt5: MT5 connector instance for trade execution
            balance_per_lot: Account balance required per 0.01 lot (default: 1000.0)
        """
        self.logger = logging.getLogger(__name__)
        self.mt5 = mt5
        self.balance_per_lot = balance_per_lot
        self.last_lot_size = None  # Track last used lot size for position info
        
    def calculate_position_size(self, account_balance: float) -> float:
        """
        Calculate position size using the same method as the backtest environment.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            float: Position size in lots
        """
        try:
            # Get symbol trading information for min/max lots
            _, min_lot, max_lot = self.mt5.get_symbol_info(MT5_SYMBOL)
            
            # Calculate lot size exactly as done in backtest environment
            lot_size = max(
                min_lot,  # MIN_LOTS (0.01)
                min(
                    max_lot,  # MAX_LOTS (50.0)
                    round((account_balance / self.balance_per_lot) * min_lot, 2)
                )
            )
            
            self.logger.debug(
                f"Lot size calculation: Balance: {account_balance:.2f} | "
                f"Balance per lot: {self.balance_per_lot:.2f} | "
                f"Lot size: {lot_size:.2f} lots"
            )
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum lot size on error
        
    def execute_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Execute trade based on model prediction.
        
        Args:
            prediction: Dictionary containing model prediction with action and description
        
        Returns:
            bool: True if trade executed successfully, False otherwise
        """
        try:
            action = prediction['action']  # 0=hold, 1=buy, 2=sell, 3=close
            
            if action == 0:  # Hold
                self.logger.debug("Hold signal - no trade execution")
                return True
                
            # Get current positions
            positions = self.mt5.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
            if positions is None:
                self.logger.error("Failed to get open positions")
                return False
                
            # Handle close action
            if action == 3:
                if not positions:
                    self.logger.debug("No positions to close")
                    return True
                    
                # Close all positions
                for pos in positions:
                    success = self.mt5.close_position(pos.ticket)
                    if success:
                        self.logger.info(f"Closed position {pos.ticket}")
                    else:
                        self.logger.error(f"Failed to close position {pos.ticket}")
                        return False
                return True
            
            # Get symbol info for new trades
            symbol_info = mt5.symbol_info(MT5_SYMBOL)
            if symbol_info is None:
                self.logger.error("Failed to get symbol info")
                return False
                
            # Get current price
            if action == 1:  # Buy
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[1]  # Ask
            else:  # Sell
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[0]  # Bid
                
            if current_price is None:
                self.logger.error("Failed to get current price")
                return False
            
            # Calculate position size
            lot_size = self.calculate_position_size(self.mt5.get_account_balance())
            
            # Get filling type
            filling_type = self.mt5.check_filling_type(
                MT5_SYMBOL,
                'buy' if action == 1 else 'sell'
            )
            
            # Execute the trade without SL/TP - let model control position closure
            success = self.mt5.open_trade(
                symbol=MT5_SYMBOL,
                lot=lot_size,
                price=current_price,
                order_type='buy' if action == 1 else 'sell',
                filling_type=filling_type
            )
            
            if success:
                self.last_lot_size = lot_size  # Store last used lot size
                self.logger.info(
                    f"Trade executed: {'BUY' if action == 1 else 'SELL'} "
                    f"{lot_size:.2f} lots @ {current_price:.5f} "
                    f"(No SL/TP - model controlled)"
                )
            else:
                self.logger.error("Trade execution failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error executing trade: {e}")
            return False
