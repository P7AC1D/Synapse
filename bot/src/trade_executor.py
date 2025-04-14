import logging
from typing import Dict, Any, Optional, Tuple

import MetaTrader5 as mt5
from mt5_connector import MT5Connector
from config import MT5_SYMBOL, MT5_COMMENT, STOP_LOSS_PIPS

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
        
    def calculate_stop_loss(self, entry_price: float, trade_type: str) -> float:
        """
        Calculate stop loss price based on pips from entry price.
        
        Args:
            entry_price: Entry price for the trade
            trade_type: Either 'buy' or 'sell'
            
        Returns:
            float: Stop loss price
        """
        try:
            # Get point value for the symbol
            point = mt5.symbol_info(MT5_SYMBOL).point
            digits = mt5.symbol_info(MT5_SYMBOL).digits
            
            # For XAUUSD, 1 pip = 0.1 points (need to multiply by 10)
            pip_value = point * 10 if 'XAU' in MT5_SYMBOL else point
            
            # Calculate stop loss price
            if trade_type == 'buy':
                sl_price = entry_price - (STOP_LOSS_PIPS * pip_value)
            else:  # sell
                sl_price = entry_price + (STOP_LOSS_PIPS * pip_value)
                
            # Round to symbol digits
            sl_price = round(sl_price, digits)
            
            self.logger.debug(
                f"Stop Loss calculation: Entry: {entry_price:.{digits}f} | "
                f"Type: {trade_type} | SL: {sl_price:.{digits}f} | "
                f"Distance: {STOP_LOSS_PIPS} pips"
            )
            return sl_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0  # Return 0 on error to indicate failure

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
                min_lot,
                min(
                    max_lot,
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
                
            # Prevent new trades if we already have a position
            if positions and action in [1, 2]:
                self.logger.info(f"Trade rejected: Position already exists ({len(positions)} active positions)")
                return True
                
            # Handle close action
            if action == 3:
                if not positions:
                    self.logger.debug("No positions to close")
                    return True
                    
                # Close all positions
                for pos in positions:
                    success = self.mt5.close_position(pos.ticket)
                    if not success:
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
            
            # Calculate stop loss
            trade_type = 'buy' if action == 1 else 'sell'
            stop_loss = self.calculate_stop_loss(current_price, trade_type)
            if stop_loss == 0.0:
                self.logger.error("Failed to calculate stop loss")
                return False

            # Execute the trade with stop loss
            success = self.mt5.open_trade(
                symbol=MT5_SYMBOL,
                lot=lot_size,
                price=current_price,
                order_type=trade_type,
                filling_type=filling_type,
                sl=stop_loss
            )
            
            if success:
                self.last_lot_size = lot_size  # Store last used lot size
                # Calculate stop loss distance in points
                sl_points = abs(current_price - stop_loss)
                self.logger.info(
                    f"Trade executed: {trade_type.upper()} "
                    f"{lot_size:.2f} lots @ {current_price:.5f} "
                    f"(SL: {stop_loss:.5f}, {STOP_LOSS_PIPS} pips)"
                )
            else:
                self.logger.error("Trade execution failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error executing trade: {e}")
            return False
