import logging
from typing import Dict, Any, Optional, Tuple

import MetaTrader5 as mt5
from mt5_connector import MT5Connector
from config import MT5_COMMENT

class TradeExecutor:
    """Class for executing trades based on model predictions."""
    
    def __init__(self, mt5: MT5Connector, symbol: str, balance_per_lot: float = 1000.0, 
                 stop_loss_pips: float = 2500.0, take_profit_pips: float = 2500.0):
        """
        Initialize the trade executor.
        
        Args:
            mt5: MT5 connector instance for trade execution
            symbol: Trading symbol to use for trade execution
            balance_per_lot: Account balance required per 0.01 lot (default: 1000.0)
            stop_loss_pips: Stop loss in pips for all trades (default: 2500.0)
            take_profit_pips: Take profit in pips for all trades (default: 2500.0)
        """
        self.logger = logging.getLogger(__name__)
        self.mt5 = mt5
        self.symbol = symbol  # Store the symbol as instance variable
        self.balance_per_lot = balance_per_lot
        self.last_lot_size = None  # Track last used lot size for position info
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        
    def calculate_stop_loss(self, entry_price: float, trade_type: str, symbol: str) -> float:
        """
        Calculate stop loss price based on pips from entry price.
        
        Args:
            entry_price: Entry price for the trade
            trade_type: Either 'buy' or 'sell'
            symbol: Trading symbol (e.g. XAUUSDm, USTECm)
            
        Returns:
            float: Stop loss price
        """
        try:
            # Get symbol info from connector
            _, _, _, _, point, digits = self.mt5.get_symbol_info(symbol)
            
            # Calculate stop loss directly using points instead of pips
            # This simplifies the calculation by removing the separate pip concept
            sl_points = self.stop_loss_pips * point
            
            # Calculate stop loss price based on trade type
            if trade_type == 'buy':
                sl_price = entry_price - sl_points
            else:  # sell
                sl_price = entry_price + sl_points
                
            # Round to symbol digits
            sl_price = round(sl_price, digits)
            
            self.logger.debug(
                f"Stop Loss calculation: Entry: {entry_price:.{digits}f} | "
                f"Type: {trade_type} | SL: {sl_price:.{digits}f} | "
                f"Distance: {self.stop_loss_pips} points"
            )
            return sl_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0  # Return 0 on error to indicate failure

    def calculate_take_profit(self, entry_price: float, trade_type: str, symbol: str) -> float:
        """
        Calculate take profit price based on pips from entry price.
        
        Args:
            entry_price: Entry price for the trade
            trade_type: Either 'buy' or 'sell'
            symbol: Trading symbol (e.g. XAUUSDm, USTECm)
            
        Returns:
            float: Take profit price
        """
        try:
            # Get symbol info from connector
            _, _, _, _, point, digits = self.mt5.get_symbol_info(symbol)
            
            # Calculate take profit directly using points
            tp_points = self.take_profit_pips * point
            
            # Calculate take profit price based on trade type (opposite direction from stop loss)
            if trade_type == 'buy':
                tp_price = entry_price + tp_points
            else:  # sell
                tp_price = entry_price - tp_points
                
            # Round to symbol digits
            tp_price = round(tp_price, digits)
            
            self.logger.debug(
                f"Take Profit calculation: Entry: {entry_price:.{digits}f} | "
                f"Type: {trade_type} | TP: {tp_price:.{digits}f} | "
                f"Distance: {self.take_profit_pips} points"
            )
            return tp_price
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
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
            # Get symbol trading information
            _, min_lot, max_lot, volume_step = self.mt5.get_symbol_info(self.symbol)[:4]
            
            # Calculate raw lot size
            raw_lot_size = (account_balance / self.balance_per_lot) * volume_step
            
            # Round to nearest volume step
            steps = round(raw_lot_size / volume_step)
            lot_size = steps * volume_step
            
            # Ensure within min/max bounds
            lot_size = max(min_lot, min(max_lot, lot_size))
            
            self.logger.debug(
                f"Lot size calculation: Balance: {account_balance:.2f} | "
                f"Balance per lot: {self.balance_per_lot:.2f} | "
                f"Raw size: {raw_lot_size:.3f} | "
                f"Steps: {steps} | "
                f"Final lot size: {lot_size:.2f} lots"
            )
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return min_lot  # Return minimum lot size on error
        
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
            positions = self.mt5.get_open_positions(self.symbol, MT5_COMMENT)
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
            
            # Get current price
            if action == 1:  # Buy
                current_price = self.mt5.get_symbol_info_tick(self.symbol)[1]  # Ask
            else:  # Sell
                current_price = self.mt5.get_symbol_info_tick(self.symbol)[0]  # Bid
                
            if current_price is None:
                self.logger.error("Failed to get current price")
                return False
            
            # Calculate position size
            lot_size = self.calculate_position_size(self.mt5.get_account_balance())
            
            # Get filling type
            filling_type = self.mt5.check_filling_type(
                self.symbol,
                'buy' if action == 1 else 'sell'
            )
            
            # Calculate stop loss and take profit
            trade_type = 'buy' if action == 1 else 'sell'
            stop_loss = self.calculate_stop_loss(current_price, trade_type, self.symbol)
            take_profit = self.calculate_take_profit(current_price, trade_type, self.symbol)
            
            if stop_loss == 0.0:
                self.logger.error("Failed to calculate stop loss")
                return False
                
            if take_profit == 0.0:
                self.logger.error("Failed to calculate take profit")
                return False

            # Execute the trade with stop loss and take profit
            success = self.mt5.open_trade(
                symbol=self.symbol,
                lot=lot_size,
                price=current_price,
                order_type=trade_type,
                filling_type=filling_type,
                sl=stop_loss,
                tp=take_profit
            )
            
            if success:
                self.last_lot_size = lot_size  # Store last used lot size
                # Calculate stop loss and take profit distances in points
                sl_points = abs(current_price - stop_loss)
                tp_points = abs(current_price - take_profit)
                self.logger.info(
                    f"Trade executed: {trade_type.upper()} "
                    f"{lot_size:.2f} lots @ {current_price:.5f} "
                    f"(SL: {stop_loss:.5f}, {self.stop_loss_pips} points | "
                    f"TP: {take_profit:.5f}, {self.take_profit_pips} points)"
                )
            else:
                self.logger.error("Trade execution failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error executing trade: {e}")
            return False
