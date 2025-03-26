import logging
from typing import Dict, Any, Optional

import numpy as np
from mt5_connector import MT5Connector
from config import MT5_SYMBOL, MT5_BASE_SYMBOL, RISK_PERCENTAGE, MT5_COMMENT

class TradeExecutor:
    """Class for executing trades based on model predictions."""
    
    def __init__(self, mt5: MT5Connector):
        """
        Initialize the trade executor.
        
        Args:
            mt5: MT5 connector instance for trade execution
        """
        self.logger = logging.getLogger(__name__)
        self.mt5 = mt5
        
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              account_balance: float, risk_multiplier: float = 1.0) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            risk_multiplier: Multiplier for risk amount (e.g., 0.4 for second position)
            
        Returns:
            float: Calculated position size in lots
        """
        try:
            # Get symbol trading information
            contract_size, min_lot, max_lot = self.mt5.get_symbol_info(MT5_SYMBOL)
            
            # Get USDZAR price for conversion
            usd_zar_bid, _ = self.mt5.get_symbol_info_tick(MT5_BASE_SYMBOL)
            
            # Calculate risk amount based on percentage from config
            base_risk = account_balance * (RISK_PERCENTAGE / 100)
            risk_amount = base_risk * risk_multiplier
            risk_in_usd = risk_amount / usd_zar_bid
            
            # Calculate stop-loss distance in price points
            stop_loss_distance = abs(entry_price - stop_loss_price)
            
            # Calculate position size
            lot_size = risk_in_usd / (stop_loss_distance * contract_size)
            
            # Cap the lot size to the minimum and maximum values
            lot_size = round(max(min_lot, min(lot_size, max_lot)), 2)
            
            self.logger.debug(
                f"Lot size calculation: Contract size: {contract_size} | "
                f"USDZAR: {usd_zar_bid:.2f} | Risk %: {RISK_PERCENTAGE}% | "
                f"Risk: R{risk_amount:.2f} | Risk USD: ${risk_in_usd:.2f} | "
                f"SL Distance: {stop_loss_distance:.5f} | Lot Size: {lot_size:.2f}"
            )
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum lot size on error
        
    def execute_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Execute a trade based on model prediction.
        
        Args:
            prediction: Dictionary containing model prediction details
        
        Returns:
            bool: True if trade executed successfully, False otherwise
        """
        try:
            position = prediction['position']  # -1 for sell, 0 for hold, 1 for buy
            sl_points = prediction['sl_points']
            tp_points = prediction['tp_points']
            atr = prediction['atr']
            
            if position == 0:
                self.logger.debug("Hold signal - no trade execution")
                return True
            
            # Get current price
            if position == 1:
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[1]  # Ask for buy
            else:
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[0]  # Bid for sell
                
            if current_price is None:
                self.logger.error("Failed to get current price")
                return False
                
            # Set SL/TP based on points
            if position == 1:  # Buy
                sl_price = current_price - sl_points
                tp_price = current_price + tp_points
            else:  # Sell
                sl_price = current_price + sl_points
                tp_price = current_price - tp_points
            
            # Check existing positions
            positions = self.mt5.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
            if positions is None:
                self.logger.error("Failed to get open positions")
                return False
                
            # Count positions by direction
            long_positions = sum(1 for p in positions if p.type == 0)  # 0 = buy
            short_positions = sum(1 for p in positions if p.type == 1)  # 1 = sell
            
            # Apply position management rules
            if position == 1 and long_positions >= 1:
                self.logger.debug("Max long positions reached")
                return True
            if position == -1 and short_positions >= 1:
                self.logger.debug("Max short positions reached")
                return True
                
            # Calculate lot size with risk management
            total_positions = len(positions)
            if total_positions >= 2:
                self.logger.debug("Max total positions reached")
                return True
                
            # Use 40% risk for second position
            risk_multiplier = 0.4 if total_positions == 1 else 1.0
            
            lot_size = self.calculate_position_size(
                entry_price=current_price,
                stop_loss_price=sl_price,
                account_balance=self.mt5.get_account_balance(),
                risk_multiplier=risk_multiplier
            )
            
            # Get filling type
            filling_type = self.mt5.check_filling_type('buy' if position == 1 else 'sell')
            
            # Execute the trade
            success = self.mt5.open_trade(
                lot=lot_size,
                price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                order_type='buy' if position == 1 else 'sell',
                filling_type=filling_type
            )
            
            if success:
                self.logger.info(
                    f"Trade executed: {'BUY' if position == 1 else 'SELL'} "
                    f"{lot_size:.2f} lots @ {current_price:.2f} | "
                    f"SL: {sl_price:.2f} TP: {tp_price:.2f} | "
                    f"RRR: {tp_points/sl_points:.2f} | "
                    f"Risk: {RISK_PERCENTAGE*risk_multiplier:.1f}%"
                )
            else:
                self.logger.error("Trade execution failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error executing trade: {e}")
            return False
