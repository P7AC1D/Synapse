import logging
from typing import Dict, Any, Optional

import numpy as np
import MetaTrader5 as mt5
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
        
    def calculate_grid_position_size(self, entry_price: float, grid_size_pips: float,
                                   account_balance: float, risk_multiplier: float = 1.0) -> float:
        """
        Calculate position size based on grid parameters.
        
        Args:
            entry_price: Entry price for the position
            grid_size_pips: Grid size in pips (from model prediction)
            account_balance: Current account balance
            risk_multiplier: Risk multiplier for position pyramiding
            
        Returns:
            float: Position size in lots
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
            
            # Convert grid size from pips to price points
            stop_distance = grid_size_pips * self.mt5.get_point()
            
            # Calculate base position size using grid size
            lot_size = risk_in_usd / (stop_distance * contract_size)
            
            # Cap the lot size to the minimum and maximum values
            lot_size = round(max(min_lot, min(lot_size, max_lot)), 2)
            
            self.logger.debug(
                f"Lot size calculation: Contract size: {contract_size} | "
                f"USDZAR: {usd_zar_bid:.2f} | Risk %: {RISK_PERCENTAGE}% | "
                f"Risk: R{risk_amount:.2f} | Risk USD: ${risk_in_usd:.2f} | "
                f"Grid Size: {grid_size_pips:.5f} | Lot Size: {lot_size:.2f}"
            )
            return lot_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum lot size on error
        
    def execute_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Execute a grid-based trade based on model prediction.
        
        Args:
            prediction: Dictionary containing model prediction details including grid parameters
        
        Returns:
            bool: True if trade executed successfully, False otherwise
        """
        try:
            position = prediction['position']  # -1 for sell, 0 for hold, 1 for buy
            grid_size_pips = prediction['grid_size_pips']
            grid_multiplier = prediction.get('grid_multiplier', 1.0)
            
            if position == 0:
                self.logger.debug("Hold signal - no trade execution")
                return True
            
            # Get symbol info
            symbol_info = mt5.symbol_info(MT5_SYMBOL)
            if symbol_info is None:
                self.logger.error("Failed to get symbol info")
                return False

            # Get current price
            if position == 1:  # Buy
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[1]  # Ask
            else:  # Sell
                current_price = self.mt5.get_symbol_info_tick(MT5_SYMBOL)[0]  # Bid
                
            if current_price is None:
                self.logger.error("Failed to get current price")
                return False

            # Check existing positions
            positions = self.mt5.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
            if positions is None:
                self.logger.error("Failed to get open positions")
                return False
                
            # Split positions by direction
            long_positions = [p for p in positions if p.type == 0]  # 0 = buy
            short_positions = [p for p in positions if p.type == 1]  # 1 = sell
            current_positions = long_positions if position == 1 else short_positions
            
            # Grid position management
            if current_positions:
                # Calculate average position price
                avg_price = sum(p.price_open * p.volume for p in current_positions) / sum(p.volume for p in current_positions)
                price_diff = abs(current_price - avg_price)
                
                # Only add position if price moved beyond grid size
                if price_diff < grid_size_pips * symbol_info.point:
                    self.logger.debug(f"Price {price_diff:.5f} within grid size {grid_size_pips * symbol_info.point:.5f} - no new position")
                    return True
                
                # Adjust risk for pyramiding
                risk_multiplier = max(0.2, 1.0 / (len(current_positions) + 1))
            else:
                risk_multiplier = 1.0
            
            # Calculate lot size based on grid parameters
            lot_size = self.calculate_grid_position_size(
                entry_price=current_price,
                grid_size_pips=grid_size_pips,
                account_balance=self.mt5.get_account_balance(),
                risk_multiplier=risk_multiplier
            )
            
            # Get filling type
            filling_type = self.mt5.check_filling_type(
                MT5_SYMBOL, 
                'buy' if position == 1 else 'sell'
            )
            
            # Set grid boundaries for trade management
            grid_points = grid_size_pips * symbol_info.point
            if position == 1:  # Buy
                sl_price = current_price - grid_points
                tp_price = current_price + grid_points
            else:  # Sell
                sl_price = current_price + grid_points
                tp_price = current_price - grid_points
            
            # Execute the trade
            success = self.mt5.open_trade(
                symbol=MT5_SYMBOL,
                lot=lot_size,
                price=current_price,
                sl_price=sl_price,
                tp_price=tp_price,
                order_type='buy' if position == 1 else 'sell',
                filling_type=filling_type
            )
            
            if success:
                self.logger.info(
                    f"Grid trade executed: {'BUY' if position == 1 else 'SELL'} "
                    f"{lot_size:.2f} lots @ {current_price:.5f} | "
                    f"Grid Size: {grid_size_pips:.1f} pips | "
                    f"Grid Multiplier: {grid_multiplier:.2f} | "
                    f"Risk: {RISK_PERCENTAGE*risk_multiplier:.1f}%"
                )
            else:
                self.logger.error("Trade execution failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"Error executing trade: {e}")
            return False
