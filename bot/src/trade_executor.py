"""Module for executing trades based on model predictions."""

import logging
from typing import Dict, Any, Optional, Union, Tuple, List
from enum import Enum

from config import (
    MT5_SYMBOL,
    MT5_COMMENT,
    MT5_BASE_SYMBOL,
    MAX_SPREAD,
    RISK_PERCENTAGE
)
from mt5_connector import MT5Connector


class PositionType(Enum):
    """Enum for position types."""
    BUY = 0
    SELL = 1


class TradeExecutor:
    """Executes trades based on model predictions."""

    def __init__(self, mt5_connector: MT5Connector):
        """
        Initialize the trade executor.
        
        Args:
            mt5_connector: MetaTrader 5 connector instance
        """
        self.mt5_connector = mt5_connector
        self.logger = logging.getLogger(__name__)

    def apply_trailing_stops(self, trailing_points: float) -> None:
        """
        Apply trailing stops to open positions.
        
        Args:
            trailing_points: Points to trail the stop loss by
        """
        positions = self.mt5_connector.get_open_positions(MT5_SYMBOL, MT5_COMMENT)

        if positions is None or len(positions) == 0:
            return

        for position in positions:
            if position.type == PositionType.BUY.value:
                new_sl = position.price_current - trailing_points
                if new_sl > position.sl and new_sl >= position.price_open:
                    self.mt5_connector.modify_stop_loss(position.ticket, new_sl, position.tp)

            elif position.type == PositionType.SELL.value:
                new_sl = position.price_current + trailing_points
                if new_sl < position.sl and new_sl <= position.price_open:
                    self.mt5_connector.modify_stop_loss(position.ticket, new_sl, position.tp)

    def open_trade(self, order_type: str, risk_reward_ratio: float, 
                  atr: float) -> bool:
        """
        Open a new trade based on the specified parameters.
        
        Args:
            order_type: Type of order ('buy' or 'sell')
            risk_reward_ratio: Risk to reward ratio
            atr: Average True Range value used for SL/TP calculation
            
        Returns:
            bool: True if trade was executed successfully, False otherwise
        """
        account_balance = self.mt5_connector.get_account_balance()
        filling_type = self.mt5_connector.check_filling_type(order_type)

        bid, ask = self.mt5_connector.get_symbol_info_tick(MT5_SYMBOL)
        self.logger.debug(f"Bid: {bid} | Ask: {ask}")

        # Check if spread is acceptable
        spread = ask - bid
        if spread >= MAX_SPREAD:
            self.logger.warning(
                f"Trade rejected. Spread too high. Spread: {spread} | Max: {MAX_SPREAD}"
            )
            return False
        
        # Set entry price based on order type
        price = ask if order_type == 'buy' else bid

        # Calculate stop loss and take profit prices
        sl_price = self._calculate_stop_loss_price(order_type, price, atr, spread)
        tp_price = self._calculate_take_profit_price(order_type, price, risk_reward_ratio, atr, spread)

        # Calculate position size based on risk
        lot = self.calculate_position_size(price, sl_price, account_balance)

        # Execute the trade
        if self.mt5_connector.open_trade(lot, price, sl_price, tp_price, order_type, filling_type):
            self.logger.info(
                f"Trade executed successfully. Balance: {account_balance} | "
                f"Order: {order_type} | Lot: {lot} | Price: {price} | "
                f"SL: {round(sl_price, 3)} | TP: {round(tp_price, 3)}"
            )
            return True
        return False
    
    def close_position(self, position_index: int) -> bool:
        """
        Close a position by its index in the open positions list.
        
        Args:
            position_index: Index of the position to close
            
        Returns:
            bool: True if the position was closed, False otherwise
        """
        open_positions = self.mt5_connector.get_open_positions(MT5_SYMBOL, MT5_COMMENT)
        if open_positions is None or len(open_positions) == 0:
            self.logger.warning("No open positions to close")
            return False
        
        self.logger.debug(f"Open positions: {len(open_positions)} | Position to close: {position_index}")

        if position_index < len(open_positions):
            position_to_close = open_positions[position_index]
            result = self.mt5_connector.close_position(position_to_close.ticket)
            self.logger.info(f"Position {position_to_close.ticket} closed: {result}")
            return result
        else:
            self.logger.warning(f"Invalid position index: {position_index}")
            return False

    def calculate_position_size(self, entry_price: float, 
                              stop_loss_price: float, 
                              account_balance: float) -> float:
        """
        Calculate position size based on risk management parameters.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            
        Returns:
            float: Calculated position size in lots
        """
        contract_size, min_lot, max_lot = self.mt5_connector.get_symbol_info(MT5_SYMBOL)
        
        # Get USDZAR price for conversion
        usd_zar_bid, _ = self.mt5_connector.get_symbol_info_tick(MT5_BASE_SYMBOL)

        # Calculate risk amount based on percentage
        risk_amount = account_balance * (RISK_PERCENTAGE / 100)
        risk_in_usd = risk_amount / usd_zar_bid

        # Calculate stop-loss distance in price points
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        # Calculate position size
        lot = risk_in_usd / (stop_loss_distance * contract_size)

        # Cap the lot size to the minimum and maximum values
        lot = round(max(min_lot, min(lot, max_lot)), 2)

        self.logger.debug(
            f"Lot size calculation: Contract size: {contract_size} | "
            f"USDZAR: {usd_zar_bid:.2f} | Risk: R{risk_amount:.2f} | "
            f"Risk USD: ${risk_in_usd:.2f} | SL Distance: {stop_loss_distance:.5f}"
        )
        return lot

    def _calculate_stop_loss_price(self, order_type: str, price: float, 
                                 atr: float, spread: float) -> float:
        """
        Calculate stop loss price based on order type and ATR.
        
        Args:
            order_type: Type of order ('buy' or 'sell')
            price: Entry price
            atr: Average True Range value
            spread: Current spread
            
        Returns:
            float: Calculated stop loss price
        """
        if order_type == 'buy':
            return price - atr - spread
        else:  # sell order
            return price + atr + spread

    def _calculate_take_profit_price(self, order_type: str, price: float, 
                                   risk_reward_ratio: float, atr: float, 
                                   spread: float) -> float:
        """
        Calculate take profit price based on order type, ATR and risk-reward ratio.
        
        Args:
            order_type: Type of order ('buy' or 'sell')
            price: Entry price
            risk_reward_ratio: Risk to reward ratio
            atr: Average True Range value
            spread: Current spread
            
        Returns:
            float: Calculated take profit price
        """
        if order_type == 'buy':
            return price + (risk_reward_ratio * atr) + spread
        else:  # sell order
            return price - (risk_reward_ratio * atr) - spread

    def execute_trade(self, prediction: Dict[str, Any]) -> bool:
        """
        Execute a trade based on model prediction.
        
        Args:
            prediction: Dictionary containing prediction details
            
        Returns:
            bool: True if a trade was executed, False otherwise
        """
        trade_action = prediction['action_type']
        risk_reward_ratio = prediction['risk_reward_ratio']
        atr = prediction['atr']

        if trade_action == 'BUY':
            return self.open_trade('buy', risk_reward_ratio, atr)
        elif trade_action == 'SELL':
            return self.open_trade('sell', risk_reward_ratio, atr)
        else:
            self.logger.info(f"No trade action for prediction: {trade_action}")
            return False