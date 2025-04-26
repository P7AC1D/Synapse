"""MetaTrader 5 connection and trading interface."""

import logging
import pytz
from datetime import datetime
from typing import Optional, Tuple, List, Any, Dict, Union

import MetaTrader5 as mt5

from config import MT5_PATH, MT5_COMMENT, MT5_BASE_SYMBOL, MT5_SYMBOL
from creds import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

def to_mt5_timeframe(timeframe_minutes: int) -> int:
    """Convert minutes to MT5 timeframe constant."""
    timeframe_map = {
        1: mt5.TIMEFRAME_M1,
        2: mt5.TIMEFRAME_M2,
        3: mt5.TIMEFRAME_M3,
        4: mt5.TIMEFRAME_M4,
        5: mt5.TIMEFRAME_M5,
        6: mt5.TIMEFRAME_M6,
        10: mt5.TIMEFRAME_M10,
        12: mt5.TIMEFRAME_M12,
        15: mt5.TIMEFRAME_M15,
        20: mt5.TIMEFRAME_M20,
        30: mt5.TIMEFRAME_M30,
        60: mt5.TIMEFRAME_H1,
        120: mt5.TIMEFRAME_H2,
        180: mt5.TIMEFRAME_H3,
        240: mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
        10080: mt5.TIMEFRAME_W1,
        43200: mt5.TIMEFRAME_MN1
    }
    return timeframe_map.get(timeframe_minutes, mt5.TIMEFRAME_M1)

class MT5Connector:
    """Connector for interacting with MetaTrader 5 platform."""
    
    def __init__(self):
        """Initialize MT5 connector."""
        self.connected = False
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """Connect to MT5 platform."""
        if self.connected:
            return True
            
        success = mt5.initialize(
            path=MT5_PATH,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            timeout=600000,
            portable=False
        )
        
        if not success:
            self.logger.critical(f"MT5 initialization failed: {mt5.last_error()}")
            self.connected = False
            return False

        account_info = mt5.account_info()
        if account_info is None:
            self.logger.critical(f"Failed to connect to account: {mt5.last_error()}")
            self.connected = False
            return False
            
        self.logger.info(f"Connected to MT5 account: {account_info.login}")
        self.connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from MT5 platform."""
        if self.connected:
            mt5.shutdown()
            self.logger.info("Disconnected from MT5")
        self.connected = False

    def _ensure_connected(self) -> bool:
        """Ensure connection to MT5 is active."""
        if not self.connected:
            return self.connect()
        return True
    
    def fetch_current_bar(self, symbol: str, timeframe_minutes: int) -> Optional[Any]:
        """Fetch current price bar."""
        if not self._ensure_connected():
            return None

        return mt5.copy_rates_from(
            symbol, 
            to_mt5_timeframe(timeframe_minutes), 
            datetime.now(pytz.utc), 
            1
        )
    
    def fetch_data(self, symbol: str, timeframe_minutes: int, bar_count: int) -> Optional[Any]:
        """Fetch historical price data."""
        if not self._ensure_connected():
            return None

        rates = mt5.copy_rates_from_pos(
            symbol, 
            to_mt5_timeframe(timeframe_minutes), 
            0,  # start position
            bar_count
        )
        
        if rates is not None:
            # Convert to pandas DataFrame
            import pandas as pd
            df = pd.DataFrame(rates)
            return df
        
        return None
    
    def check_filling_type(self, symbol: str, order_type: str) -> int:
        """Check appropriate filling type for the symbol."""
        if not self._ensure_connected():
            return mt5.ORDER_FILLING_IOC  # Default filling type

        price = mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid

        for filling_type in range(2):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": mt5.symbol_info(symbol).volume_min,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "type_filling": filling_type,
                "type_time": mt5.ORDER_TIME_GTC
            }

            result = mt5.order_check(request)
            if result and result.comment == "Done":
                return filling_type
                
        return mt5.ORDER_FILLING_IOC  # Default to IOC if no valid filling type found
    
    def open_trade(self, symbol: str, lot: float, price: float, order_type: str, filling_type: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        """
        Open a new trade with optional stop loss and take profit.
        
        Args:
            symbol: Trading symbol
            lot: Trade volume in lots
            price: Entry price
            order_type: Either 'buy' or 'sell'
            filling_type: MT5 filling type
            sl: Stop Loss price (optional)
            tp: Take Profit price (optional)
        """
        if not self._ensure_connected():
            return False

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 0,
            "comment": MT5_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
            "sl": sl,  # Set stop loss if provided
            "tp": tp   # Set take profit if provided
        }

        # Pre-format stop loss and take profit values
        sl_str = f"{sl:.5f}" if sl > 0 else "None"
        tp_str = f"{tp:.5f}" if tp > 0 else "None"
        
        self.logger.debug(
            f"Sending order for {symbol}. Type: {order_type} | Price: {price:.5f} | "
            f"Lot: {lot} | SL: {sl_str} | TP: {tp_str}"
        )
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.warning(f"Failed to send order: {result.retcode} | {result.comment}")
            return False
        return True
    
    def get_account_balance(self) -> float:
        """Get account balance."""
        if not self._ensure_connected():
            raise Exception("Not connected to MT5")

        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        return account_info.balance
    
    def get_symbol_info(self, symbol: str) -> Tuple[float, float, float]:
        """Get symbol trading information."""
        if not self._ensure_connected():
            raise Exception(f"Not connected to MT5")
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Failed to get symbol info for {symbol}")
        return symbol_info.trade_contract_size, symbol_info.volume_min, symbol_info.volume_max
    
    def get_symbol_info_tick(self, symbol: str) -> Tuple[float, float]:
        """Get current bid/ask prices."""
        if not self._ensure_connected():
            raise Exception(f"Not connected to MT5")

        symbol_info_tick = mt5.symbol_info_tick(symbol)
        if symbol_info_tick is None:
            raise Exception(f"Failed to get {symbol} info")
        
        return symbol_info_tick.bid, symbol_info_tick.ask
    
    def get_open_positions(self, symbol: str, comment: str) -> List[Any]:
        """Get open positions for a symbol."""
        if not self._ensure_connected():
            return []

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []

        filtered_positions = [pos for pos in positions if pos.comment == comment]
        self.logger.debug(f"Fetched {len(filtered_positions)} open positions for {symbol}.")
        return filtered_positions
    
    def close_position(self, ticket: int) -> bool:
        """Close a specific position."""
        if not self._ensure_connected():
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            self.logger.warning(f"Position {ticket} not found.")
            return False

        position = positions[0]
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
            "deviation": 20,
            "magic": 0,
            "comment": MT5_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.warning(f"Failed to close position {position.ticket}, error: {result.comment}")
            return False
            
        self.logger.info(f"Closed position {position.ticket} for {position.symbol}.")
        return True
            
    def close_open_positions(self, symbol: str, comment: str) -> int:
        """Close all open positions for a symbol."""
        if not self._ensure_connected():
            return 0

        positions = self.get_open_positions(symbol, comment)
        if not positions:
            return 0

        closed_count = 0
        for position in positions:
            success = self.close_position(position.ticket)
            if success:
                closed_count += 1
                
        return closed_count
