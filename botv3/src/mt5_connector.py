import MetaTrader5 as mt5
import logging
import pytz
from config import *
from creds import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
from datetime import datetime

def to_mt5_timeframe(timeframe_minutes):
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
    return timeframe_map.get(timeframe_minutes, None)

class MT5Connector:
    def __init__(self):
        self.connected = False

    def connect(self):    
        if mt5.initialize(path=MT5_PATH,
                          login=MT5_LOGIN,
                          password=MT5_PASSWORD,
                          server=MT5_SERVER,
                          timeout=60000,
                          portable=False):
            logging.info("Platform MT5 launched correctly")

            account_info = mt5.account_info()
            if account_info is None:
                logging.critical("Failed to connect to the account!")
            else:
                logging.info(f"Connected to account: {account_info.login}")
                self.connected = True            
        else:
            logging.critical(f"There has been a problem with initialization: {mt5.last_error()}")
            self.connected = False

    def disconnect(self):
        mt5.shutdown()
        self.connected = False

    def is_connected(self):
        return self.connected
    
    def fetch_current_bar(self, symbol, timeframe_minutes):
        if self.connected == False:
            self.connect()

        return mt5.copy_rates_from(symbol, to_mt5_timeframe(timeframe_minutes), datetime.now(pytz.utc), 1)
    
    def fetch_data(self, symbol, timeframe_minutes, bar_count):
        if self.connected == False:
            self.connect()

        return mt5.copy_rates_from(symbol, to_mt5_timeframe(timeframe_minutes), datetime.now(pytz.utc), bar_count)
    
    def modify_stop_loss(self, ticket, sl, tp):
        if self.connected == False:
            self.connect()

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.warning(f"Failed to update SL for ticket {ticket}, error code: {result.retcode}")

    def check_filling_type(self, order_type):
        if self.connected == False:
            self.connect()
    
        symbol = MT5_SYMBOL
        price = mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid

        for i in range(2):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": mt5.symbol_info(symbol).volume_min,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "type_filling": i,
                "type_time": mt5.ORDER_TIME_GTC
            }

            result = mt5.order_check(request)

            if result.comment == "Done":
                break
        return i
    
    def open_trade(self, lot, price, sl_price, tp_price, order_type, filling_type):
        if self.connected == False:
            self.connect()

        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": MT5_SYMBOL,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": deviation,
            "magic": 0,
            "comment": MT5_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        logging.debug(f"Sending order. Type: {order_type} | Price: {price} | SL: {sl_price} | TP: {tp_price} | Lot: {lot}")
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.warning(f"Failed to send order: {result.retcode} | {result.comment}")
            return False
        return True
    
    def get_account_balance(self):
        if self.connected == False:
            self.connect()

        account_info = mt5.account_info()
        if account_info is None:
            raise Exception("Failed to get account info")
        return account_info.balance
    
    def get_symbol_info(self, symbol):
        if self.connected == False:
            self.connect()
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Failed to get symbol info for {symbol}")
        return symbol_info.trade_contract_size, symbol_info.volume_min, symbol_info.volume_max
    
    def get_symbol_info_tick(self, symbol):
        if self.connected == False:
            self.connect()

        symbol_info_tick = mt5.symbol_info_tick(symbol)
        if symbol_info_tick is None:
            raise Exception(f"Failed to get {symbol} info")
        
        return symbol_info_tick.bid, symbol_info_tick.ask
    
    def get_open_positions(self, symbol, comment):
        if self.connected == False:
            self.connect()

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []

        filtered_positions = [pos for pos in positions if pos.comment == comment]
        return filtered_positions
    
    def close_position(self, ticket):
        if not self.connected:
            self.connect()

        position_to_close = mt5.positions_get(ticket=ticket)
        if not position_to_close:
            logging.warning(f"Position {ticket} not found.")
            return

        request = {
            "action": mt5.TRADE_ACTION_CLOSE_BY,
            "position": ticket,
            "symbol": MT5_SYMBOL,
            "volume": position_to_close.volume,
            "magic": 0,
            "comment": MT5_COMMENT
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.warning(f"Failed to close position {position_to_close.ticket}, error: {result.comment}")
        else:
            logging.info(f"Closed position {position_to_close.ticket} for {MT5_SYMBOL}.")
    
    def close_open_positions(self, symbol, comment):
        if not self.connected:
            self.connect()

        positions = self.get_open_positions(symbol, comment)
        if not positions:
            return

        for position in positions:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                "deviation": 20,
                "magic": 0,
                "comment": "Closing position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.warning(f"Failed to close position {position.ticket}, error: {result.comment}")
            else:
                logging.info(f"Closed position {position.ticket} for {symbol}.")