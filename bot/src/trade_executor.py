import logging
from config import *

class TradeExecutor:
    def __init__(self, mt5_connector):
        self.mt5_connector = mt5_connector

    def apply_trailing_stops(self, trailing_points):
        positions = self.mt5_connector.get_open_positions(MT5_SYMBOL, MT5_COMMENT)

        if positions is None or len(positions) == 0:
            return

        for pos in positions:
            if pos.type == 0:  # Buy position
                new_sl = pos.price_current - trailing_points
                if new_sl > pos.sl and new_sl >= pos.price_open:
                    self.mt5_connector.modify_stop_loss(pos.ticket, new_sl, pos.tp)

            elif pos.type == 1:  # Sell position
                new_sl = pos.price_current + trailing_points
                if new_sl < pos.sl and new_sl <= pos.price_open:
                    self.mt5_connector.modify_stop_loss(pos.ticket, new_sl, pos.tp)

    def open_trade(self, order_type, sl, tp):
        account_balance = self.mt5_connector.get_account_balance()
        filling_type = self.mt5_connector.check_filling_type(order_type)

        bid, ask = self.mt5_connector.get_symbol_info_tick(MT5_SYMBOL)
        logging.debug(f"Bid: {bid} | Ask: {ask}")

        spread = ask - bid
        if spread >= MAX_SPREAD:
            logging.warning(f"Trade rejected. Spread too high. Spread: {spread} | Max: {MAX_SPREAD}")
            return False
        
        price = ask if order_type == 'buy' else bid

        sl_price = price - sl - spread if order_type == 'buy' else price + sl + spread
        tp_price = price + tp + spread if order_type == 'buy' else price - tp - spread

        lot = self.get_lot_size(price, sl_price, account_balance)

        if self.mt5_connector.open_trade(lot, price, sl_price, tp_price, order_type, filling_type):
            logging.info(f"Trade executed successfully. Balance: {account_balance} | Order: {order_type} | Lot: {lot} | Price: {price} | SL: {round(sl_price, 3)} | TP: {round(tp_price, 3)}")
            return True
        return False

    def get_lot_size(self, entry_price, stop_loss_price, account_balance):        
        contract_size, min_lot, max_lot = self.mt5_connector.get_symbol_info(MT5_SYMBOL)
        
        # Get USDZAR price for conversion
        usd_zar_bid, _ = self.mt5_connector.get_symbol_info_tick(MT5_BASE_SYMBOL)

        # Calculate risk in ZAR and convert to USD
        risk_amount = account_balance * (RISK_PERCENTAGE / 100)
        risk_in_usd = risk_amount / usd_zar_bid

        # Calculate stop-loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        # Calculate lot size
        lot = risk_in_usd / (stop_loss_distance * contract_size)

        # Cap the lot size to the min and max values
        lot = round(max(min_lot, min(lot, max_lot)), 2)

        logging.debug(f"Lot size calculation: Contract: {contract_size} | USDZAR: {usd_zar_bid:.2f} | Risk: R{risk_amount:.2f} | Risk: ${risk_in_usd:.2f} | SL: {stop_loss_distance:.2f}")
        return lot

    def execute_trade(self, trade_action, sl, tp):
        if trade_action == 1:
            self.open_trade('buy', sl, tp)
        elif trade_action == 2:
            self.open_trade('sell', sl, tp)