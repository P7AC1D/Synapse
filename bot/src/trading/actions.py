"""Action handling and execution for trading environment."""
from enum import IntEnum
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np

class Action(IntEnum):
    """Trading actions enumeration."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

class ActionHandler:
    """Handles action processing and execution."""
    
    def __init__(self, env):
        """Initialize action handler.
        
        Args:
            env: Trading environment instance
        """
        self.env = env

    def _safe_get_price_data(self, price_type: str, step: int = None) -> float:
        """Safely get price data with bounds checking.
        
        Args:
            price_type: Type of price data ('close', 'spread', 'atr', etc.)
            step: Step index to retrieve (uses current_step if None)
            
        Returns:
            Price value at the specified step
        """
        if step is None:
            step = self.env.current_step
            
        max_index = len(self.env.prices[price_type]) - 1
        
        if step > max_index:
            print(f"WARNING: step {step} exceeds {price_type} data bounds {max_index}. Using last available index.")
            step = max_index
        elif step < 0:
            print(f"WARNING: step {step} is negative for {price_type} data. Using index 0.")
            step = 0
            
        return self.env.prices[price_type][step]

    def process_action(self, action: Union[int, np.ndarray]) -> int:
        """Convert action to trading decision.
        
        Args:
            action: Integer action from policy (0: hold, 1: buy, 2: sell, 3: close)
            
        Returns:
            int: Processed action (0: hold, 1: buy, 2: sell, 3: close)
        """
        # Handle array input from policy
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Ensure action is within valid range
        action = int(action) % 4
        
        return action

    def execute_trade(self, direction: int, raw_spread: float) -> None:
        """Execute a trade with the given direction.
        
        Args:
            direction: Direction of the trade (1: buy, 2: sell)
            raw_spread: Current spread to adjust entry price        """
        current_price = self._safe_get_price_data('close')
        current_atr = self._safe_get_price_data('atr')
        
        # Apply slippage if configured
        slippage = 0.0
        if hasattr(self.env, 'slippage_range') and self.env.slippage_range > 0:
            slippage = np.random.uniform(0, self.env.slippage_range) * self.env.POINT_VALUE
            if direction == 1:  # Buy - slippage increases entry price
                current_price += slippage
            else:  # Sell - slippage decreases entry price
                current_price -= slippage
        # Calculate lot size based on account balance in USD equivalent
        usd_balance = self.env.balance / self.env.currency_conversion
        usd_balance_per_lot = self.env.BALANCE_PER_LOT / self.env.currency_conversion
        
        lot_size = max(
            self.env.MIN_LOTS,
            min(
                self.env.MAX_LOTS,
                round((usd_balance / usd_balance_per_lot) * self.env.MIN_LOTS, 2)
            )
        )
        
        # Calculate adjusted entry price based on direction and spread
        if direction == 1:  # Long position
            # For long positions, we buy at the ask price (close + spread)
            adjusted_entry_price = current_price + raw_spread
        else:  # Short position
            # For short positions, we sell at the bid price (close)
            adjusted_entry_price = current_price

        self.env.current_position = {
            "direction": 1 if direction == 1 else -1,
            "entry_price": adjusted_entry_price,
            "entry_spread": raw_spread,
            "lot_size": lot_size,
            "entry_time": str(self.env.original_index[self.env.current_step]),
            "entry_step": self.env.current_step,
            "entry_atr": current_atr,
            "current_profit_points": 0.0
        }
        
        self.env.trade_metrics['current_direction'] = self.env.current_position["direction"]

    def close_position(self) -> Tuple[float, Dict[str, Any]]:
        """Close current position and calculate P/L.
        
        Returns:
            Tuple of (pnl, trade_info)
        """
        if not self.env.current_position:
            return 0.0, {}
          # Validate current_step is within bounds and get price data safely
        current_price = self._safe_get_price_data('close')
        direction = self.env.current_position["direction"]
        entry_price = self.env.current_position["entry_price"]
        lot_size = self.env.current_position["lot_size"]
        entry_step = self.env.current_position["entry_step"]
        
        # Get current spread for exit price adjustment with bounds checking
        current_spread = self._safe_get_price_data('spread') * self.env.POINT_VALUE
        
        # Apply slippage if configured
        slippage = 0.0
        if hasattr(self.env, 'slippage_range') and self.env.slippage_range > 0:
            slippage = np.random.uniform(0, self.env.slippage_range) * self.env.POINT_VALUE
            if direction == 1:  # Long position closing - slippage decreases exit price
                current_price -= slippage
            else:  # Short position closing - slippage increases exit price
                current_price += slippage
        
        # Calculate profit or loss with spread at exit
        if direction == 1:  # Long position
            # For long exits, we sell at bid price
            exit_price = current_price
            profit_points = exit_price - entry_price
        else:  # Short position
            # For short exits, we buy back at ask price
            exit_price = current_price + current_spread
            profit_points = entry_price - exit_price
            
        # Calculate P&L in USD then convert to account currency
        usd_pnl = profit_points * lot_size * self.env.CONTRACT_SIZE
        # Multiply by currency conversion rate to get P&L in account currency
        pnl = usd_pnl * self.env.currency_conversion
        profit_points_normalized = profit_points / self.env.POINT_VALUE
        
        # Create trade info
        trade_info = {
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": self.env.current_position["entry_time"],
            "exit_time": str(self.env.original_index[self.env.current_step]),
            "profit_points": profit_points_normalized,
            "pnl": pnl,
            "hold_time": self.env.current_step - entry_step,
            "lot_size": lot_size,
            "entry_spread": self.env.current_position["entry_spread"],
            "exit_spread": current_spread,
            "entry_step": entry_step,
            "exit_step": self.env.current_step
        }
        
        return pnl, trade_info

    def manage_position(self) -> Tuple[float, float]:
        """Calculate current position's unrealized P/L.
        
        Returns:
            Tuple of (unrealized_pnl, profit_points)
        """
        if not self.env.current_position:
            return 0.0, 0.0
            
        current_price = self._safe_get_price_data('close')
        direction = self.env.current_position["direction"]
        entry_price = self.env.current_position["entry_price"]
        lot_size = self.env.current_position["lot_size"]
        
        # Get current spread for unrealized P&L calculation
        current_spread = self._safe_get_price_data('spread') * self.env.POINT_VALUE
        
        # Calculate raw P&L first - use current price without spread for display
        if direction == 1:  # Long position
            current_exit_price = current_price - current_spread  # Subtract spread for long exits
            profit_points = current_exit_price - entry_price
        else:  # Short position
            current_exit_price = current_price + current_spread  # Add spread for short exits
            profit_points = entry_price - current_exit_price
            
        # Calculate P&L in USD then convert to account currency
        usd_pnl = profit_points * lot_size * self.env.CONTRACT_SIZE
        # Multiply by currency conversion rate to get P&L in account currency
        unrealized_pnl = usd_pnl * self.env.currency_conversion
        profit_points_normalized = profit_points / self.env.POINT_VALUE

        return unrealized_pnl, profit_points_normalized

    def check_force_close(self) -> Tuple[bool, Optional[Dict]]:
        """Check if position should be force-closed and lock in exit price.
        
        Returns:
            Tuple of (should_close, locked_exit_data)
        """
        if not self.env.current_position:
            return False, None
            
        # Skip force-close check if max_loss_points is not set
        if not hasattr(self.env, 'max_loss_points') or self.env.max_loss_points <= 0:
            return False, None
            
        # Get current profit points
        _, profit_points = self.manage_position()
        
        # Force close if loss exceeds threshold (profit_points will be negative for losses)
        if profit_points <= -self.env.max_loss_points:
            # CRITICAL FIX: Lock in exit price immediately when threshold breached
            locked_exit_data = self._lock_exit_price()
            return True, locked_exit_data
            
        return False, None

    def _lock_exit_price(self) -> Dict[str, Any]:
        """Lock in exit price when force-close triggers to prevent further deterioration.
        
        Returns:
            Dictionary containing locked exit price and related data
        """
        current_price = self._safe_get_price_data('close')
        current_spread = self._safe_get_price_data('spread') * self.env.POINT_VALUE
        direction = self.env.current_position["direction"]
        entry_price = self.env.current_position["entry_price"]
        
        # Calculate exact exit price based on position direction
        if direction == 1:  # Long position
            locked_exit_price = current_price  # Sell at bid (current price)
        else:  # Short position  
            locked_exit_price = current_price + current_spread  # Buy at ask (current price + spread)
        
        # Calculate locked profit points for validation
        if direction == 1:
            locked_profit_points = locked_exit_price - entry_price
        else:
            locked_profit_points = entry_price - locked_exit_price
            
        locked_profit_points_normalized = locked_profit_points / self.env.POINT_VALUE
        
        return {
            'locked_exit_price': locked_exit_price,
            'locked_profit_points': locked_profit_points_normalized,
            'lock_timestamp': self.env.current_step,
            'market_price': current_price,
            'spread_used': current_spread,
            'direction': direction,
            'entry_price': entry_price
        }

    def execute_force_close(self, locked_exit_data: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
        """Execute a forced closure using locked exit price to prevent timing issues.
        
        Args:
            locked_exit_data: Dictionary containing locked exit price data
            
        Returns:
            Tuple of (pnl, trade_info)
        """
        if not self.env.current_position:
            return 0.0, {}
        
        # Use locked exit data if provided, otherwise fall back to regular close logic
        if locked_exit_data:
            return self._execute_force_close_with_locked_price(locked_exit_data)
        else:
            # Fallback to regular close logic (for backward compatibility)
            print("WARNING: Force-close executed without locked price data - timing bug may occur")
            pnl, trade_info = self.close_position()
            
            # Mark the trade as force-closed
            if trade_info:
                trade_info['force_closed'] = True
                trade_info['close_reason'] = 'max_loss_reached'
                
            return pnl, trade_info
    
    def _execute_force_close_with_locked_price(self, locked_exit_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Execute force-close using locked exit price to ensure consistent timing.
        
        Args:
            locked_exit_data: Dictionary containing locked exit price and related data
            
        Returns:
            Tuple of (pnl, trade_info)
        """
        position = self.env.current_position
        direction = position["direction"]
        entry_price = position["entry_price"]
        lot_size = position["lot_size"]
        entry_step = position["entry_step"]
        
        # Use LOCKED exit price instead of recalculating
        exit_price = locked_exit_data['locked_exit_price']
        profit_points_normalized = locked_exit_data['locked_profit_points']
        
        # CRITICAL VALIDATION: Ensure we don't exceed threshold due to timing
        if profit_points_normalized < -self.env.max_loss_points:
            # Safety fallback - adjust to exact threshold
            target_loss_points = -self.env.max_loss_points
            
            # Recalculate exit price to exactly meet threshold
            if direction == 1:
                exit_price = entry_price + (target_loss_points * self.env.POINT_VALUE)
            else:
                exit_price = entry_price - (target_loss_points * self.env.POINT_VALUE)
                
            profit_points_normalized = target_loss_points
        
        # Calculate profit points for PnL calculation
        if direction == 1:  # Long position
            profit_points = exit_price - entry_price
        else:  # Short position
            profit_points = entry_price - exit_price
        
        # Calculate P&L in USD then convert to account currency
        usd_pnl = profit_points * lot_size * self.env.CONTRACT_SIZE
        pnl = usd_pnl * self.env.currency_conversion
        
        # Create comprehensive trade info
        trade_info = {
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_time": position["entry_time"],
            "exit_time": str(self.env.original_index[self.env.current_step]),
            "profit_points": profit_points_normalized,
            "pnl": pnl,
            "hold_time": self.env.current_step - entry_step,
            "lot_size": lot_size,
            "entry_spread": position["entry_spread"],
            "exit_spread": locked_exit_data["spread_used"],
            "entry_step": entry_step,
            "exit_step": self.env.current_step,
            "force_closed": True,
            "close_reason": "max_loss_reached",
            "locked_price_used": True,
            "lock_timestamp": locked_exit_data["lock_timestamp"],
            "market_price_at_lock": locked_exit_data["market_price"]
        }
        
        # Final validation
        if profit_points_normalized < -self.env.max_loss_points - 0.1:  # Small tolerance for rounding
            print(f"ERROR: Force-close validation failed! Loss {profit_points_normalized:.1f} exceeds limit {-self.env.max_loss_points:.1f}")
        
        return pnl, trade_info
