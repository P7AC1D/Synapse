"""Reward calculation for trading environment with sparse rewards."""
import numpy as np
from typing import Union, Optional
from .actions import Action

class RewardCalculator:
    """Handles sparse reward calculation for trading actions."""
    
    def __init__(self, env):
        """Initialize reward calculator.
        
        Args:
            env: Trading environment instance
        """
        self.env = env
        self.previous_balance_high = env.initial_balance
        self.trade_entry_balance = env.initial_balance  # Track balance at trade entry
        self.last_direction = None  # Track trade direction for reversals
        self.min_hold_bars = 20  # Minimum bars for long hold reward
        
        # Track max unrealized profit for each trade
        self.max_unrealized_pnl = 0.0
        
        # Reward constants
        self.INVALID_ACTION_PENALTY = -1.0      # Penalty for invalid actions
        self.GOOD_HOLD_REWARD = 0.1            # Reduced reward for holding profitable positions
        self.LOSING_HOLD_PENALTY = -0.2        # Increased penalty for holding losing positions
        self.TIME_PRESSURE_THRESHOLD = 50       # Reduced threshold for time pressure
        self.MAX_HOLD_TIME = 100               # Maximum hold time before forced penalties
        self.PNL_SCALE = 1000.0               # Fixed scale for PnL rewards
        
    def _is_market_flat(self) -> bool:
        """Check if market is in consolidation using raw price action."""
        current_idx = self.env.current_step
        range_ratio = self.env.raw_data['range_ratio'].iloc[current_idx]
        body_ratio = abs(self.env.raw_data['body_ratio'].iloc[current_idx])
        volume = abs(self.env.raw_data['volume_change'].iloc[current_idx])
        
        # Market is flat when:
        # - Small price ranges relative to price level
        # - Small real bodies (indecisive)
        # - Volume is not spiking
        return (range_ratio < 0.002 and 
                body_ratio < 0.3 and 
                volume < 0.5)
        
    def _is_successful_reversal(self, position_type: int, pnl: float) -> bool:
        """Check if a trade reversal was successful."""
        if self.last_direction is None or position_type == 0:
            return False
            
        is_reversal = (
            (position_type == 1 and self.last_direction == 2) or
            (position_type == 2 and self.last_direction == 1)
        )
        
        return is_reversal and pnl > 0

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate rewards combining sparse and scaled components."""
        # Handle invalid actions first
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
            
        reward = 0.0
        
        # Position management rewards
        if position_type != 0:  # If we have an open position
            # Update max unrealized PnL
            if pnl > self.max_unrealized_pnl:
                self.max_unrealized_pnl = pnl
            else:
                # Penalty for giving back profits
                profit_drawdown = (self.max_unrealized_pnl - pnl) / self.trade_entry_balance
                reward -= profit_drawdown * 0.5
            
            if action == Action.HOLD:
                # Enhanced hold rewards/penalties with stronger time pressure
                if pnl > 0:
                    # Faster decay for profitable positions
                    hold_decay = max(0.1, 1.0 - (current_hold / self.TIME_PRESSURE_THRESHOLD))
                    reward += self.GOOD_HOLD_REWARD * hold_decay
                    
                    # Additional time pressure on profitable trades
                    if current_hold > self.TIME_PRESSURE_THRESHOLD:
                        reward -= 0.05 * (current_hold - self.TIME_PRESSURE_THRESHOLD) / self.TIME_PRESSURE_THRESHOLD
                else:
                    # Increasing penalty for holding losing positions
                    loss_multiplier = min(3.0, 1.0 + current_hold / self.TIME_PRESSURE_THRESHOLD)
                    reward += self.LOSING_HOLD_PENALTY * loss_multiplier
                
                # Force closure through extreme penalty after MAX_HOLD_TIME
                if current_hold > self.MAX_HOLD_TIME:
                    reward -= 0.5 * (current_hold - self.MAX_HOLD_TIME) / self.TIME_PRESSURE_THRESHOLD
                    
            elif action == Action.CLOSE:
                # Sparse direction reward
                reward += 1.0 if pnl > 0 else -1.0
                
                # Fixed-scale PnL component using PNL_SCALE
                reward += pnl / (self.PNL_SCALE * self.env.MIN_LOTS)  # Scale relative to minimum position
                
                # Track direction for reversals
                self.last_direction = position_type
                
                # Reset max unrealized PnL
                self.max_unrealized_pnl = 0.0
                
                # Reversal bonus (removed long hold bonus to discourage excessive holding)
                if self._is_successful_reversal(position_type, pnl):
                    reward += 1.0
        else:  # No position open
            # Update trade entry balance and reset max unrealized PnL for new positions
            if action in [Action.BUY, Action.SELL]:
                self.trade_entry_balance = self.env.balance
                self.max_unrealized_pnl = 0.0
                
            # Small reward for staying out during consolidation, with time limit
            if action == Action.HOLD:
                if self._is_market_flat() and current_hold <= 10:
                    reward += 0.02  # Reduced reward to discourage permanent sitting out
                    
        # New Balance High bonus with scaled reward
        if self.env.balance > self.previous_balance_high:
            percent_gain = (self.env.balance - self.previous_balance_high) / self.previous_balance_high
            reward += min(2.0, percent_gain * 10.0)  # Cap at 2.0 reward
            self.previous_balance_high = self.env.balance
        
        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -20.0  # Severe bankruptcy penalty
        
        # End of episode bonus
        if balance > initial_balance:
            return 10.0
        
        return 0.0
