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
        self.bars_since_consolidation = 0
        self.min_hold_bars = 20  # Minimum bars for long hold reward
        self.consolidation_threshold = 0.0015  # BB width threshold for consolidation
        
    def _is_market_flat(self) -> bool:
        """Check if market is in consolidation based on volatility breakout."""
        current_idx = self.env.current_step
        # volatility_breakout close to 0.5 indicates price near middle of range
        # values between 0.4-0.6 suggest consolidation
        volatility_breakout = self.env.raw_data['volatility_breakout'].iloc[current_idx]
        return 0.4 <= volatility_breakout <= 0.6
        
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
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None) -> float:
        """Calculate rewards combining sparse and scaled components."""
        reward = 0.0
        
        # Update trade entry balance when opening new position
        if action in [Action.BUY, Action.SELL] and position_type == 0:
            self.trade_entry_balance = self.env.balance
            
        # Trade closure rewards
        if action == Action.CLOSE and position_type != 0:
            # Sparse direction reward
            reward += 1.0 if pnl > 0 else -1.0
            
            # Scaled PnL component
            reward += pnl / self.trade_entry_balance
            
            # Track direction for reversals
            self.last_direction = position_type
            
            # Long hold bonus
            if current_hold >= self.min_hold_bars and pnl > 0:
                reward += 0.5
            
            # Reversal bonus
            if self._is_successful_reversal(position_type, pnl):
                reward += 1.0
                
        # New Balance High
        if self.env.balance > self.previous_balance_high:
            reward += 1.0
            self.previous_balance_high = self.env.balance
            
        # Correct Non-Action (Discipline)
        if action == Action.HOLD and position_type == 0:
            if self._is_market_flat():
                reward += 1.0
                self.bars_since_consolidation = 0
            else:
                self.bars_since_consolidation += 1
        
        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -2.0  # Severe bankruptcy penalty
        
        # End of episode bonus
        if balance > initial_balance:
            return 10.0
        
        return 0.0
