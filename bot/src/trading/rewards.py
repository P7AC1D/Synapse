"""Reward calculation for trading environment."""
import numpy as np
from typing import Union
from .actions import Action

class RewardCalculator:
    """Handles reward calculation for trading actions."""
    
    def __init__(self, max_hold_bars: int = 64):
        """Initialize reward calculator.
        
        Args:
            max_hold_bars: Maximum bars to hold a position
        """
        self.max_hold_bars = max_hold_bars

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, bars_held: int) -> float:
        """Calculate reward based on action and position state.
        
        Args:
            action: The action taken (HOLD, BUY, SELL, CLOSE)
            position_type: Current position type (-1=short, 0=none, 1=long)
            pnl: Current unrealized or realized PnL
            atr: Current ATR value
            bars_held: Number of bars position has been held
            
        Returns:
            float: Calculated reward
        """
        reward = 0.0

        if action == Action.CLOSE and position_type != 0:
            # Normalize pnl by ATR and clip
            reward = np.clip(pnl / atr, -2.0, 2.0)
            
            # Optional shaping
            reward = np.sign(reward) * (abs(reward) ** 0.5)

        elif action == Action.HOLD and position_type != 0:
            reward = pnl * 0.001  # light shaping
            if bars_held > self.max_hold_bars:
                reward -= 0.1  # discourage overstaying

        elif action in [Action.BUY, Action.SELL] and position_type == 0:
            reward = 0  # Neutral, or small penalty to reduce randomness

        # Optional time penalty
        reward -= 0.001  # per step cost

        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state.
        
        Args:
            balance: Current account balance
            initial_balance: Initial account balance
            
        Returns:
            float: Terminal reward
        """
        if balance <= 0:
            return -1.0  # Bankruptcy penalty
            
        return 0.0  # No additional terminal reward
