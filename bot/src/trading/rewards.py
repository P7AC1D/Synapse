"""Reward calculation for trading environment."""
import numpy as np
from typing import Union
from .actions import Action

class RewardCalculator:
    """Handles reward calculation for trading actions."""
    
    def __init__(self, env, max_hold_bars: int = 64):
        """Initialize reward calculator.
        
        Args:
            env: Trading environment instance
            max_hold_bars: Maximum bars to hold a position
        """
        self.env = env
        self.max_hold_bars = max_hold_bars

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, bars_held: int) -> float:
        """Calculate reward based on action and position state."""
        reward = 0.0

        # Scale rewards by percentage of current balance
        normalized_pnl = pnl / self.env.balance if self.env.balance > 0 else 0

        if action == Action.CLOSE and position_type != 0:
            if pnl > 0:
                # Scale reward by percentage gain
                reward = normalized_pnl + 0.2
            else:
                # Scale penalty by percentage loss
                reward = normalized_pnl * 0.6

        elif action == Action.HOLD and position_type != 0:
            if pnl > 0:
                # Scale holding reward by percentage gain
                reward = normalized_pnl * 0.05
            else:
                # Scale holding penalty by percentage loss
                reward = normalized_pnl * 0.03

        elif action in [Action.BUY, Action.SELL] and position_type == 0:
            # Keep exploration incentive for opening positions
            reward = 0.1

        # Reduce per-step cost to be less punishing
        reward -= 0.0005  # Small time decay

        # Add inactivity penalty to prevent extended periods without trading
        if hasattr(self, 'bars_since_trade'):
            self.bars_since_trade += 1
        else:
            self.bars_since_trade = 0
            
        # Reset counter when trade is opened
        if action in [Action.BUY, Action.SELL]:
            self.bars_since_trade = 0
            
        # Apply increasing penalty for prolonged inactivity
        if self.bars_since_trade > 100:  # After 100 bars of no trading
            inactivity_penalty = min(0.005, (self.bars_since_trade - 100) * 0.0001)
            reward -= inactivity_penalty

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
