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
        
        # Only reward/penalize on position close
        if action == Action.CLOSE and position_type != 0:
            normalized_pnl = pnl / self.env.balance if self.env.balance > 0 else 0
            reward = normalized_pnl  # Direct PnL scaled by balance
            
        # Penalize trying to open position when one exists
        elif action in [Action.BUY, Action.SELL] and position_type != 0:
            reward = -0.5  # Fixed penalty for invalid action
            
        # Track inactivity
        if hasattr(self, 'bars_since_trade'):
            self.bars_since_trade += 1
        else:
            self.bars_since_trade = 0
            
        if action in [Action.BUY, Action.SELL]:
            self.bars_since_trade = 0
            
        # Exponential inactivity penalty
        if self.bars_since_trade > 100:
            penalty_base = 1.05  # 5% exponential growth
            scaled_penalty = penalty_base ** (self.bars_since_trade - 100) - 1
            reward -= min(0.5, scaled_penalty * 0.01)  # Cap at -0.5

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
