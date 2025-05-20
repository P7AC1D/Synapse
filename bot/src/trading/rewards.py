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
        self.trade_entry_balance = env.initial_balance
        
        # Core reward constants
        self.INVALID_ACTION_PENALTY = -0.5
        self.DRAWDOWN_PENALTY_FACTOR = 1.0  # Scale factor for drawdown penalty
        
    def reset(self, initial_balance: float, min_bars: int = None) -> None:
        """Reset reward calculator state.
        
        Args:
            initial_balance: Starting balance for new episode
            min_bars: Minimum bars required per episode (unused in simplified reward)
        """
        self.trade_entry_balance = initial_balance

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate rewards based on pure returns and drawdown penalty.
        
        Args:
            action: Current action taken
            position_type: Current position type (0: none, 1: long, 2: short)
            pnl: Current profit/loss for the position
            atr: Average True Range value (unused in simplified reward)
            current_hold: Number of bars position has been held (unused in simplified reward)
            optimal_hold: Optional optimal holding period (unused in simplified reward)
            invalid_action: Whether the action was invalid
            
        Returns:
            float: Calculated reward value
        """
        # Handle invalid actions first
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
            
        reward = 0.0
        
        if position_type != 0:  # If we have an open position
            if action == Action.CLOSE:
                # Pure returns reward on position close
                reward = pnl / self.trade_entry_balance
                
                # Add drawdown penalty
                current_drawdown = self.env.metrics.get_equity_drawdown()
                drawdown_penalty = -current_drawdown * self.DRAWDOWN_PENALTY_FACTOR
                reward += drawdown_penalty
        else:  # No position open
            # Update trade entry balance for new positions
            if action in [Action.BUY, Action.SELL]:
                self.trade_entry_balance = self.env.balance
        
        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state.
        
        Args:
            balance: Final account balance
            initial_balance: Initial account balance
            
        Returns:
            float: Terminal reward value
        """
        if balance <= 0:
            return -10.0  # Bankruptcy penalty
        
        # End of episode bonus for profitability
        if balance > initial_balance:
            return 10.0
        
        return 0.0
