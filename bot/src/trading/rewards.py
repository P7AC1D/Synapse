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
        self.trade_entry_balance = env.initial_balance  # Track balance at trade entry
        
        # Reward constants
        self.INVALID_ACTION_PENALTY = -1.0  # Penalty for invalid actions
        self.TRANSACTION_COST = 0.0005  # 5 basis points per trade
        self.TIME_STEP_COST = 0.00001  # Reduced time step cost (was 0.0001)
        self.TIME_PRESSURE_THRESHOLD = 100  # Needed for environment observation scaling
        
        # State tracking
        self.previous_balance_high = env.initial_balance
        self.last_direction = None
        self.max_unrealized_pnl = 0.0
        self.bars_since_consolidation = 0
        
    def reset(self, initial_balance: float, min_bars: int = None) -> None:
        """Reset reward calculator state.
        
        Args:
            initial_balance: Starting balance for new episode
            min_bars: Minimum bars required per episode
        """
        self.trade_entry_balance = initial_balance
        self.previous_balance_high = initial_balance
        self.last_direction = None
        self.max_unrealized_pnl = 0.0
        self.bars_since_consolidation = 0
        
    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate rewards using a simplified approach focused on PnL."""
        # Handle invalid actions first
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
            
        reward = -self.TIME_STEP_COST  # Small negative reward per timestep
        
        # Transaction cost for new trades
        if action in [Action.BUY, Action.SELL]:
            reward -= self.TRANSACTION_COST
            self.trade_entry_balance = self.env.balance
            
        # Core PnL reward on position close
        elif action == Action.CLOSE and position_type != 0:
            # Scaled PnL component with slightly higher weight
            reward += 1.5 * (pnl / self.trade_entry_balance)  # Increased from 1.0 to 1.5
            # Transaction cost for closing
            reward -= self.TRANSACTION_COST
        
        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -5.0  # Bankruptcy penalty
        
        # End of episode reward based on total return
        return (balance - initial_balance) / initial_balance
