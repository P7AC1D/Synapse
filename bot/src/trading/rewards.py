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
        
        # Calculate position metrics
        normalized_pnl = pnl / self.env.balance if self.env.balance > 0 else 0
        risk_adjusted_pnl = normalized_pnl / (atr * 0.01) if atr > 0 else normalized_pnl  # Scale by volatility
        
        # Reward for closing positions
        if action == Action.CLOSE and position_type != 0:
            if pnl > 0:
                # Reward profitable trades based on risk-adjusted return
                reward = risk_adjusted_pnl * 2.0  # Double the reward for good trades
                # Extra reward for quick profitable trades
                if bars_held < self.max_hold_bars * 0.5:
                    reward *= 1.5
            else:
                # Penalize losses more heavily
                reward = risk_adjusted_pnl * 3.0  # Triple the penalty for losses
                
        # Penalize invalid actions
        elif action in [Action.BUY, Action.SELL] and position_type != 0:
            reward = -1.0  # Stronger penalty for invalid actions
            
        # HOLD rewards based on position performance
        elif action == Action.HOLD and position_type != 0:
            if pnl > 0:
                # Small reward for holding winners
                reward = risk_adjusted_pnl * 0.1
            else:
                # Larger penalty for holding losers
                reward = risk_adjusted_pnl * 0.2
                
        # Track and penalize inactivity
        if hasattr(self, 'bars_since_trade'):
            self.bars_since_trade += 1
        else:
            self.bars_since_trade = 0
            
        if action in [Action.BUY, Action.SELL]:
            self.bars_since_trade = 0
            
        # Logarithmic inactivity penalty
        if self.bars_since_trade > 100:
            excess_bars = min(self.bars_since_trade - 100, 1000)  # Cap at 1000 bars
            if excess_bars > 0:
                # Use log scaling for smoother, bounded growth
                scaled_penalty = 0.1 * np.log1p(excess_bars / 100)  # Normalize by 100 for gradual scaling
                reward -= min(1.0, scaled_penalty)  # Still cap at -1.0

        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -2.0  # Severe bankruptcy penalty
            
        # Reward/penalize based on final return
        return_pct = (balance - initial_balance) / initial_balance
        
        if return_pct > 0:
            # Bonus for finishing with profit
            return return_pct * 2.0  # Double the positive return as reward
        else:
            # Penalty for finishing with loss
            return return_pct * 3.0  # Triple the negative return as penalty
