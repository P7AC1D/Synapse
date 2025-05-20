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
        
        # Track max unrealized profit for each trade
        self.max_unrealized_pnl = 0.0
        
        # Reward constants
        self.INVALID_ACTION_PENALTY = -0.5  # Less punitive for exploration
        self.GOOD_HOLD_REWARD = 0.3    # More encouragement for profitable trades
        self.LOSING_HOLD_PENALTY = -0.05  # Reduced penalty for holding losses
        self.TIME_PRESSURE_THRESHOLD = 150  # Extended time before pressure kicks in
        
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
        
    def _is_market_flat(self) -> bool:
        """Check if market is in consolidation based on volatility breakout."""
        current_idx = self.env.current_step
        # volatility_breakout close to 0.5 indicates price near middle of range
        # values between 0.4-0.6 suggest consolidation
        volatility_breakout = self.env.features_df['volatility_breakout'].iloc[current_idx]
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
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate rewards combining sparse and scaled components.
        
        Args:
            action: Current action taken
            position_type: Current position type (0: none, 1: long, 2: short)
            pnl: Current profit/loss for the position
            atr: Average True Range value
            current_hold: Number of bars position has been held
            optimal_hold: Optional optimal holding period
            invalid_action: Whether the action was invalid
            
        Returns:
            float: Calculated reward value
        """
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
                if pnl > 0:
                    # Decay the hold reward over time
                    hold_decay = max(0.2, 1.0 - (current_hold / 100))
                    reward += self.GOOD_HOLD_REWARD * hold_decay
                else:
                    # Penalty for holding losing positions
                    reward += self.LOSING_HOLD_PENALTY
                    
                # Add time pressure after threshold
                if current_hold > self.TIME_PRESSURE_THRESHOLD:
                    reward -= 0.01 * (current_hold - self.TIME_PRESSURE_THRESHOLD) / 100
                    
            elif action == Action.CLOSE:
                # Sparse direction reward
                reward += 1.0 if pnl > 0 else -1.0
                
                # Scaled PnL component
                reward += pnl / self.trade_entry_balance
                
                # Track direction for reversals
                self.last_direction = position_type
                
                # Reset max unrealized PnL
                self.max_unrealized_pnl = 0.0
                
                # Reversal bonus
                if self._is_successful_reversal(position_type, pnl):
                    reward += 1.0
        else:  # No position open
            # Update trade entry balance and reset max unrealized PnL for new positions
            if action in [Action.BUY, Action.SELL]:
                self.trade_entry_balance = self.env.balance
                self.max_unrealized_pnl = 0.0
                
            # Reward for staying out during consolidation
            if action == Action.HOLD:
                if self._is_market_flat():
                    reward += 0.1
                    self.bars_since_consolidation = 0
                else:
                    self.bars_since_consolidation += 1
                    
        # New Balance High bonus (applies to all situations)
        if self.env.balance > self.previous_balance_high:
            reward += 1.0
            self.previous_balance_high = self.env.balance
        
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
            return -10.0  # Significant but less extreme bankruptcy penalty
        
        # End of episode bonus
        if balance > initial_balance:
            return 10.0
        
        return 0.0
