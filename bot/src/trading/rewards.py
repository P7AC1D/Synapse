"""
Simple risk-adjusted reward system focused on pure trading performance.
This eliminates complex incentive structures that may confuse the agent.
"""
import numpy as np
from typing import Union, Optional
from .actions import Action

class SimpleRewardCalculator:
    """Simple reward calculator focused on risk-adjusted returns."""
    
    def __init__(self, env):
        """Initialize simple reward calculator.
        
        Args:
            env: Trading environment instance
        """
        self.env = env
        self.initial_balance = env.initial_balance
        
        # Simple parameters
        self.INVALID_ACTION_PENALTY = -1.0
        self.RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
        
        # Track for Sharpe calculation
        self.returns = []
        self.episode_start_balance = env.initial_balance
        
    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate simple risk-adjusted reward.
        
        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
            position_type: Current position (0=none, 1=long, -1=short)
            pnl: Profit/loss from the action
            atr: Average True Range (volatility measure)
            current_hold: Current holding time
            optimal_hold: Optimal holding time (unused in simple version)
            invalid_action: Whether action was invalid
            
        Returns:
            Simple risk-adjusted reward
        """
        
        # Handle invalid actions
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
              # Only give rewards when trades are closed (realized P&L)
        if action == Action.CLOSE and pnl != 0:
            # Calculate return as percentage of current balance
            trade_return = pnl / self.env.balance
            
            # Risk-adjust using ATR (market volatility)
            if atr > 0:
                # Normalize return by volatility
                risk_adjusted_return = trade_return / (atr / self.env.prices['close'][self.env.current_step])
            else:
                risk_adjusted_return = trade_return
                
            # Scale the reward to reasonable magnitude
            reward = risk_adjusted_return * 100.0
            
            # Track returns for Sharpe calculation
            self.returns.append(trade_return)
            
            return float(reward)
        
        # All other actions get zero reward
        return 0.0

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate terminal reward based on overall performance.
        
        Args:
            balance: Final balance
            initial_balance: Starting balance
            
        Returns:
            Terminal reward based on risk-adjusted performance
        """
        
        # Bankruptcy penalty
        if balance <= 0:
            return -100.0
            
        # Calculate total return
        total_return = (balance - initial_balance) / initial_balance
        
        # Calculate Sharpe ratio if we have enough trades
        sharpe_bonus = 0.0
        if len(self.returns) >= 5:  # Need at least 5 trades
            returns_array = np.array(self.returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                # Annualized Sharpe ratio (assuming daily trades)
                sharpe_ratio = (mean_return - self.RISK_FREE_RATE/252) / std_return * np.sqrt(252)
                sharpe_bonus = max(0, sharpe_ratio) * 10.0  # Bonus for positive Sharpe
        
        # Base terminal reward
        base_reward = total_return * 50.0  # Scale total return
        
        return float(base_reward + sharpe_bonus)
        
    def reset_episode_tracking(self):
        """Reset episode-specific tracking variables."""
        self.returns = []
        self.episode_start_balance = self.env.initial_balance
