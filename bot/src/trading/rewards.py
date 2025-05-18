"""Reward calculation for trading environment with sparse rewards."""
import numpy as np
from typing import Union, Optional
from .actions import Action

class RewardCalculator:
    """Handles enhanced reward calculation for trading actions with multiple components."""
    
    def __init__(self, env):
        """Initialize reward calculator.
        
        Args:
            env: Trading environment instance
        """
        self.env = env
        self.trade_entry_balance = env.initial_balance  # Track balance at trade entry
        
        # Enhanced reward constants
        self.INVALID_ACTION_PENALTY = -1.0  # Penalty for invalid actions
        self.TRANSACTION_COST = 0.001  # Increased from 0.0005 to limit overtrading
        self.TIME_STEP_COST = 0.00001  # Time decay cost
        self.TIME_PRESSURE_THRESHOLD = 100  # For observation scaling
        self.PNL_SCALE = 2.0  # Increased from 1.5 for stronger profit signals
        self.HOLD_REWARD_SCALE = 0.1  # Progressive holding reward
        self.TREND_ALIGNMENT_BONUS = 0.2  # Trend following bonus
        self.RISK_ADJUSTMENT_WINDOW = 20  # Window for Sharpe calculation
        
        # State tracking
        self.previous_balance_high = env.initial_balance
        self.last_direction = None
        self.max_unrealized_pnl = 0.0
        self.bars_since_consolidation = 0
        self.returns_history = []  # For Sharpe ratio calculation
        
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
        self.returns_history = []
        
    def _calculate_trend(self, window: int = 20) -> float:
        """Calculate market trend direction using SMA."""
        if len(self.env.features_df) < window:
            return 0.0
            
        prices = self.env.prices['close'][max(0, self.env.current_step - window):self.env.current_step + 1]
        sma = np.mean(prices)
        current_price = prices[-1]
        return (current_price - sma) / sma

    def _calculate_rolling_sharpe(self) -> float:
        """Calculate rolling Sharpe ratio for risk adjustment."""
        if len(self.returns_history) < self.RISK_ADJUSTMENT_WINDOW:
            return 0.0
            
        returns = np.array(self.returns_history[-self.RISK_ADJUSTMENT_WINDOW:])
        if np.std(returns) == 0:
            return 0.0
            
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252 * 96)  # Annualized

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate enhanced rewards with multiple components."""
        # Handle invalid actions first
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
            
        reward = -self.TIME_STEP_COST  # Base time decay
        
        # Transaction costs
        if action in [Action.BUY, Action.SELL]:
            reward -= self.TRANSACTION_COST
            self.trade_entry_balance = self.env.balance
            
            # Add trend alignment bonus
            trend = self._calculate_trend()
            if (action == Action.BUY and trend > 0) or (action == Action.SELL and trend < 0):
                reward += self.TREND_ALIGNMENT_BONUS
                
        # Enhanced PnL rewards
        elif action == Action.CLOSE and position_type != 0:
            # Stronger PnL scaling
            scaled_pnl = self.PNL_SCALE * (pnl / self.trade_entry_balance)
            
            # Risk-adjusted component
            if pnl != 0:
                self.returns_history.append(pnl / self.trade_entry_balance)
                sharpe = self._calculate_rolling_sharpe()
                risk_multiplier = 1.0 + max(0, sharpe) * 0.1
                scaled_pnl *= risk_multiplier
                
            reward += scaled_pnl
            reward -= self.TRANSACTION_COST
            
        # Progressive hold reward for profitable positions
        elif position_type != 0 and pnl > 0:
            hold_factor = min(current_hold / self.TIME_PRESSURE_THRESHOLD, 1.0)
            hold_reward = self.HOLD_REWARD_SCALE * hold_factor * (pnl / self.trade_entry_balance)
            reward += hold_reward
        
        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -5.0  # Bankruptcy penalty
        
        # End of episode reward based on total return
        return (balance - initial_balance) / initial_balance
