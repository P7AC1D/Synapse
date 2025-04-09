"""Reward calculation for trading environment."""
import numpy as np
from typing import Union
from .actions import Action

class RewardCalculator:
    """Handles reward calculation for trading actions."""
    
    def __init__(self, env, max_hold_bars: int = 64, ema_alpha: float = 0.05,
                 direction_reward: float = 0.3, drawdown_penalty: float = 0.1):
        """Initialize reward calculator.
        
        Args:
            env: Trading environment instance
            max_hold_bars: Maximum bars to hold a position
            ema_alpha: Exponential moving average factor for direction tracking
            direction_reward: Reward multiplier for improving direction balance
            drawdown_penalty: Penalty multiplier for drawdown
        """
        self.env = env
        self.max_hold_bars = max_hold_bars
        self.ema_alpha = ema_alpha
        self.direction_reward = direction_reward
        self.drawdown_penalty = drawdown_penalty
        
        # Initialize direction tracking
        self.long_ratio = 0.5  # Start with balanced ratio
        self.trade_count = 0  # Track total trades for ratio calculation
        self.last_drawdown = 0.0  # Track drawdown changes

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: int) -> float:
        """Calculate reward based on action and position state."""
        reward = 0.0
        
        # Base reward for taking actions (to encourage exploration)
        if action in [Action.BUY, Action.SELL] and position_type == 0:
            reward += 0.1
        
        # Calculate position metrics with safety checks
        normalized_pnl = pnl / max(self.env.balance, 1e-8)
        risk_adjusted_pnl = normalized_pnl / max(atr * 0.01, 1e-8)
        
        # Reward for closing positions
        if action == Action.CLOSE and position_type != 0:
            if pnl > 0:
                # Reward profitable trades based on risk-adjusted return
                reward = risk_adjusted_pnl * 2.0  # Double the reward for good trades
                # Extra reward for profitable trades
                hold_efficiency = min(1.0, optimal_hold / max(current_hold, 1))
                reward = risk_adjusted_pnl * 2.0 * (1.0 + hold_efficiency)
            else:
                # More balanced loss handling
                hold_deviation = abs(current_hold - optimal_hold) / max(optimal_hold, 1)
                penalty_factor = 1.0 + min(0.3, hold_deviation)  # Reduced penalty cap to 30%
                reward = risk_adjusted_pnl * 1.5 * penalty_factor  # Less punishing
                
        # Penalize invalid actions, but less severely
        elif action in [Action.BUY, Action.SELL] and position_type != 0:
            reward = -0.5  # Reduced penalty for invalid actions
            
        # HOLD rewards based on position performance and hold time
        elif action == Action.HOLD and position_type != 0:
            hold_efficiency = min(1.0, optimal_hold / max(current_hold, 1))
            
            if pnl > 0:
                # Simpler reward structure for holding winners
                reward = risk_adjusted_pnl * 0.3 * max(0.5, hold_efficiency)
            else:
                # More lenient penalties for holding losers
                reward = risk_adjusted_pnl * 0.2 * max(0.3, hold_efficiency)
                
        # Add direction balance incentive and exploration reward
        elif action in [Action.BUY, Action.SELL] and position_type == 0:
            # Update long/short ratio
            is_long = (action == Action.BUY)
            self.trade_count += 1
            self.long_ratio = (1 - self.ema_alpha) * self.long_ratio + self.ema_alpha * (1.0 if is_long else 0.0)
            
            # Dynamic base reward scaled by ATR
            atr_scale = atr / self.env.balance
            volatility_factor = min(1.5, 1.0 / max(atr_scale, 0.0001))
            base_reward = 0.2 * volatility_factor
            reward += base_reward
            
            # Enhanced direction balance reward with trend consideration
            if (is_long and self.long_ratio < 0.4) or (not is_long and self.long_ratio > 0.6):
                reward += self.direction_reward * volatility_factor
            
            # Add trend alignment consideration
            trend_strength = self.env.raw_data['trend_strength'].iloc[self.env.current_step]
            if (is_long and trend_strength > 0.3) or (not is_long and trend_strength < -0.3):
                reward += 0.1  # Additional reward for trend-aligned trades
                
        # Track inactivity with milder penalty
        if hasattr(self, 'bars_since_trade'):
            self.bars_since_trade += 1
        else:
            self.bars_since_trade = 0
            
        if action in [Action.BUY, Action.SELL]:
            self.bars_since_trade = 0
            
        # Milder inactivity penalty
        if self.bars_since_trade > 200:  # Increased threshold
            excess_bars = min(self.bars_since_trade - 200, 1000)  # Cap at 1000 bars
            if excess_bars > 0:
                # Reduced penalty scaling
                scaled_penalty = 0.05 * np.log1p(excess_bars / 200)
                reward -= min(0.5, scaled_penalty)  # Reduced cap

        # Lighter drawdown penalty only for severe drawdowns
        current_drawdown = self.env.metrics.get_drawdown()
        if current_drawdown > 0.1:  # Only penalize >10% drawdowns
            drawdown_increase = max(0, current_drawdown - self.last_drawdown)
            reward -= drawdown_increase * self.drawdown_penalty
        self.last_drawdown = current_drawdown
        
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
            # More balanced loss penalty
            return return_pct * 2.0  # Double (not triple) the negative return as penalty
