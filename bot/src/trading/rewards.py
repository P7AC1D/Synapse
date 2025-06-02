"""
Overhauled reward system for active trading with proper risk management.
This version eliminates the "do nothing" strategy and encourages profitable trading.
"""
import numpy as np
from typing import Union, Optional
from .actions import Action

class RewardCalculator:
    """Handles reward calculation optimized for active trading strategies."""
    
    def __init__(self, env):
        """Initialize reward calculator with trading-focused incentives.
        
        Args:
            env: Trading environment instance
        """
        self.env = env
        self.previous_balance_high = env.initial_balance
        self.trade_entry_balance = env.initial_balance
        self.last_direction = None
        self.consecutive_holds = 0  # Track consecutive hold actions
        self.trades_in_episode = 0  # Track trading activity
        self.last_action = None
        
        # Track position metrics
        self.max_unrealized_pnl = 0.0
        self.position_opened_at_step = None
        self.total_hold_steps = 0
        
        # === REWARD CONFIGURATION ===
        # Core trading rewards
        self.PROFITABLE_TRADE_REWARD = 5.0      # Strong reward for winning trades
        self.LOSING_TRADE_PENALTY = -2.0        # Moderate penalty for losing trades
        self.MARKET_ENGAGEMENT_BONUS = 1.0      # Bonus for taking positions
        
        # Position management
        self.PROFIT_HOLD_REWARD = 1.5           # INCREASED: Strong reward for holding profitable positions
        self.LOSS_HOLD_PENALTY = -0.1           # REDUCED: Smaller penalty for holding losing positions
        self.PROFIT_PROTECTION_BONUS = 0.5      # Bonus for protecting unrealized profits
        self.SIGNIFICANT_PROFIT_THRESHOLD = 0.005 # REDUCED: 0.5% profit threshold for bonus rewards
        self.SIGNIFICANT_PROFIT_BONUS = 0.8     # INCREASED: Extra bonus for significantly profitable positions
        
        # Activity incentives
        self.HOLD_COST = -0.005                 # Small cost for inaction (accumulates)
        self.EXCESSIVE_HOLD_PENALTY = -0.02     # Penalty for excessive holding
        self.INACTIVITY_THRESHOLD = 50          # Steps before inactivity penalty
        
        # Risk management
        self.INVALID_ACTION_PENALTY = -2.0      # Penalty for invalid actions
        self.OVERTRADING_PENALTY = -0.5         # Penalty for too frequent trading
        self.MIN_POSITION_HOLD = 3              # Minimum bars to hold position
        
        # Performance bonuses
        self.NEW_HIGH_BONUS = 2.0               # Bonus for new equity highs
        self.CONSISTENCY_BONUS = 0.3            # Bonus for consistent performance
        self.RISK_ADJUSTED_MULTIPLIER = 1.5     # Multiplier for risk-adjusted returns
        
    def _calculate_position_quality_score(self, pnl: float, hold_time: int, atr: float) -> float:
        """Calculate quality score for position management."""
        if hold_time == 0:
            return 0.0
            
        # Base score from PnL relative to ATR (market volatility)
        pnl_atr_ratio = pnl / (atr * self.trade_entry_balance) if atr > 0 else 0
        
        # FIXED: Time efficiency that encourages holding profitable positions longer
        # For profitable trades, don't penalize holding (flat bonus)
        # For losing trades, encourage quick closure
        if pnl > 0:
            # Profitable positions: flat time efficiency (no decay for first 50 bars)
            time_efficiency = max(0.7, 1.0 - max(0, (hold_time - 50) / 100.0))
        else:
            # Losing positions: encourage quick closure
            time_efficiency = max(0.1, 1.0 - (hold_time / 30.0))
        
        # Combine scores
        quality_score = pnl_atr_ratio * time_efficiency
        return np.clip(quality_score, -2.0, 2.0)
        
    def _calculate_market_timing_bonus(self, action: int, atr: float) -> float:
        """Reward good market timing based on volatility."""
        if action not in [Action.BUY, Action.SELL]:
            return 0.0
            
        # Check recent ATR trend
        current_idx = self.env.current_step
        if current_idx < 20:
            return 0.0
            
        recent_atr = np.mean(self.env.prices['atr'][max(0, current_idx-10):current_idx+1])
        longer_atr = np.mean(self.env.prices['atr'][max(0, current_idx-20):current_idx-10])
        
        # Bonus for entering positions when volatility is increasing
        if recent_atr > longer_atr * 1.1:
            return 0.3
        elif recent_atr < longer_atr * 0.9:
            return -0.1  # Small penalty for entering during low volatility
            
        return 0.0
        
    def _calculate_risk_management_score(self, action: int, pnl: float, hold_time: int) -> float:
        """Calculate risk management component of reward."""
        score = 0.0
        
        # Reward profit protection
        if action == Action.CLOSE and pnl > 0:
            # Bonus for taking profits at good levels
            profit_ratio = pnl / self.trade_entry_balance
            if profit_ratio > 0.01:  # > 1% profit
                score += self.PROFIT_PROTECTION_BONUS * min(profit_ratio * 10, 2.0)
                
        # Penalty for holding losing positions too long
        if pnl < 0 and hold_time > 30:
            loss_ratio = abs(pnl) / self.trade_entry_balance
            score -= loss_ratio * 2.0  # Escalating penalty
            
        # Reward cutting losses quickly
        if action == Action.CLOSE and pnl < 0 and hold_time < 10:
            score += 0.2  # Small bonus for quick loss cutting
            
        return score

    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None,
                        invalid_action: bool = False) -> float:
        """Calculate comprehensive reward encouraging active profitable trading."""
        
        # Handle invalid actions with strong penalty
        if invalid_action:
            return self.INVALID_ACTION_PENALTY
            
        total_reward = 0.0
        
        # Track consecutive actions
        if action == Action.HOLD:
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
            
        # === CORE ACTION REWARDS ===
        
        if position_type != 0:  # We have an open position
            # Update max unrealized PnL tracking
            if pnl > self.max_unrealized_pnl:
                self.max_unrealized_pnl = pnl
                # Small bonus for increasing unrealized profits
                total_reward += 0.1
            else:
                # Penalty for giving back significant profits
                profit_drawdown = (self.max_unrealized_pnl - pnl) / max(self.trade_entry_balance, 1)
                if profit_drawdown > 0.005:  # > 0.5% drawdown from peak
                    total_reward -= profit_drawdown * 5.0
            
            if action == Action.HOLD:
                # Position holding rewards/penalties
                if pnl > 0:
                    # FIXED: Ensure ALL profitable positions get positive hold rewards
                    profit_ratio = pnl / self.trade_entry_balance
                    hold_decay = max(0.8, 1.0 - max(0, (current_hold - 30) / 200.0))  # FIXED: Much slower decay, no penalty for first 30 bars
                    
                    # Base profitable hold reward
                    base_reward = self.PROFIT_HOLD_REWARD * hold_decay
                    
                    # Add significant profit bonus
                    profit_bonus = 0.0
                    if profit_ratio > self.SIGNIFICANT_PROFIT_THRESHOLD:
                        profit_bonus = self.SIGNIFICANT_PROFIT_BONUS * min(profit_ratio * 50, 5.0)  # INCREASED multiplier
                    
                    # CRITICAL: Ensure minimum positive reward for ANY profitable position
                    total_profit_reward = base_reward + profit_bonus
                    if total_profit_reward < 0.1:  # Guarantee minimum positive reward
                        total_profit_reward = 0.1
                        
                    total_reward += total_profit_reward
                else:
                    # REDUCED: Less aggressive penalty for holding losing positions  
                    loss_penalty_multiplier = min(1.5, 1.0 + (current_hold / 40.0))  # REDUCED: Slower escalation
                    total_reward += self.LOSS_HOLD_PENALTY * loss_penalty_multiplier
                    
                # Add risk management score
                total_reward += self._calculate_risk_management_score(action, pnl, current_hold)
                    
            elif action == Action.CLOSE:
                # Trade completion rewards
                hold_time = current_hold
                
                if pnl > 0:
                    # Profitable trade - strong positive reward
                    total_reward += self.PROFITABLE_TRADE_REWARD
                    
                    # Quality bonus based on position management
                    quality_score = self._calculate_position_quality_score(pnl, hold_time, atr)
                    total_reward += quality_score
                    
                    # Risk-adjusted return bonus
                    risk_adjusted_return = (pnl / self.trade_entry_balance) * self.RISK_ADJUSTED_MULTIPLIER
                    total_reward += risk_adjusted_return * 10.0  # Scale up
                    
                else:
                    # Losing trade - moderate penalty
                    total_reward += self.LOSING_TRADE_PENALTY
                    
                    # Reduce penalty for quick loss cutting
                    if hold_time < self.MIN_POSITION_HOLD * 2:
                        total_reward += 0.5  # Partial penalty reduction
                
                # Add risk management score
                total_reward += self._calculate_risk_management_score(action, pnl, hold_time)
                
                # Update tracking
                self.trades_in_episode += 1
                self.last_direction = position_type
                self.max_unrealized_pnl = 0.0
                
        else:  # No position open
            if action in [Action.BUY, Action.SELL]:
                # Opening new position
                self.trade_entry_balance = self.env.balance
                self.max_unrealized_pnl = 0.0
                self.position_opened_at_step = self.env.current_step
                
                # Market engagement bonus
                total_reward += self.MARKET_ENGAGEMENT_BONUS
                
                # Market timing bonus
                timing_bonus = self._calculate_market_timing_bonus(action, atr)
                total_reward += timing_bonus
                
                # Prevent overtrading
                if self.trades_in_episode > 5:  # More than 5 trades in episode
                    total_reward += self.OVERTRADING_PENALTY
                    
            elif action == Action.HOLD:
                # No position, holding - apply inactivity cost
                total_reward += self.HOLD_COST
                
                # Escalating penalty for excessive inactivity
                if self.consecutive_holds > self.INACTIVITY_THRESHOLD:
                    excess_holds = self.consecutive_holds - self.INACTIVITY_THRESHOLD
                    total_reward += self.EXCESSIVE_HOLD_PENALTY * (excess_holds / 10.0)
        
        # === PERFORMANCE BONUSES ===
        
        # New equity high bonus
        current_equity = self.env.balance + (pnl if position_type != 0 else 0)
        if current_equity > self.previous_balance_high:
            total_reward += self.NEW_HIGH_BONUS
            self.previous_balance_high = current_equity
            
        # Consistency bonus for maintaining profitable trading
        if self.trades_in_episode >= 3:
            win_rate = self.env.metrics.win_count / max(len(self.env.metrics.trades), 1)
            if win_rate >= 0.6:  # 60%+ win rate
                total_reward += self.CONSISTENCY_BONUS
        
        # Update tracking
        self.last_action = action
        self.total_hold_steps += 1 if action == Action.HOLD else 0
        
        return float(total_reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate comprehensive terminal reward based on overall performance."""
        
        # Bankruptcy penalty
        if balance <= 0:
            return -50.0  # Severe penalty for account blow-up
        
        # Calculate overall performance metrics
        total_return = (balance - initial_balance) / initial_balance
        
        # Base terminal reward
        if total_return > 0.1:  # > 10% return
            base_reward = 20.0
        elif total_return > 0.05:  # > 5% return
            base_reward = 10.0
        elif total_return > 0:  # Profitable
            base_reward = 5.0
        elif total_return > -0.05:  # Small loss
            base_reward = -2.0
        else:  # Significant loss
            base_reward = -10.0
            
        # Performance multipliers
        performance_multiplier = 1.0
        
        # Trading activity bonus/penalty
        activity_ratio = self.trades_in_episode / max(self.total_hold_steps / 100, 1)
        if activity_ratio > 0.1:  # Active trading
            performance_multiplier += 0.5
        elif activity_ratio < 0.02:  # Too passive
            performance_multiplier -= 0.3
            
        # Risk management bonus
        if hasattr(self.env.metrics, 'max_drawdown'):
            max_dd = self.env.metrics.max_drawdown
            if max_dd < 0.1:  # < 10% max drawdown
                performance_multiplier += 0.3
            elif max_dd > 0.3:  # > 30% max drawdown
                performance_multiplier -= 0.5
                
        final_reward = base_reward * max(0.1, performance_multiplier)
        
        return float(final_reward)
        
    def reset_episode_tracking(self):
        """Reset episode-specific tracking variables."""
        self.consecutive_holds = 0
        self.trades_in_episode = 0
        self.last_action = None
        self.total_hold_steps = 0
        self.max_unrealized_pnl = 0.0
        self.position_opened_at_step = None
