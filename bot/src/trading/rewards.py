"""Psychological reward system modeling human trading behavior."""
import numpy as np
import logging
from typing import Dict, Any, Optional
from .actions import Action

class RewardCalculator:
    """Calculates rewards based on human trading psychology principles.
    
    This reward system models key aspects of successful human trading behavior:
    1. Entry Quality - Rewards trades aligned with market context
    2. Exit Quality - Rewards intelligent exit timing and profit taking
    3. Emotional Control - Rewards disciplined trading behavior
    4. Learning Reinforcement - Rewards consistent improvement and adaptation
    """
    
    def __init__(self, env):
        """Initialize reward calculator with psychological components.
        
        Args:
            env: Trading environment instance
        """
        self.env = env
        self.logger = logging.getLogger(__name__)
        
        # Core reward constants
        self.INVALID_ACTION_PENALTY = -0.05  # Reduced penalty
        self.MIN_TRADE_INTERVAL = 5  # Reduced to encourage more frequent trading
        self.OPTIMAL_HOLD_TIME = 20  # Target hold time for bell curve
        self.VOLATILITY_THRESHOLD = 0.8  # High volatility threshold for opportunity detection
        
        # Component weights
        self.ENTRY_QUALITY_WEIGHT = 0.3  # Increased to encourage more trading
        self.EXIT_QUALITY_WEIGHT = 0.3
        self.EMOTIONAL_CONTROL_WEIGHT = 0.2  # Reduced to lower trading barriers
        self.LEARNING_WEIGHT = 0.2
        
        # Entry rewards
        self.BASE_ENTRY_REWARD = 0.2  # Base reward for taking trades
        self.VOLATILITY_SCALE = 0.3   # Reward for trading in volatile conditions
        self.TREND_ALIGNMENT_SCALE = 0.3  # Reduced from 0.4 for more counter-trend trades
        
        # Exit rewards
        self.PROFIT_BASE_SCALE = 0.4  # Base scale for profitable exits
        self.HOLD_TIME_SCALE = 0.3    # Scale for hold time optimization
        
        # Emotional control
        self.FOMO_PENALTY_SCALE = 0.1  # Reduced from 0.2
        self.PATIENCE_SCALE = 0.2      # Increased for recovery trades
        self.MISSED_OPP_PENALTY = 0.1  # New penalty for missing good setups
        
        self.reset()
        
    def reset(self, initial_balance: Optional[float] = None, min_bars: Optional[int] = None) -> None:
        """Reset reward calculator state.
        
        Args:
            initial_balance: Optional new initial balance
            min_bars: Optional minimum bars per episode
        """
        self.previous_action_step = 0
        self.last_pnl = 0.0
        self.trade_count_window = []
        
    def calculate_reward(self, action: int, position_type: int, 
                        pnl: float, atr: float, current_hold: int,
                        optimal_hold: Optional[int] = None, 
                        invalid_action: bool = False) -> float:
        """Calculate comprehensive psychological reward.
        
        Args:
            action: Current action taken
            position_type: Current position type
            pnl: Current profit/loss
            atr: Average True Range value
            current_hold: Number of bars position has been held
            
        Returns:
            float: Calculated reward incorporating psychological factors
        """
        # Handle invalid actions first
        if invalid_action:
            self.logger.debug("Invalid action penalty applied")
            return self.INVALID_ACTION_PENALTY
            
        reward = 0.0
        self.last_pnl = pnl
        
        # Calculate component rewards
        # Calculate market opportunity
        volatility = self.env.features_df['atr'].iloc[self.env.current_step]
        vol_breakout = self.env.features_df['volatility_breakout'].iloc[self.env.current_step]
        high_volatility = vol_breakout > self.VOLATILITY_THRESHOLD
        
        # Calculate rewards
        entry_quality = (self._calculate_entry_quality() + self.BASE_ENTRY_REWARD) if action in [Action.BUY, Action.SELL] else 0.0
        exit_quality = self._calculate_exit_quality(pnl) if action == Action.CLOSE else 0.0
        emotional_control = self._calculate_emotional_control(high_volatility)
        learning_reward = self._calculate_learning_reward(pnl)
        
        # Combine weighted components
        reward = (
            self.ENTRY_QUALITY_WEIGHT * entry_quality +
            self.EXIT_QUALITY_WEIGHT * exit_quality +
            self.EMOTIONAL_CONTROL_WEIGHT * emotional_control +
            self.LEARNING_WEIGHT * learning_reward
        )
        
        self.logger.debug(
            f"Reward components: entry={entry_quality:.3f}, exit={exit_quality:.3f}, "
            f"emotional={emotional_control:.3f}, learning={learning_reward:.3f}"
        )
        
        return float(reward)
        
    def _calculate_entry_quality(self) -> float:
        """Calculate entry quality based on market context alignment.
        
        Returns:
            float: Entry quality reward [-1.0, 1.0]
        """
        # Get current market features
        current_features = {
            'rsi': self.env.features_df['rsi'].iloc[self.env.current_step],
            'trend': self.env.features_df['trend_strength'].iloc[self.env.current_step],
            'vol_breakout': self.env.features_df['volatility_breakout'].iloc[self.env.current_step]
        }
        
        if not self.env.current_position:
            return 0.0
            
        direction = self.env.current_position["direction"]
        
        # More balanced weighting between trend and counter-trend
        trend_alignment = 0.5 * (direction * current_features['trend'] + 1)
        momentum_quality = 0.5 * (direction * current_features['rsi'] + 1)
        
        # Increased emphasis on volatility timing
        vol_timing = abs(current_features['vol_breakout'] - 0.5) * 2
        
        # Counter-trend bonus when volatility is high
        counter_trend_bonus = 0.0
        if vol_timing > self.VOLATILITY_THRESHOLD:
            counter_trend_bonus = (1 - trend_alignment) * 0.2
        
        # Combine factors with more balanced weighting
        entry_quality = (
            self.TREND_ALIGNMENT_SCALE * trend_alignment +
            0.3 * momentum_quality +
            self.VOLATILITY_SCALE * vol_timing +
            counter_trend_bonus
        )
        
        self.logger.debug(
            f"Entry quality: trend={trend_alignment:.3f}, "
            f"momentum={momentum_quality:.3f}, vol={vol_timing:.3f}"
        )
        
        return entry_quality
        
    def _calculate_exit_quality(self, pnl: float) -> float:
        """Calculate exit quality based on profit target achievement and market conditions.
        
        Args:
            pnl: Profit/loss from the trade
            
        Returns:
            float: Exit quality reward [-1.0, 1.0]
        """
        if pnl == 0:  # No exit
            return 0.0
            
        # Get market features at exit
        volatility = self.env.features_df['atr'].iloc[self.env.current_step]
        vol_breakout = self.env.features_df['volatility_breakout'].iloc[self.env.current_step]
        trend = self.env.features_df['trend_strength'].iloc[self.env.current_step]
        
        # Scale PnL relative to initial balance
        profit_factor = np.clip(pnl / (self.env.initial_balance * 0.01), -1, 1)
        
        if pnl > 0:  # Profitable exit
            # Reward taking profits at volatility extremes
            exit_timing = abs(vol_breakout - 0.5) * 2
            # Bonus for exiting against weakening trend
            trend_exit = 1 - abs(trend)  # Higher reward for exiting when trend weakens
            
            exit_quality = profit_factor * (
                0.6 +                    # Base reward for profit
                0.2 * exit_timing +      # 20% weight on timing
                0.2 * trend_exit        # 20% weight on trend context
            )
        else:  # Loss exit
            # Bell curve for hold time reward
            optimal_hold_factor = np.exp(-0.5 * ((self.env.current_hold_time - self.OPTIMAL_HOLD_TIME) / (self.OPTIMAL_HOLD_TIME/2))**2)
            # Reward cutting losses in strong counter-trend
            trend_alignment = -np.sign(pnl) * trend  # Positive when trend against position
            
            # Dynamic exit quality based on market conditions
            exit_quality = profit_factor * (
                self.PROFIT_BASE_SCALE +  # Base reward/penalty
                self.HOLD_TIME_SCALE * optimal_hold_factor +  # Optimal hold time curve
                0.3 * trend_alignment  # Market context
            )
            
        self.logger.debug(
            f"Exit quality: profit={profit_factor:.3f}, "
            f"timing={exit_timing if pnl > 0 else optimal_hold_factor:.3f}, "
            f"hold_time={self.env.current_hold_time}/{self.OPTIMAL_HOLD_TIME} bars"
        )
            
        return exit_quality
        
    def _calculate_emotional_control(self, high_volatility: bool) -> float:
        """Calculate emotional discipline reward based on trading behavior.
        
        Args:
            high_volatility: Whether current market volatility is high
            
        Returns:
            float: Emotional control reward [-1.0, 1.0]
        """
        reward = 0.0
        
        # Update trade frequency tracking
        self.trade_count_window = [t for t in self.trade_count_window 
                                 if t > self.env.current_step - self.MIN_TRADE_INTERVAL]
        if self.env.current_position and len(self.trade_count_window) == 0:
            self.trade_count_window.append(self.env.current_step)
            
        # FOMO control - penalize frequent trading
        trade_frequency = len(self.trade_count_window)
        fomo_penalty = -0.2 * max(0, trade_frequency - 2)
        
        # Patience after losses
        if self.env.metrics.current_loss_streak > 0:
            patience_reward = min(self.env.metrics.current_loss_streak * 0.1, 0.5)
            reward += patience_reward
            
        # Position holding discipline
        if self.env.current_position:
            unrealized_pnl = self.env.metrics.current_unrealized_pnl
            hold_time = self.env.current_hold_time
            
            if unrealized_pnl > 0:
                # Reward holding winners with diminishing returns
                hold_reward = min(hold_time / 10, 0.3)
                reward += hold_reward
            else:
                # More forgiving hold penalty in volatile conditions
                if high_volatility:
                    hold_penalty = -min(max(0, hold_time - 10) * 0.01, 0.2)
                else:
                    hold_penalty = -min(max(0, hold_time - 5) * 0.02, 0.3)
                reward += hold_penalty
                
        # Missed opportunity penalty
        elif high_volatility:
            reward -= self.MISSED_OPP_PENALTY
                
        emotional_reward = reward + fomo_penalty
        
        self.logger.debug(
            f"Emotional control: fomo={fomo_penalty:.3f}, "
            f"patience={reward-fomo_penalty:.3f}"
        )
        
        return np.clip(emotional_reward, -1, 1)
        
    def _calculate_learning_reward(self, pnl: float) -> float:
        """Calculate learning reinforcement reward based on adaptation and improvement.
        
        Args:
            pnl: Profit/loss from the trade
            
        Returns:
            float: Learning reward [-1.0, 1.0]
        """
        # Base reward starts negative to encourage action
        reward = -0.05
        
        # Enhanced streak rewards
        if pnl > 0:
            streak = self.env.metrics.current_win_streak
            streak_reward = (1 - 0.8 ** streak) * 0.6  # Faster growth, higher cap
            reward += streak_reward
            
            # Additional reward for consistent small wins
            if 0 < pnl < (self.env.initial_balance * 0.01):
                reward += 0.1
                
            # Increased recovery bonus
            if pnl > 0 and self.env.metrics.current_loss_streak > 1:  # Reduced threshold
                recovery_bonus = 0.4  # Increased bonus
                reward += recovery_bonus
            
        # Penalty for consecutive large losses
        if pnl < 0:
            large_loss = abs(pnl) > (self.env.initial_balance * 0.02)
            if large_loss and self.env.metrics.current_loss_streak > 1:
                reward -= 0.2 * self.env.metrics.current_loss_streak
                
        self.logger.debug(
            f"Learning reward: streak={streak_reward if pnl > 0 else 0:.3f}, "
            f"total={reward:.3f}"
        )
        
        return np.clip(reward, -1, 1)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state based on overall performance.
        
        Args:
            balance: Final account balance
            initial_balance: Initial account balance
            
        Returns:
            float: Terminal reward value
        """
        if balance <= 0:
            return -2.0  # Bankruptcy penalty
            
        # Calculate return metrics
        total_return = (balance - initial_balance) / initial_balance
        
        # Get consistency metrics
        win_rate = self.env.metrics.metrics['win_rate']
        profit_factor = max(0, self.env.metrics.get_performance_summary()['profit_factor'])
        max_drawdown = self.env.metrics.get_max_equity_drawdown()
        
        # Reward consistent profitable trading
        # Calculate trade activity score (reward more active trading)
        total_trades = len(self.env.trades)
        trade_activity = min(total_trades / 20, 1.0)  # Target ~20 trades per episode
        
        if total_return > 0:
            terminal_reward = min(2.0, (
                0.35 * (total_return * 10) +       # 35% weight on returns
                0.25 * (win_rate / 100) +          # 25% weight on win rate
                0.25 * min(profit_factor, 3) / 3 + # 25% weight on profit factor
                0.15 * trade_activity              # 15% weight on trading activity
            ))
            
            # More forgiving drawdown penalty
            if max_drawdown > 0.15:  # Increased threshold to 15%
                terminal_reward *= (1.2 - max_drawdown)  # Less severe penalty
        else:
            terminal_reward = max(-2.0, total_return)
            
        self.logger.debug(
            f"Terminal reward: {terminal_reward:.3f} "
            f"(return={total_return:.1%}, win_rate={win_rate:.1f}%, "
            f"profit_factor={profit_factor:.2f})"
        )
            
        return float(terminal_reward)
