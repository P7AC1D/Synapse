import numpy as np 
from typing import Union, Optional 
from .actions import Action

""" Robust risk-adjusted reward system for trading, with risk management, directional diversity, and holding duration control. """ 

class SimpleRewardCalculator:
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
        self.PNL_SCALE = 1.5  # Reduced from 2.0 to temper in-sample profit bias
        self.HOLD_REWARD_SCALE = 0.5  # Increased hold reward scale
        self.HOLD_BONUS_EXP = 1.2     # Exponential bonus for longer holds
        self.SHORT_HOLD_PENALTY = -0.2  # Penalty for closing too soon
        self.LONG_HOLD_PENALTY = -0.2   # Penalty for holding too long
        self.MIN_HOLD_BARS_FOR_NO_PENALTY = 3
        self.MAX_HOLD_BARS_FOR_NO_PENALTY = 30  # After this, penalize for holding too long
        self.TREND_ALIGNMENT_BONUS = 0.2  # Trend following bonus
        self.RISK_ADJUSTMENT_WINDOW = 30  # Increased from 20 to smooth Sharpe adjustment
        self.DIVERSITY_BONUS = 0.2  # Max bonus for balanced long/short
        self.DIVERSITY_PENALTY = -0.2  # Penalty for extreme imbalance
        self.DRAWDOWN_PENALTY = -0.5   # Penalty for excessive drawdown
        self.DRAWDOWN_THRESHOLD = 0.05 # 5% drawdown threshold

        # State tracking
        self.previous_balance_high = env.initial_balance
        self.last_direction = None
        self.max_unrealized_pnl = 0.0
        self.bars_since_consolidation = 0
        self.returns_history = []  # For Sharpe ratio calculation
        self.long_trades = 0
        self.short_trades = 0

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
        self.long_trades = 0
        self.short_trades = 0

    def _calculate_trend(self, window: int = 20) -> float:
        """Calculate market trend direction using SMA."""
        if len(self.env.raw_data) < window:
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

    def _calculate_drawdown(self, balance: float) -> float:
        """Calculate current drawdown from previous balance high."""
        if balance > self.previous_balance_high:
            self.previous_balance_high = balance
            return 0.0
        drawdown = (self.previous_balance_high - balance) / self.previous_balance_high
        return drawdown

    def _calculate_directional_diversity(self) -> float:
        """Calculate diversity bonus/penalty based on long/short trade ratio."""
        total_trades = self.long_trades + self.short_trades
        if total_trades == 0:
            return 0.0
        ratio = min(self.long_trades, self.short_trades) / max(self.long_trades, self.short_trades)
        # If both sides are present, reward proportional to balance, else penalize
        if ratio >= 0.5:
            # Well balanced, full bonus
            return self.DIVERSITY_BONUS
        elif ratio > 0.2:
            # Somewhat balanced, partial bonus
            return self.DIVERSITY_BONUS * ratio
        else:
            # Very imbalanced, penalize
            return self.DIVERSITY_PENALTY * (1 - ratio)

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

            # Track trade direction for diversity bonus
            if action == Action.BUY:
                self.long_trades += 1
            elif action == Action.SELL:
                self.short_trades += 1

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

            # Penalty for closing trades too quickly
            if current_hold < self.MIN_HOLD_BARS_FOR_NO_PENALTY:
                reward += self.SHORT_HOLD_PENALTY
            # Penalty for holding too long (same for winners and losers)
            elif current_hold > self.MAX_HOLD_BARS_FOR_NO_PENALTY:
                reward += self.LONG_HOLD_PENALTY

        # Progressive hold reward for profitable positions
        elif position_type != 0 and pnl > 0:
            hold_factor = min(current_hold / self.TIME_PRESSURE_THRESHOLD, 1.0)
            # Nonlinear bonus for longer holds
            hold_bonus = (hold_factor ** self.HOLD_BONUS_EXP)
            hold_reward = self.HOLD_REWARD_SCALE * hold_bonus * (pnl / self.trade_entry_balance)
            reward += hold_reward

        # Drawdown penalty (applied at every step)
        drawdown = self._calculate_drawdown(self.env.balance)
        if drawdown > self.DRAWDOWN_THRESHOLD:
            reward += self.DRAWDOWN_PENALTY * (drawdown - self.DRAWDOWN_THRESHOLD)

        return float(reward)

    def calculate_terminal_reward(self, balance: float, initial_balance: float) -> float:
        """Calculate reward for terminal state."""
        if balance <= 0:
            return -5.0  # Bankruptcy penalty

        # Diversity bonus/penalty for using both long and short trades
        diversity_score = self._calculate_directional_diversity()

        # End of episode reward based on total return plus diversity score
        return (balance - initial_balance) / initial_balance + diversity_score

    def reset_episode_tracking(self):
        """Reset episode-specific tracking variables."""
        self.reset(self.env.initial_balance)
        self.long_trades = 0
        self.short_trades = 0
