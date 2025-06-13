"""Trade metrics and statistics tracking."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

class MetricsTracker:
    """Tracks and calculates trading metrics and statistics."""

    def __init__(self, initial_balance: float):
        """Initialize metrics tracker.
        
        Args:
            initial_balance: Starting account balance
        """
        self.initial_balance = initial_balance
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.trades: List[Dict[str, Any]] = []
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.current_unrealized_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        # Reset streak counters
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'current_direction': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0
        }
        
        # Track histories for accurate drawdown calculations
        self.balance_history = [self.initial_balance]  # Balance history points
        self.balance_peaks = [self.initial_balance]    # Running balance peaks
        
        self.equity_history = [self.initial_balance]   # Total equity points
        self.equity_peaks = [self.initial_balance]     # Running equity peaks
        
        # Maximum drawdowns seen
        self.max_balance_dd = 0.0  # Maximum balance drawdown
        self.max_equity_dd = 0.0   # Maximum equity drawdown

    def add_trade(self, trade_info: Dict[str, Any]) -> None:
        """Add a completed trade and update metrics.
        
        Args:
            trade_info: Dictionary containing trade details
        """
        self.trades.append(trade_info)
        
        # Update win/loss counts and streaks
        pnl = trade_info['pnl']
        if pnl > 0 and abs(pnl) >= 1e-8:  # Clear win
            self.win_count += 1
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        else:  # Loss or zero PnL
            self.loss_count += 1
            self.current_loss_streak += 1
            self.current_win_streak = 0
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)
            
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update trading metrics based on completed trades."""
        if not self.trades:
            return
            
        # Use PnL threshold for win/loss classification
        winning_trades = [t for t in self.trades if t['pnl'] > 0 and abs(t['pnl']) >= 1e-8]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0 or abs(t['pnl']) < 1e-8]
        
        self.metrics.update({
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0.0,
            'avg_profit': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0,
            'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0,
            'max_consecutive_wins': self.max_win_streak,
            'max_consecutive_losses': self.max_loss_streak,
            'current_consecutive_wins': self.current_win_streak,
            'current_consecutive_losses': self.current_loss_streak
        })

    def update_balance(self, pnl: float) -> None:
        """Update account balance and track balance peaks/drawdowns.
        
        Args:
            pnl: Profit/loss to add to balance
        """
        self.balance += pnl
        self.balance_history.append(self.balance)
        
        # Update balance peaks
        if not self.balance_peaks:
            self.balance_peaks.append(self.balance)
        else:
            self.balance_peaks.append(max(self.balance_peaks[-1], self.balance))
            
        # Track max balance and calculate drawdown
        peak = self.balance_peaks[-1]
        if peak > 0:
            dd = (peak - self.balance) / peak
            self.max_balance_dd = max(self.max_balance_dd, dd)
            
        # Update max balance for backward compatibility
        self.max_balance = peak
        self.current_unrealized_pnl = 0

    def update_unrealized_pnl(self, unrealized_pnl: float) -> None:
        """Update unrealized PnL and track equity peaks/drawdowns.
        
        Args:
            unrealized_pnl: Current unrealized profit/loss
        """
        self.current_unrealized_pnl = unrealized_pnl
        current_equity = self.balance + unrealized_pnl
        current_dd = 1 - (current_equity / self.balance)
        if current_dd < 0.0:
            current_dd = 0.0

        self.max_equity_dd = max(self.max_equity_dd, current_dd)

    def get_drawdown(self) -> float:
        """Calculate current drawdown percentage based on balance peaks.
        
        Returns:
            Current drawdown as a percentage
        """
        if not self.balance_history:
            return 0.0
            
        peak = self.balance_peaks[-1] if self.balance_peaks else self.balance
        
        if peak <= 0:
            return 1.0
            
        current_dd = (peak - self.balance) / peak if peak > 0 else 0.0
        return max(current_dd, self.max_balance_dd)

    def get_position_metrics(self) -> Dict[str, Any]:
        """Get current position metrics.
        
        Returns:
            Dictionary of current position metrics
        """
        if not self.trades:
            return {}

        latest_trade = self.trades[-1]
        
        return {
            "direction": "long" if latest_trade["direction"] == 1 else "short",
            "entry_price": latest_trade["entry_price"],
            "lot_size": latest_trade["lot_size"],
            "profit_points": latest_trade.get("profit_points", 0.0),
            "hold_time": latest_trade.get("hold_time", 0)
        }

    def get_equity_drawdown(self) -> float:
        """Calculate current equity drawdown percentage including unrealized PnL.
        
        Returns:
            Current equity drawdown as a percentage
        """            
        current_equity = self.balance + self.current_unrealized_pnl
        current_dd = 1 - (current_equity / self.balance)
        if current_dd < 0.0:
            current_dd = 0.0
        
        return current_dd
    
    def get_max_equity_drawdown(self) -> float:
        """Get maximum equity drawdown seen.
        
        Returns:
            Maximum equity drawdown as a percentage
        """
        return self.max_equity_dd
    
    def get_balance_drawdown(self) -> float:
        """Calculate current balance drawdown percentage.
        
        Returns:
            Current balance drawdown as a percentage
        """
        if not self.balance_history:
            return 0.0
            
        peak = self.balance_peaks[-1] if self.balance_peaks else self.balance
        
        if peak <= 0:
            return 1.0
            
        current_dd = (peak - self.balance) / peak if peak > 0 else 0.0
        return max(current_dd, self.max_balance_dd)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Create base summary with default values (for the no-trade case)
        summary = {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": self.balance - self.initial_balance,
            "return_pct": ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_points": 0.0,
            "avg_loss_points": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            
            # Risk metrics
            "max_drawdown_pct": self.get_drawdown() * 100,
            "max_equity_drawdown_pct": self.get_max_equity_drawdown() * 100,
            "current_equity_drawdown_pct": self.get_equity_drawdown() * 100,
            "current_drawdown_pct": self.get_balance_drawdown() * 100,
            
            # Directional metrics
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            
            # Hold time analysis
            "avg_hold_time": 0,
            "win_hold_time": 0,
            "loss_hold_time": 0,
            
            # Streak metrics
            "max_consecutive_wins": self.max_win_streak,
            "max_consecutive_losses": self.max_loss_streak,
            "current_consecutive_wins": self.current_win_streak,
            "current_consecutive_losses": self.current_loss_streak
        }
          # Initialize empty DataFrames to avoid UnboundLocalError
        winning_trades = pd.DataFrame()
        losing_trades = pd.DataFrame()
        
        # If we have trades, update the summary with actual values
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df["pnl"].apply(lambda x: x > 0 and abs(x) >= 1e-8)]
            losing_trades = trades_df[trades_df["pnl"].apply(lambda x: x <= 0 or abs(x) < 1e-8)]
            
            long_trades = trades_df[trades_df["direction"] == 1]
            short_trades = trades_df[trades_df["direction"] == -1]
            # Apply the same PnL threshold to directional wins
            long_wins = long_trades[long_trades["pnl"].apply(lambda x: x > 0 and abs(x) >= 1e-8)]
            short_wins = short_trades[short_trades["pnl"].apply(lambda x: x > 0 and abs(x) >= 1e-8)]

            # Calculate Sharpe ratio based on trade returns
            sharpe_ratio = 0.0
            if len(trades_df) > 1:  # Need at least 2 trades for standard deviation
                # Calculate returns as percentage of initial balance for each trade
                trade_returns = trades_df["pnl"] / self.initial_balance
                mean_return = trade_returns.mean()
                std_return = trade_returns.std()
                
                # Sharpe ratio = (mean return - risk free rate) / std deviation
                # Assuming risk-free rate of 0 for simplicity (can be adjusted)
                if std_return > 0:
                    sharpe_ratio = mean_return / std_return
                    # Annualize the Sharpe ratio (assuming daily trades, adjust as needed)
                    # For trading systems, we often use sqrt(252) for daily or sqrt(number of periods per year)
                    sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualized for daily trading

            summary.update({
                "total_trades": len(trades_df),
                "win_rate": len(winning_trades) / len(trades_df) * 100,
                "avg_win": winning_trades["pnl"].mean() if not winning_trades.empty else 0.0,
                "avg_loss": losing_trades["pnl"].mean() if not losing_trades.empty else 0.0,
                "avg_win_points": winning_trades["profit_points"].mean() if not winning_trades.empty else 0.0,
                "avg_loss_points": losing_trades["profit_points"].mean() if not losing_trades.empty else 0.0,
                "profit_factor": abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum()) if not losing_trades.empty else float('inf'),
                "sharpe_ratio": sharpe_ratio,
                
                # Directional metrics
                "long_trades": len(long_trades),
                "short_trades": len(short_trades),
                "long_win_rate": len(long_wins) / len(long_trades) * 100 if not long_trades.empty else 0.0,
                "short_win_rate": len(short_wins) / len(short_trades) * 100 if not short_trades.empty else 0.0,
                
                # Hold time analysis
                "avg_hold_time": trades_df["hold_time"].mean() if "hold_time" in trades_df else 0,
                "win_hold_time": winning_trades["hold_time"].mean() if not winning_trades.empty and "hold_time" in winning_trades else 0,
                "loss_hold_time": losing_trades["hold_time"].mean() if not losing_trades.empty and "hold_time" in losing_trades else 0
            })
        
        return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in summary.items()}
