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
        self.reset()

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.trades: List[Dict[str, Any]] = []
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.current_unrealized_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.metrics = {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'current_direction': 0
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
        
        if trade_info['pnl'] > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update trading metrics based on completed trades."""
        if not self.trades:
            return
            
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        self.metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0.0
        self.metrics['avg_profit'] = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        self.metrics['avg_loss'] = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

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

    def update_unrealized_pnl(self, unrealized_pnl: float) -> None:
        """Update unrealized PnL and track equity peaks/drawdowns.
        
        Args:
            unrealized_pnl: Current unrealized profit/loss
        """
        self.current_unrealized_pnl = unrealized_pnl
        current_equity = self.balance + unrealized_pnl
        
        # Track equity history and peaks
        self.equity_history.append(current_equity)
        if not self.equity_peaks:
            self.equity_peaks.append(current_equity)
        else:
            self.equity_peaks.append(max(self.equity_peaks[-1], current_equity))
            
        # Calculate current equity drawdown
        peak = self.equity_peaks[-1]
        if peak > 0:
            dd = (peak - current_equity) / peak
            self.max_equity_dd = max(self.max_equity_dd, dd)

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
            "profit_pips": latest_trade.get("profit_pips", 0.0),
            "hold_time": latest_trade.get("hold_time", 0)
        }

    def get_equity_drawdown(self) -> float:
        """Calculate current equity drawdown percentage including unrealized PnL.
        
        Returns:
            Current equity drawdown as a percentage
        """
        if not self.equity_history:
            return 0.0
            
        current_equity = self.balance + self.current_unrealized_pnl
        peak = self.equity_peaks[-1] if self.equity_peaks else current_equity
        
        if peak <= 0:
            return 1.0
            
        current_dd = (peak - current_equity) / peak if peak > 0 else 0.0
        return max(current_dd, self.max_equity_dd)
    
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
        if not self.trades:
            return {"total_trades": 0}

        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        long_trades = trades_df[trades_df["direction"] == 1]
        short_trades = trades_df[trades_df["direction"] == -1]
        long_wins = long_trades[long_trades["pnl"] > 0]
        short_wins = short_trades[short_trades["pnl"] > 0]

        summary = {
            "total_trades": len(trades_df),
            "win_rate": len(winning_trades) / len(trades_df) * 100,
            "total_pnl": self.balance - self.initial_balance,
            "return_pct": ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            "avg_win": winning_trades["pnl"].mean() if not winning_trades.empty else 0.0,
            "avg_loss": losing_trades["pnl"].mean() if not losing_trades.empty else 0.0,
            "profit_factor": abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum()) if not losing_trades.empty else float('inf'),
            
            # Risk metrics
            "max_drawdown_pct": self.get_drawdown() * 100,  # Balance-based drawdown
            "max_equity_drawdown_pct": self.get_max_equity_drawdown() * 100,  # Equity-based drawdown
            "current_equity_drawdown_pct": self.get_equity_drawdown() * 100,  # Current equity drawdown
            "current_drawdown_pct": self.get_balance_drawdown() * 100,  # Current balance drawdown
            
            # Directional metrics
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_win_rate": len(long_wins) / len(long_trades) * 100 if not long_trades.empty else 0.0,
            "short_win_rate": len(short_wins) / len(short_trades) * 100 if not short_trades.empty else 0.0,
            
            # Hold time analysis
            "avg_hold_time": trades_df["hold_time"].mean() if "hold_time" in trades_df else 0,
            "win_hold_time": winning_trades["hold_time"].mean() if "hold_time" in winning_trades else 0,
            "loss_hold_time": losing_trades["hold_time"].mean() if "hold_time" in losing_trades else 0
        }
        
        return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in summary.items()}
