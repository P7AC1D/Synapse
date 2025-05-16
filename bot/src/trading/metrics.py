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

    def reset(self, initial_balance: Optional[float] = None) -> None:
        """Reset all metrics to initial state.
        
        Args:
            initial_balance: Optional new initial balance. If None, uses previous initial_balance.
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
            
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
        
        # Reset drawdown tracking
        self.balance_history = [self.initial_balance]
        self.balance_peaks = [self.initial_balance]
        self.equity_history = [self.initial_balance]
        self.equity_peaks = [self.initial_balance]
        self.max_balance_dd = 0.0
        self.max_equity_dd = 0.0

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

            summary.update({
                "total_trades": len(trades_df),
                "win_rate": len(winning_trades) / len(trades_df) * 100,
                "avg_win": winning_trades["pnl"].mean() if not winning_trades.empty else 0.0,
                "avg_loss": losing_trades["pnl"].mean() if not losing_trades.empty else 0.0,
                "avg_win_points": winning_trades["profit_points"].mean() if not winning_trades.empty else 0.0,
                "avg_loss_points": losing_trades["profit_points"].mean() if not losing_trades.empty else 0.0,
                "profit_factor": abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum()) if not losing_trades.empty else float('inf'),
                
                # Directional metrics
                "long_trades": len(long_trades),
                "short_trades": len(short_trades),
                "long_win_rate": len(long_wins) / len(long_trades) * 100 if not long_trades.empty else 0.0,
                "short_win_rate": len(short_wins) / len(short_trades) * 100 if not short_trades.empty else 0.0,
                
                # Hold time analysis
                "avg_hold_time": trades_df["hold_time"].mean() if "hold_time" in trades_df else 0,
                "win_hold_time": winning_trades["hold_time"].mean() if "hold_time" in winning_trades else 0,
                "loss_hold_time": losing_trades["hold_time"].mean() if "hold_time" in losing_trades else 0
            })
        
        return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in summary.items()}

    def print_evaluation_metrics(self, phase: str = "Evaluation", timestep: Optional[int] = None, 
                               model: Optional['RecurrentPPO'] = None) -> None:
        """Print formatted evaluation metrics.
        
        Args:
            phase: Name of the evaluation phase (e.g. "Training", "Validation", "Test")
            timestep: Optional current timestep for progress tracking
            model: Optional model for printing network stats
        """
        metrics = self.get_performance_summary()
        
        step_info = f" (Timestep {timestep:,d})" if timestep is not None else ""
        print(f"\n===== {phase} Metrics{step_info} =====")
        
        # Account summary
        print(f"  Balance: ${self.balance:.2f} (${self.balance + self.current_unrealized_pnl:.2f})")
        print(f"  Unrealized PnL: {self.current_unrealized_pnl:.2f}")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}% ({metrics['max_equity_drawdown_pct']:.2f}%)")
        print(f"  Total Reward: {metrics['total_pnl']:.2f}")
        
        # Network stats (if model provided)
        if model is not None:
            try:
                training_stats = {
                    # Value network stats
                    "value_loss": float(model.logger.name_to_value.get('train/value_loss', 0.0)),
                    "explained_variance": float(model.logger.name_to_value.get('train/explained_variance', 0.0)),
                    # Policy network stats
                    "policy_loss": float(model.logger.name_to_value.get('train/policy_gradient_loss', 0.0)),
                    "entropy_loss": float(model.logger.name_to_value.get('train/entropy_loss', 0.0)),
                    "approx_kl": float(model.logger.name_to_value.get('train/approx_kl', 0.0)),
                    # Training stats
                    "total_loss": float(model.logger.name_to_value.get('train/loss', 0.0)),
                    "clip_fraction": float(model.logger.name_to_value.get('train/clip_fraction', 0.0)),
                    "learning_rate": float(model.logger.name_to_value.get('train/learning_rate', 0.0)),
                    "n_updates": int(model.logger.name_to_value.get('train/n_updates', 0))
                }
                
                print("\n  Network Stats:")
                print(f"    Value Network:")
                print(f"      Loss: {training_stats['value_loss']:.4f}")
                print(f"      Explained Var: {training_stats['explained_variance']:.2f}")
                print(f"    Policy Network:")
                print(f"      Loss: {training_stats['policy_loss']:.4f}")
                print(f"      Entropy: {training_stats['entropy_loss']:.4f}")
                print(f"      KL Div: {training_stats['approx_kl']:.4f}")
                print(f"    Training:")
                print(f"      Total Loss: {training_stats['total_loss']:.4f}")
                print(f"      Clip Fraction: {training_stats['clip_fraction']:.4f}")
                print(f"      Learning Rate: {training_stats['learning_rate']:.6f}")
                print(f"      Updates: {training_stats['n_updates']}")
            except Exception as e:
                print("\n  Network Stats: Not available")
        
        # Performance metrics
        print("\n  Performance Metrics:")
        print(f"    Total Trades: {metrics['total_trades']} ({metrics['win_rate']:.2f}% win)")
        print(f"    Average Win: {metrics['avg_win_points']:.1f} points ({metrics['win_hold_time']:.1f} bars)")
        print(f"    Average Loss: {metrics['avg_loss_points']:.1f} points ({metrics['loss_hold_time']:.1f} bars)")
        print(f"    Long Trades: {metrics['long_trades']} ({metrics['long_win_rate']:.1f}% win)")
        print(f"    Short Trades: {metrics['short_trades']} ({metrics['short_win_rate']:.1f}% win)")
        print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
