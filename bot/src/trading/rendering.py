"""Visualization and statistics display for trading environment."""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class Renderer:
    """Handles visualization and statistics display."""
    
    def __init__(self):
        """Initialize renderer."""
        self.episode_count = 0

    def render_episode_stats(self, env: Any) -> None:
        """Render trading statistics for current episode.
        
        Args:
            env: Trading environment instance
        """
        print(f"\n===== Episode {env.completed_episodes}, Step {env.episode_steps} =====")
        print(f"Current Balance: {env.balance:.2f}")
        print(f"Current Position: {'None' if not env.current_position else ('Long' if env.current_position['direction'] == 1 else 'Short')}")
        
        self._render_position_details(env)
        self._render_trade_statistics(env)

    def _render_position_details(self, env: Any) -> None:
        """Render current position details.
        
        Args:
            env: Trading environment instance
        """
        if not env.current_position:
            return
            
        current_spread = env.prices['spread'][env.current_step] * env.POINT_VALUE
        current_price = env.prices['close'][env.current_step]
        
        if env.current_position["direction"] == 1:  # Long
            current_exit_price = current_price - current_spread
            unrealized_pnl = (current_exit_price - env.current_position["entry_price"]) * \
                            env.current_position["lot_size"] * env.CONTRACT_SIZE
        else:  # Short
            current_exit_price = current_price + current_spread
            unrealized_pnl = (env.current_position["entry_price"] - current_exit_price) * \
                            env.current_position["lot_size"] * env.CONTRACT_SIZE
                
        print(f"\nPosition Details:")
        print(f"  Entry Price: {env.current_position['entry_price']:.5f}")
        print(f"  Current Price: {current_price:.5f}")
        print(f"  Current Spread: {current_spread:.5f}")
        print(f"  Potential Exit Price: {current_exit_price:.5f}")
        print(f"  Lot Size: {env.current_position['lot_size']:.2f}")
        print(f"  Unrealized P/L: {unrealized_pnl:.2f}")
        print(f"  Hold Time: {env.current_step - env.current_position['entry_step']} bars")

    def _render_trade_statistics(self, env: Any) -> None:
        """Render comprehensive trade statistics.
        
        Args:
            env: Trading environment instance
        """
        if len(env.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(env.trades)
        
        # Basic trade analysis
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] < 0]
        
        # Directional analysis
        long_trades = trades_df[trades_df["direction"] == 1]
        short_trades = trades_df[trades_df["direction"] == -1]
        long_wins = long_trades[long_trades["pnl"] > 0]
        short_wins = short_trades[short_trades["pnl"] > 0]
        
        # Hold time analysis
        avg_hold_time = trades_df["hold_time"].mean() if "hold_time" in trades_df.columns else 0
        avg_win_hold = winning_trades["hold_time"].mean() if "hold_time" in winning_trades.columns else 0
        avg_loss_hold = losing_trades["hold_time"].mean() if "hold_time" in losing_trades.columns else 0
        
        # Print performance metrics
        print("\n===== Performance Metrics =====")
        print(f"Total Return: {((env.balance - env.initial_balance) / env.initial_balance * 100):.2f}%")
        print(f"Total Trades: {len(env.trades)}")
        print(f"Overall Win Rate: {(len(winning_trades) / len(env.trades) * 100):.2f}%")
        print(f"Average Win: {winning_trades['pnl'].mean():.2f}")
        print(f"Average Loss: {losing_trades['pnl'].mean():.2f}")
        
        # Calculate and print profit factor
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) \
                       if losing_trades['pnl'].sum() != 0 else float('inf')
        print(f"Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "Profit Factor: ∞")
        
        print(f"Current Drawdown: {((env.max_balance - env.balance) / env.max_balance * 100):.2f}%")
        
        # Print hold time analysis
        print("\n===== Hold Time Analysis =====")
        print(f"Average Hold Time: {avg_hold_time:.1f} bars")
        print(f"Winners Hold Time: {avg_win_hold:.1f} bars")
        print(f"Losers Hold Time: {avg_loss_hold:.1f} bars")
        
        # Print directional performance
        total_trades = len(trades_df)
        long_pct = (len(long_trades) / total_trades * 100) if total_trades > 0 else 0.0
        short_pct = (len(short_trades) / total_trades * 100) if total_trades > 0 else 0.0
        
        print("\n===== Directional Performance =====")
        print(f"Long Trades: {len(long_trades)} ({long_pct:.1f}%)")
        if len(long_trades) > 0:
            print(f"Long Win Rate: {(len(long_wins) / len(long_trades) * 100):.1f}% "
                  f"(Avg PnL: {long_trades['pnl'].mean():.2f})")
        else:
            print("Long Win Rate: N/A")
            
        print(f"Short Trades: {len(short_trades)} ({short_pct:.1f}%)")
        if len(short_trades) > 0:
            print(f"Short Win Rate: {(len(short_wins) / len(short_trades) * 100):.1f}% "
                  f"(Avg PnL: {short_trades['pnl'].mean():.2f})")
        else:
            print("Short Win Rate: N/A")
