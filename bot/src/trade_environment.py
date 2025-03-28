import gymnasium
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces

class TradingEnv(gymnasium.Env):
    metadata = {"render_modes": None, "render_fps": None}
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lot_percentage: float = 0.01, bar_count: int = 50, 
                 random_start: bool = False):
        """Trading environment focused on balance growth."""
        super().__init__()
        
        # Trading constants
        self.POINT_VALUE = 0.01  # Each point = $0.01 for crypto
        self.PIP_VALUE = 0.0001  # Standard pip size
        self.MIN_LOTS = 0.01     # Minimum position size (0.01 BTC)
        self.MAX_LOTS = 100.0    # Maximum position size (100 BTC)
        self.CONTRACT_SIZE = 1.0  # 1 BTC per lot
        
        # Store raw prices first
        self.prices = {
            'close': data['close'].values,
            'high': data['high'].values,
            'low': data['low'].values,
            'spread': data['spread'].values,
            'atr': data['ATR'].values
        }
        
        # Preprocess the data to select and compute the most relevant features
        self.raw_data = self._preprocess_data(data)
        self.current_step = 0
        self.bar_count = bar_count
        self.random_start = random_start
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.lot_percentage = lot_percentage
        
        self.trades: List[Dict[str, Any]] = []
        self.open_positions: List[Dict[str, Any]] = []
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        
        self._setup_action_space()
        self._setup_observation_space(self.raw_data.shape[1])
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with advanced price action indicators and safe calculations."""
        df = pd.DataFrame(index=data.index)
        
        # Safe returns calculation
        df['returns'] = data['close'].pct_change().fillna(0).clip(-0.1, 0.1)
        
        # Safe volatility calculation
        returns_std = df['returns'].rolling(20, min_periods=1).std()
        df['volatility'] = returns_std.fillna(returns_std.mean()).clip(1e-8, 0.1)
        
        # Normalized price range
        high_low_range = (data['high'] - data['low']) / data['close']
        df['avg_range'] = high_low_range.rolling(10, min_periods=1).mean().clip(0, 0.1)
        
        # Technical indicators with safe normalization
        df['rsi'] = ((data['RSI'] - 50) / 25).clip(-2, 2)  # Normalized RSI
        df['trend'] = np.where(data['EMA_fast'] > data['EMA_slow'], 1, -1)
        
        # Safe trend strength calculation
        ema_diff = (data['EMA_fast'] - data['EMA_slow']).abs()
        df['trend_strength'] = (ema_diff / data['close']).clip(0, 0.1)
        
        # Safe momentum calculation
        returns_mean = df['returns'].rolling(10, min_periods=1).mean()
        df['momentum'] = (returns_mean / df['volatility']).clip(-3, 3)
        
        # Volume analysis with safe normalization
        volume_sma = data['volume'].rolling(20, min_periods=1).mean()
        volume_std = data['volume'].rolling(20, min_periods=1).std()
        volume_std = volume_std.replace(0, volume_std.mean())
        df['volume'] = ((data['volume'] - volume_sma) / volume_std).clip(-3, 3)
        
        # Relative spread
        # Convert spread from points to percentage (1 point = $0.01)
        df['spread'] = ((data['spread'] / 100.0) / data['close']).clip(0, 0.01)  # 1% max spread
        
        # ATR ratio with safe calculation
        atr_ma = data['ATR'].rolling(50, min_periods=1).mean()
        atr_ma = atr_ma.replace(0, atr_ma.mean())
        df['atr_ratio'] = (data['ATR'] / atr_ma).clip(0.1, 10)
        
        # Combined entry signal
        rsi_signal = -1 * df['rsi']  # Invert RSI for better signal alignment
        volume_signal = df['volume']
        trend_signal = df['trend'] * df['trend_strength']
        df['entry_signal'] = ((rsi_signal + volume_signal + trend_signal) / 3).clip(-1, 1)
        
        # Price deviation from trend
        df['price_vs_ema'] = ((data['close'] / data['EMA_slow'] - 1) * 100).clip(-2, 2)
        
        # Safe swing calculations
        high_max = data['high'].rolling(5, min_periods=1).max()
        low_min = data['low'].rolling(5, min_periods=1).min()
        df['swing_high'] = ((high_max - data['close']) / data['close']).clip(0, 0.1)
        df['swing_low'] = ((data['close'] - low_min) / data['close']).clip(0, 0.1)
        
        # Replace any remaining NaN values with 0
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _setup_action_space(self) -> None:
        """Configure combined continuous action space."""
        self.action_space = spaces.Box(
            low=np.array([-1, 0.5, 1.0]),     # Position [-1=Sell, 0=Hold, 1=Buy]
            high=np.array([1, 2.0, 4.0]),     # SL [0.5-2.0 ATR], TP [1.0-4.0 ATR]
            dtype=np.float32
        )
        
        # ATR multiplier ranges for intraday trading
        self.SL_MIN_ATR = 0.5  # Minimum stop loss distance in ATR
        self.SL_MAX_ATR = 2.0  # Maximum stop loss distance in ATR
        self.TP_MIN_ATR = 1.0  # Minimum take profit distance in ATR
        self.TP_MAX_ATR = 4.0  # Maximum take profit distance in ATR        
    
    def _setup_observation_space(self, feature_count: int) -> None:
        """Configure the observation space dimensions including trade state."""
        # Market features + position features for each potential position
        # Position features: [position_type, lot_size, unrealized_pnl, distance_to_sl, distance_to_tp]
        position_features = 5
        max_positions = 4  # Allow up to 4 concurrent positions
        
        obs_dim = (self.bar_count * feature_count) + (position_features * max_positions)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_dim,), dtype=np.float32
        )
        self.max_positions = max_positions
        
    def _process_action(self, action: np.ndarray) -> Tuple[int, float, float]:
        """Process the continuous action into position and SL/TP points."""
        
        if action[0] > 0.33:  # Upper third for buy
            position = 1
        elif action[0] < -0.33:  # Lower third for sell
            position = -1
        else:  # Middle third for hold
            position = 0
            
        # Scale SL/TP by ATR for adaptive position sizing
        current_atr = self.prices['atr'][self.current_step]
        
        # Apply ATR multipliers with proper clipping
        sl_multiplier = np.clip(action[1], self.SL_MIN_ATR, self.SL_MAX_ATR)
        tp_multiplier = np.clip(action[2], self.TP_MIN_ATR, self.TP_MAX_ATR)
        
        # Calculate price distances based on ATR
        sl_distance = sl_multiplier * current_atr
        tp_distance = tp_multiplier * current_atr
        
        return position, sl_distance, tp_distance
        
    def get_history(self) -> np.ndarray:
        """Return feature window and position state for observation."""
        # Get market features
        start = max(0, self.current_step - self.bar_count + 1)
        end = self.current_step + 1
        window = self.raw_data.iloc[start:end].values
        
        if window.shape[0] < self.bar_count:
            padding = np.zeros((self.bar_count - window.shape[0], window.shape[1]))
            window = np.vstack((padding, window))
        
        # Add position features
        position_features = []
        current_price = self.prices['close'][self.current_step]
        
        for pos in self.open_positions:
            # Calculate distances to SL/TP as percentage of current price
            sl_distance = abs(current_price - pos["sl_price"]) / current_price
            tp_distance = abs(current_price - pos["tp_price"]) / current_price
            
            # Calculate unrealized PnL in points and convert to percentage of balance
            if pos["position"] == 1:  # BUY
                points = (current_price - pos["entry_price"]) / self.POINT_VALUE
            else:  # SELL
                points = (pos["entry_price"] - current_price) / self.POINT_VALUE
            
            unrealized_pnl = ((points * pos["lot_size"] * self.POINT_VALUE) / self.balance) * 100
            
            position_features.extend([
                pos["position"],           # Position type (-1, 1)
                pos["lot_size"],          # Position size
                unrealized_pnl,           # Unrealized PnL
                sl_distance,              # Distance to SL
                tp_distance               # Distance to TP
            ])
        
        # Pad remaining position slots with zeros
        remaining_slots = self.max_positions - len(self.open_positions)
        position_features.extend([0.0] * (remaining_slots * 5))
        
        return np.concatenate([window.flatten(), position_features]).astype(np.float32)
    
    def _execute_trade(self, position: int, sl_points: float, tp_points: float, raw_spread: float) -> None:
        """Execute a trade using raw spread for entry price."""
        if position == 0:
            self.steps_since_trade += 1
            return
            
        if len(self.open_positions) >= self.max_positions:
            return
            
        long_positions = sum(1 for p in self.open_positions if p["position"] == 1)
        short_positions = sum(1 for p in self.open_positions if p["position"] == -1)
        
        if position == 1 and long_positions >= 1:
            return
        if position == -1 and short_positions >= 1:
            return
            
        current_price = self.prices['close'][self.current_step]

        if position == 1:  # BUY
            entry_price = current_price + raw_spread  # Buy at ask (price + spread)
            sl_price = entry_price - sl_points
            tp_price = entry_price + tp_points
        else:  # SELL
            entry_price = current_price - raw_spread  # Sell at bid (price - spread)
            sl_price = entry_price + sl_points
            tp_price = entry_price - tp_points
        
        # Calculate RRR based on price distances
        rrr = tp_price / sl_price if sl_price > 0 else 0
        
        # Calculate position size based on risk per point
        stop_loss_distance = abs(entry_price - sl_price)  # Price distance to SL
        value_per_lot = stop_loss_distance * self.CONTRACT_SIZE
        risk_amount = self.lot_percentage * self.balance
        lot_size = risk_amount / value_per_lot  # Risk amount divided by value per lot
        
        # Round to 2 decimals and limit between min and max lots
        lot_size = min(max(round(lot_size, 2), self.MIN_LOTS), self.MAX_LOTS)
        
        self.open_positions.append({
            "position": position,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lot_size": lot_size,
            "entry_step": self.current_step,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "rrr": rrr,
            "entry_spread": self.prices['spread'][self.current_step] * self.POINT_VALUE,
            "entry_atr": self.prices['atr'][self.current_step]
        })
        self.steps_since_trade = 0
            
    def _evaluate_positions(self, high_price: float, low_price: float, raw_spread: float,
                          prev_balance: float) -> Tuple[float, List[int]]:
        """Evaluate positions using current price and spread."""
        reward = 0
        closed_positions = []
        
        # Evaluate unrealized positions
        for pos in self.open_positions:
            unrealized_pnl = 0
            current_price = self.prices['close'][self.current_step]

            # Calculate PnL in points and convert to dollars
            if pos["position"] == 1:  # BUY
                points = (current_price - pos["entry_price"]) / self.POINT_VALUE
            else:  # SELL
                points = (pos["entry_price"] - current_price) / self.POINT_VALUE
            
            unrealized_pnl = points * pos["lot_size"] * self.POINT_VALUE
            
            # Small RRR bonus for positive unrealized PnL
            if unrealized_pnl > 0:
                rrr_scale = min(pos["rrr"] / 1.5, 2.0)
                reward += (unrealized_pnl / prev_balance) * 0.05 * rrr_scale  # Reduced weight for unrealized PnL
            else:
                reward += (unrealized_pnl / prev_balance) * 0.05  # Reduced weight for unrealized PnL
        
        # Check for closed positions
        for i, pos in enumerate(self.open_positions):
            if pos["position"] == 1:  # BUY
                # Add spread to price when checking TP (need higher price to actually hit TP)
                # Subtract spread from price when checking SL (will hit SL at higher price)
                hit_tp = (high_price + raw_spread) >= pos["tp_price"]
                hit_sl = (low_price - raw_spread) <= pos["sl_price"]
            else:  # SELL
                # Subtract spread from price when checking TP (need lower price to hit TP)
                # Add spread to price when checking SL (will hit SL at lower price)
                hit_tp = (low_price - raw_spread) <= pos["tp_price"]
                hit_sl = (high_price + raw_spread) >= pos["sl_price"]

            if hit_tp or hit_sl:
                exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
                # Calculate PnL in points and convert to dollars
                if pos["position"] == 1:  # BUY
                    points = (exit_price - pos["entry_price"]) / self.POINT_VALUE
                else:  # SELL
                    points = (pos["entry_price"] - exit_price) / self.POINT_VALUE
                    
                pnl = points * pos["lot_size"] * self.POINT_VALUE
                
                # Calculate actual RRR based on points
                pos["actual_rrr"] = pos["tp_points"] / pos["sl_points"] if pos["sl_points"] > 0 else 0
                
                reward += (pnl / prev_balance)
                self.balance += pnl
                
                if pnl > 0:
                    rrr_bonus = min(pos["rrr"] / 1.5, 2.0)
                    reward += rrr_bonus
                    self.win_count += 1
                else:
                    if pos["rrr"] < 1.0:
                        reward -= (1.0 - pos["rrr"]) * 0.5
                    self.loss_count += 1
                    
                closed_positions.append(i)
                pos["pnl"] = pnl
                pos["exit_price"] = exit_price
                pos["exit_step"] = self.current_step
                pos["hit_tp"] = hit_tp
                self.trades.append(pos)

        return reward, closed_positions
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take one environment step."""
        position, sl_points, tp_points = self._process_action(action)

        # Get prices and convert spread to dollars once
        high_price = self.prices['high'][self.current_step]
        low_price = self.prices['low'][self.current_step]
        current_spread = self.prices['spread'][self.current_step] * self.POINT_VALUE

        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        trade_reward = 0
        growth_reward = 0
        drawdown_penalty = 0

        position_reward, closed_positions = self._evaluate_positions(
            high_price, low_price, current_spread, previous_balance
        )
        trade_reward += position_reward
        
        for i in sorted(closed_positions, reverse=True):
            self.open_positions.pop(i)
            
        if position != 0:
            self._execute_trade(position, sl_points, tp_points, current_spread)

        if previous_balance > 0:
            growth_ratio = self.balance / self.initial_balance
            if growth_ratio > 1:
                growth_reward += np.log(growth_ratio) * 10.0
            
            balance_change_ratio = self.balance / previous_balance
            if hasattr(self, 'last_balance_ratio'):
                growth_acceleration = balance_change_ratio - self.last_balance_ratio
                if growth_acceleration > 0:
                    growth_reward += growth_acceleration * 5.0
            self.last_balance_ratio = balance_change_ratio
            
            if self.balance < self.max_balance:
                drawdown = (self.max_balance - self.balance) / self.max_balance
                drawdown_penalty = drawdown * 5.0

        reward = (trade_reward * 0.3) + (growth_reward * 0.6) - (drawdown_penalty * 0.1)
        
        # Check for max drawdown or bankruptcy
        max_drawdown = (self.max_balance - self.balance) / self.max_balance
        if self.balance <= 0 or max_drawdown >= 0.8:  # 80% max drawdown
            self.open_positions.clear()
            reward = -20
            done = True

        if done:
            final_return = (self.balance / self.initial_balance) - 1
            reward += final_return * 20.0
            
            if self.balance > self.initial_balance:
                reward += (np.log(self.balance / self.initial_balance) * 10.0)
        
        obs = self.get_history()
        self.reward = reward

        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        if self.random_start:
            self.current_step = np.random.randint(0, len(self.raw_data) - self.bar_count)
        else:
            self.current_step = 0
            
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.open_positions.clear()
        self.trades.clear()
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        
        return self.get_history(), {}

    def render(self) -> None:
        """Print current environment state."""
        print(f"\n===== Environment State at Step {self.current_step} =====")
        print(f"Open Positions: {len(self.open_positions)}")
        for pos in self.open_positions:
            print(f"  Position: {pos['position']} Entry: {pos['entry_price']:.2f} "
                  f"SL: {pos['sl_price']:.2f}, TP: {pos['tp_price']:.2f} "
                  f"RRR: {pos['rrr']:.2f}")
        
        if len(self.trades) == 0:
            print("\nNo completed trades yet.")
            return
            
        trades_df = pd.DataFrame(self.trades)
        num_tp = sum(1 for trade in self.trades if trade["pnl"] > 0.0)
        num_sl = sum(1 for trade in self.trades if trade["pnl"] < 0.0)
        total_trades = len(self.trades)
        
        avg_pnl_tp = trades_df[trades_df["pnl"] > 0.0]["pnl"].mean() if num_tp > 0 else 0.0
        avg_pnl_sl = abs(trades_df[trades_df["pnl"] < 0.0]["pnl"].mean()) if num_sl > 0 else 0.0
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        expected_value = trades_df["pnl"].mean() if total_trades > 0 else 0.0
        
        # Calculate trade statistics
        avg_rrr = trades_df["actual_rrr"].mean() if total_trades > 0 else 0.0
        avg_holding_length = (trades_df["exit_step"] - trades_df["entry_step"]).mean() if total_trades > 0 else 0.0
        
        num_buy = trades_df[trades_df["position"] == 1].shape[0]
        num_sell = trades_df[trades_df["position"] == -1].shape[0]
        
        buy_win_rate = (trades_df[(trades_df["position"] == 1) & (trades_df["pnl"] > 0.0)].shape[0] / num_buy * 100) if num_buy > 0 else 0.0
        sell_win_rate = (trades_df[(trades_df["position"] == -1) & (trades_df["pnl"] > 0.0)].shape[0] / num_sell * 100) if num_sell > 0 else 0.0
        total_win_rate = (num_tp / total_trades * 100) if total_trades > 0 else 0.0

        def kelly_criterion(win_rate, win_loss_ratio):
            if win_loss_ratio == 0:
                return 0.0
            return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
            
        def sharpe_ratio(returns, risk_free_rate=0.00, trading_periods=252):
            if len(returns) < 2:
                return 0.0
            excess_returns = np.array(returns) - (risk_free_rate / trading_periods)
            return (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(trading_periods)

        kelly_criteria = kelly_criterion(total_win_rate / 100.0, avg_rrr)
        sharpe = sharpe_ratio(trades_df["pnl"] / self.initial_balance) if total_trades > 0 else 0.0

        metrics_text = (
            f"\n===== Trading Performance Metrics =====\n"
            f"Current Balance: {self.balance:.2f}\n"
            f"Total Return: {total_return:.2f}%\n"
            f"Total Trades: {total_trades}\n"
            f"Total Win Rate: {total_win_rate:.2f}%\n"
            f"Long Trades: {num_buy}\n"
            f"Long Win Rate: {buy_win_rate:.2f}%\n"
            f"Short Trades: {num_sell}\n"
            f"Short Win Rate: {sell_win_rate:.2f}%\n"
            f"Average Win: {avg_pnl_tp:.2f}\n"
            f"Average Loss: {avg_pnl_sl:.2f}\n"
            f"Average RRR: {avg_rrr:.2f}\n"
            f"Average Length: {avg_holding_length:.1f} bars\n"
            f"Expected Value: {expected_value:.2f}\n"
            f"Kelly Criterion: {kelly_criteria:.2f}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )
        print(metrics_text)
        
        # Display first and last 3 trades
        if len(trades_df) > 0:
            print("\n===== First 3 Trades =====")
            print(trades_df.head(5)[["position", "entry_price", "exit_price", "pnl", "lot_size", "entry_spread", "entry_atr", "actual_rrr", "hit_tp"]].to_string())
            print("\n===== Last 3 Trades =====")
            print(trades_df.tail(5)[["position", "entry_price", "exit_price", "pnl", "lot_size", "entry_spread", "entry_atr", "actual_rrr", "hit_tp"]].to_string())

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        np.random.seed(seed)
