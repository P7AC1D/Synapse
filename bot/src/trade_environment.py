import gymnasium as gym
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Dict, List, Tuple, Any, Optional, Union
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

# Action type constants
class ActionType(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2

def calculate_lot_size(balance: float, risk: float, entry_price: float, stop_loss: float, contract_size: float = 1.0) -> float:
    """
    Calculate position size based on account risk management.
    
    Args:
        balance: Current account balance
        risk: Risk percentage (0-1)
        entry_price: Trade entry price
        stop_loss: Stop loss price level
        contract_size: Size of one contract
        
    Returns:
        Lot size respecting risk parameters
    """
    risk_amount = balance * risk
    price_risk = abs(entry_price - stop_loss) * contract_size
    lot_size = risk_amount / price_risk if price_risk > 0 else 0.01
    return min(max(round(lot_size, 2), 0.01), 100.0)

def calculate_balance_change(lot_size: float, entry_price: float, exit_price: float, 
                            position: int, spread: float, contract_size: float = 1.0) -> float:
    """
    Calculate profit/loss for a trade including spread costs.
    
    Args:
        lot_size: Size of the position
        entry_price: Entry price
        exit_price: Exit price
        position: Position direction (1=long, -1=short)
        spread: Market spread
        contract_size: Size of one contract
        
    Returns:
        PnL amount
    """
    if position == 1:  # Buy
        pnl = ((exit_price - entry_price) - spread) * lot_size * contract_size
    elif position == -1:  # Sell
        pnl = ((entry_price - exit_price) - spread) * lot_size * contract_size
    else:
        pnl = 0.0

    assert not np.isnan(pnl), "PnL calculation resulted in NaN"
    return pnl

class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, 
                 lot_percentage: float = 0.01, bar_count: int = 50, 
                 normalization_window: int = 100, random_start: bool = False):
        """
        Trading environment for reinforcement learning.
        
        Args:
            data: DataFrame with OHLC, spread, and technical indicators
            initial_balance: Starting account balance
            lot_percentage: Risk per trade (0-1)
            bar_count: Number of bars in observation window
            normalization_window: Window for normalizing price data
            random_start: Whether to start from a random position in data
        """
        super(TradingEnv, self).__init__()
        
        # Data and position tracking
        self.raw_data = data.copy()
        self.current_step = 0
        self.bar_count = bar_count
        self.normalization_window = normalization_window
        self.random_start = random_start
        
        # Account metrics
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.lot_percentage = lot_percentage
        
        # Trading metrics
        self.trades: List[Dict[str, Any]] = []
        self.open_positions: List[Dict[str, Any]] = []
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        self.reward = 0
        
        # Risk-reward ratio options
        self.rrr = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
        
        # Define action and observation spaces
        self._setup_action_space()
        self._setup_observation_space(data.shape[1])
    
    def _setup_action_space(self) -> None:
        """Configure the action space for the environment."""
        # Actions: 0=Hold, 1-5=Buy with RRR[0-4], 6-10=Sell with RRR[0-4]
        self.action_space = spaces.Discrete(1 + 2 * len(self.rrr))
    
    def _setup_observation_space(self, feature_count: int) -> None:
        """Configure the observation space dimensions."""
        obs_dim = self.bar_count * feature_count
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )

    def normalize_window(self, window: np.ndarray) -> np.ndarray:
        """
        Normalize data using only past information available at current step.
        
        Args:
            window: Window data to normalize
            
        Returns:
            Normalized window data
        """
        # Use only historical data for normalization
        lookback_start = max(0, self.current_step - self.normalization_window)
        lookback_end = self.current_step + 1  # Include current step
        
        # Extract historical window for computing normalization parameters
        history_window = self.raw_data.iloc[lookback_start:lookback_end].values
        
        # Fit scaler on historical data only
        scaler = StandardScaler()
        scaler.fit(history_window)
        
        # Transform the window we want to return
        return scaler.transform(window)

    def get_history(self) -> np.ndarray:
        """
        Return normalized data window for the agent's observation.
        
        Returns:
            Flattened array of normalized price data
        """
        start = max(0, self.current_step - self.bar_count + 1)
        end = self.current_step + 1
        
        # Get raw window data
        window = self.raw_data.iloc[start:end].values
        
        # Apply rolling normalization
        normalized_window = self.normalize_window(window)
        
        # Handle padding if needed
        if normalized_window.shape[0] < self.bar_count:
            padding = np.zeros((self.bar_count - normalized_window.shape[0], normalized_window.shape[1]))
            normalized_window = np.vstack((padding, normalized_window))
            
        return normalized_window.flatten()

    def _decode_action(self, action: int) -> Tuple[int, int, float]:
        """
        Convert discrete action to trade parameters.
        
        Args:
            action: Action from the agent
            
        Returns:
            Tuple of (action_type, rrr_index, risk_reward_ratio)
        """
        if action == 0:
            action_type = ActionType.HOLD
            rrr_idx = 0
        elif 1 <= action <= len(self.rrr):
            action_type = ActionType.BUY
            rrr_idx = action - 1
        else:
            action_type = ActionType.SELL
            rrr_idx = action - 1 - len(self.rrr)
            
        return action_type, rrr_idx, self.rrr[rrr_idx]
    
    def _execute_new_trade(self, action_type: int, rrr: float, 
                          current_price: float, spread: float, 
                          atr: float) -> None:
        """
        Execute a new trade based on the agent's action.
        
        Args:
            action_type: Type of action (HOLD, BUY, SELL)
            rrr: Risk-reward ratio 
            current_price: Current market price
            spread: Market spread
            atr: Average True Range
        """
        if action_type == ActionType.HOLD:
            # Penalize inactivity
            self.steps_since_trade += 1
            return
            
        # Calculate position parameters    
        position = 1 if action_type == ActionType.BUY else -1
        sl_value = atr
        tp_value = atr * rrr
        entry_price = current_price
        
        # Set stop loss and take profit levels
        sl_price = entry_price - sl_value if position == 1 else entry_price + sl_value
        tp_price = entry_price + tp_value if position == 1 else entry_price - tp_value
        
        # Calculate position size
        lot_size = calculate_lot_size(
            self.balance, self.lot_percentage, entry_price, sl_price
        )
        
        # Open the position
        self.open_positions.append({
            "position": position,
            "entry_price": entry_price,
            "spread": spread,
            "atr": atr,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "lot_size": lot_size,
            "entry_step": self.current_step
        })
        self.steps_since_trade = 0
            
    def _evaluate_positions(self, high_price: float, low_price: float, 
                           current_price: float, spread: float, 
                           prev_balance: float) -> Tuple[float, List[int]]:
        """
        Evaluate open positions and handle exits.
        
        Args:
            high_price: Current candle's high price
            low_price: Current candle's low price
            current_price: Current close price
            spread: Market spread
            prev_balance: Balance before updates
            
        Returns:
            Tuple of (reward_from_positions, indices_of_closed_positions)
        """
        reward = 0
        closed_positions = []
        
        # First calculate unrealized PnL for open positions
        for pos in self.open_positions:
            unrealized_pnl = calculate_balance_change(
                pos["lot_size"], pos["entry_price"], 
                current_price, pos["position"], spread
            )
            reward += (unrealized_pnl / prev_balance) * 0.5
            
        # Then check for stop loss and take profit hits
        for i, pos in enumerate(self.open_positions):
            hit_tp = (high_price >= pos["tp_price"] if pos["position"] == 1 
                     else low_price <= pos["tp_price"])
            hit_sl = (low_price <= pos["sl_price"] if pos["position"] == 1 
                     else high_price >= pos["sl_price"])

            if hit_tp or hit_sl:
                exit_price = pos["tp_price"] if hit_tp else pos["sl_price"]
                pnl = calculate_balance_change(
                    pos["lot_size"], pos["entry_price"], 
                    exit_price, pos["position"], spread
                )
                reward += (pnl / prev_balance)
                self.balance += pnl
                
                # Update win/loss statistics
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                    
                # Mark position for closure
                closed_positions.append(i)
                pos["pnl"] = pnl
                pos["exit_price"] = exit_price
                pos["exit_step"] = self.current_step
                self.trades.append(pos)

        return reward, closed_positions
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one environment step.
        
        Args:
            action: Action selected by the agent
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action
        action_type, rrr_idx, rrr = self._decode_action(action)

        # Get current market data
        current_price = self.raw_data.iloc[self.current_step]["close"]
        high_price = self.raw_data.iloc[self.current_step]["high"]
        low_price = self.raw_data.iloc[self.current_step]["low"]
        spread = self.raw_data.iloc[self.current_step]["spread"] / 100.0
        atr = self.raw_data.iloc[self.current_step]["ATR"]

        # Track balance before update
        previous_balance = self.balance
        self.max_balance = max(self.balance, self.max_balance)
        
        # Progress time
        self.current_step += 1
        done = self.current_step >= len(self.raw_data) - 1

        # Initialize reward components
        trade_reward = 0
        risk_penalty = 0
        position_management_reward = 0
        balance_reward = 0

        # Evaluate existing positions with adjusted rewards
        position_reward, closed_positions = self._evaluate_positions(
            high_price, low_price, current_price, spread, previous_balance
        )
        trade_reward += position_reward
        
        # Remove closed positions
        for i in sorted(closed_positions, reverse=True):
            self.open_positions.pop(i)
            
        # Execute new trade if requested with minimal immediate reward
        if action_type != ActionType.HOLD:
            # Small incentive just to prevent complete inaction
            trade_reward += 0.05
            self._execute_new_trade(action_type, rrr, current_price, spread, atr)
        
        # Less punitive inactivity penalty - just enough to prevent freezing
        if action_type == ActionType.HOLD:
            inactivity_penalty = min(0.1 * np.log1p(self.steps_since_trade), 0.5)
            risk_penalty -= inactivity_penalty
        else:
            # Only penalize excessive positions (more than 5)
            open_positions_penalty = 0.05 * max(0, len(self.open_positions) - 5) if len(self.open_positions) > 5 else 0
            risk_penalty -= open_positions_penalty
        
        # Balance-focused rewards (the most important part)
        if previous_balance > 0:
            # Immediate balance change (weighted heavily)
            balance_change_ratio = self.balance / previous_balance
            balance_reward += np.log(max(balance_change_ratio, 0.1)) * 3.0  # Increased weight
            
            # Relative to initial balance (weighted heavily)
            total_return_ratio = self.balance / self.initial_balance
            balance_reward += np.log(max(total_return_ratio, 0.1)) * 1.0
            
            # Reward acceleration of balance growth
            if hasattr(self, 'last_balance_ratio'):
                growth_acceleration = balance_change_ratio - self.last_balance_ratio
                balance_reward += growth_acceleration * 2.0
            self.last_balance_ratio = balance_change_ratio

        # Trade quality metrics (reduced weight)
        if len(self.trades) > 0:
            win_rate = self.win_count / max(1, (self.win_count + self.loss_count))
            
            # Small reward for positive expectancy
            if len(self.trades) >= 5:
                avg_win = np.mean([t["pnl"] for t in self.trades if t["pnl"] > 0]) if self.win_count > 0 else 0
                avg_loss = abs(np.mean([t["pnl"] for t in self.trades if t["pnl"] < 0])) if self.loss_count > 0 else 1
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                position_management_reward += np.clip(expectancy / 100, -0.3, 0.6)
        
        # Combine reward components with heavy emphasis on balance
        reward = (trade_reward * 0.5) + risk_penalty + (position_management_reward * 0.3) + (balance_reward * 2.0)
        
        # Bankruptcy is still a major failure
        if self.balance <= 0:
            self.open_positions.clear()
            reward = -10
            done = True

        # End-of-episode final return reward (heavily weighted)
        if done:
            final_return = (self.balance / self.initial_balance) - 1
            # Much stronger weight on final performance
            reward += final_return * 5.0
            
            # Additional balance thresholds for extra rewards
            if self.balance > self.initial_balance * 1.5:  # 50% profit
                reward += 3.0
            if self.balance > self.initial_balance * 2.0:  # 100% profit
                reward += 5.0
        
        # Build observation
        obs = self.get_history()
        self.reward = reward

        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Initial observation and empty info dict
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self.random_start:
            self.current_step = np.random.randint(0, len(self.raw_data) - self.bar_count)
        else:
            self.current_step = 0
            
        # Reset account and trade tracking
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.open_positions.clear()
        self.trades.clear()
        self.steps_since_trade = 0
        self.win_count = 0
        self.loss_count = 0
        
        return self.get_history(), {}

    def render(self, mode: str = "human", close: bool = False) -> None:
        """
        Render environment state with detailed metrics.
        
        Args:
            mode: Rendering mode
            close: Whether to close rendering
        """
        if mode == "human":
            # Basic environment info
            print(f"\n===== Environment State at Step {self.current_step} =====")
            print(f"Open Positions: {len(self.open_positions)}")
            for pos in self.open_positions:
                print(f"  Position: {pos['position']} Entry: {pos['entry_price']:.2f} "
                    f"SL: {pos['sl_price']:.2f}, TP: {pos['tp_price']:.2f}")
            
            # Calculate detailed metrics
            total_trades = len(self.trades)
            if total_trades == 0:
                print("\nNo completed trades yet.")
                return
                
            # Calculate key metrics
            num_tp = sum(1 for trade in self.trades if trade["pnl"] > 0.0)
            num_sl = sum(1 for trade in self.trades if trade["pnl"] < 0.0)
            
            trades_df = pd.DataFrame(self.trades)
            avg_pnl_tp = trades_df[trades_df["pnl"] > 0.0]["pnl"].mean() if num_tp > 0 else 0.0
            avg_pnl_sl = trades_df[trades_df["pnl"] < 0.0]["pnl"].mean() if num_sl > 0 else 0.0
            
            total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
            expected_value = trades_df["pnl"].mean() if total_trades > 0 else 0.0
            
            avg_pnl_sl = abs(avg_pnl_sl) if num_sl > 0 else 0.0
            rrr = avg_pnl_tp / avg_pnl_sl if avg_pnl_sl > 0 else 0.0
            
            num_buy = trades_df[trades_df["position"] == 1].shape[0] if "position" in trades_df else 0
            num_sell = trades_df[trades_df["position"] == -1].shape[0] if "position" in trades_df else 0
            
            buy_win_rate = (trades_df[(trades_df["position"] == 1) & (trades_df["pnl"] > 0.0)].shape[0] / num_buy * 100) if num_buy > 0 else 0.0
            sell_win_rate = (trades_df[(trades_df["position"] == -1) & (trades_df["pnl"] > 0.0)].shape[0] / num_sell * 100) if num_sell > 0 else 0.0
            total_win_rate = (num_tp / total_trades * 100) if total_trades > 0 else 0.0

            # Kelly Criterion
            def kelly_criterion(win_rate, win_loss_ratio):
                if win_loss_ratio == 0:
                    return 0.0  # Avoid division by zero
                return round(win_rate - ((1 - win_rate) / win_loss_ratio), 4)
                
            # Sharpe Ratio
            def sharpe_ratio(returns, risk_free_rate=0.00, trading_periods=252):
                if len(returns) < 2:
                    return 0.0  # Avoid calculation on insufficient data
                excess_returns = np.array(returns) - (risk_free_rate / trading_periods)
                sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
                return sharpe * np.sqrt(trading_periods)

            kelly_criteria = kelly_criterion(total_win_rate / 100.0, rrr) if not np.isnan(rrr) else 0.0

            if total_trades > 0:
                daily_returns = trades_df["pnl"] / self.initial_balance
                sharpe = sharpe_ratio(daily_returns)
            else:
                sharpe = 0.0

            # Print metrics
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
                f"Average RRR: {rrr:.2f}\n"
                f"Expected Value: {expected_value:.2f}\n"
                f"Kelly Criterion: {kelly_criteria:.2f}\n"
                f"Sharpe Ratio: {sharpe:.2f}\n"
            )
            print(metrics_text)

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        np.random.seed(seed)