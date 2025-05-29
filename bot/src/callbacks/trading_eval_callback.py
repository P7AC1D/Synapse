"""Enhanced evaluation callback that logs comprehensive trading metrics."""
import os
import numpy as np
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

class TradingEvalCallback(EvalCallback):
    """Enhanced evaluation callback that logs comprehensive trading metrics."""
    
    def __init__(
        self,
        eval_env: Union[VecEnv],
        callback_on_new_best: Optional[Any] = None,
        callback_after_eval: Optional[Any] = None,
        n_eval_episodes: int = 1,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        """Initialize the trading evaluation callback.
        
        Args:
            eval_env: Environment used for evaluation
            callback_on_new_best: Callback called when new best model is found
            callback_after_eval: Callback called after each evaluation
            n_eval_episodes: Number of episodes to evaluate
            eval_freq: Frequency of evaluation
            log_path: Path to save logs
            best_model_save_path: Path to save best model
            deterministic: Whether to use deterministic policy
            render: Whether to render evaluation
            verbose: Verbosity level
            warn: Whether to show warnings
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        
        # Store callback_after_eval as instance attribute for access in _on_step
        self.callback_after_eval = callback_after_eval
        
        # Track trading metrics history
        self.trading_metrics_history = []
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset logs
            self._is_success_buffer = []
            episode_rewards = []
            episode_lengths = []
            trading_metrics_list = []

            for eval_episode in range(self.n_eval_episodes):
                # Reset environment and get initial obs
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    action, _states = self.model.predict(
                        obs, deterministic=self.deterministic
                    )
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                    episode_length += 1

                    if done:
                        # Extract trading metrics from environment
                        env = self.eval_env.envs[0] if hasattr(self.eval_env, 'envs') else self.eval_env
                        if hasattr(env, 'metrics'):
                            trading_summary = env.metrics.get_performance_summary()
                            trading_metrics_list.append(trading_summary)
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            # Calculate standard metrics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)

            # Calculate average trading metrics across episodes
            if trading_metrics_list:
                avg_trading_metrics = self._average_trading_metrics(trading_metrics_list)
                self.trading_metrics_history.append({
                    'timestep': self.num_timesteps,
                    **avg_trading_metrics
                })
                
                # Log comprehensive trading metrics
                self._log_trading_metrics(avg_trading_metrics, mean_reward, mean_ep_length)
            
            # Update best model if necessary
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("🏆 New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if we have a new best model
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Log to tensorboard/console
            if self.log_path is not None:
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
                
                # Log key trading metrics to tensorboard
                if trading_metrics_list:
                    avg_metrics = avg_trading_metrics
                    self.logger.record("eval/win_rate", avg_metrics.get('win_rate', 0.0))
                    self.logger.record("eval/total_pnl", avg_metrics.get('total_pnl', 0.0))
                    self.logger.record("eval/return_pct", avg_metrics.get('return_pct', 0.0))
                    self.logger.record("eval/max_drawdown_pct", avg_metrics.get('max_drawdown_pct', 0.0))
                    self.logger.record("eval/profit_factor", avg_metrics.get('profit_factor', 0.0))
                    self.logger.record("eval/total_trades", avg_metrics.get('total_trades', 0))

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {np.std(episode_lengths):.2f}")

            # Trigger callback after eval
            if self.callback_after_eval is not None:
                continue_training = continue_training and self.callback_after_eval.on_step()

        return continue_training
    
    def _average_trading_metrics(self, metrics_list):
        """Average trading metrics across multiple episodes."""
        if not metrics_list:
            return {}
            
        # Get all keys from first metrics dict
        keys = metrics_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                averaged[key] = np.mean(values)
            else:
                averaged[key] = 0.0
                
        return averaged
    
    def _log_trading_metrics(self, metrics, mean_reward, mean_ep_length):
        """Log comprehensive trading metrics to console."""
        print("=" * 60)
        print("📊 TRADING PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Core performance
        print(f"🎯 Episode Reward: {mean_reward:.2f}")
        print(f"📈 Total PnL: ${metrics.get('total_pnl', 0):.2f}")
        print(f"📊 Return: {metrics.get('return_pct', 0):.2f}%")
        print(f"🎲 Total Trades: {int(metrics.get('total_trades', 0))}")
        
        if metrics.get('total_trades', 0) > 0:
            print(f"🏆 Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"💰 Avg Win: ${metrics.get('avg_win', 0):.2f} ({metrics.get('avg_win_points', 0):.1f} pts)")
            print(f"💸 Avg Loss: ${metrics.get('avg_loss', 0):.2f} ({metrics.get('avg_loss_points', 0):.1f} pts)")
            print(f"⚡ Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            # Risk metrics
            print(f"⚠️ Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"📉 Current DD: {metrics.get('current_drawdown_pct', 0):.2f}%")
            
            # Directional analysis
            long_trades = int(metrics.get('long_trades', 0))
            short_trades = int(metrics.get('short_trades', 0))
            if long_trades > 0 or short_trades > 0:
                print(f"📈 Long: {long_trades} trades ({metrics.get('long_win_rate', 0):.1f}% win)")
                print(f"📉 Short: {short_trades} trades ({metrics.get('short_win_rate', 0):.1f}% win)")
            
            # Hold time analysis
            avg_hold = metrics.get('avg_hold_time', 0)
            if avg_hold > 0:
                print(f"⏱️ Avg Hold Time: {avg_hold:.1f} bars")
                
            # Streak analysis
            max_wins = int(metrics.get('max_consecutive_wins', 0))
            max_losses = int(metrics.get('max_consecutive_losses', 0))
            if max_wins > 0 or max_losses > 0:
                print(f"🔥 Max Win Streak: {max_wins}")
                print(f"❄️ Max Loss Streak: {max_losses}")
        else:
            print("⚠️ No trades executed during evaluation")
            
        print("=" * 60)
    
    def get_trading_metrics_history(self):
        """Get the history of trading metrics."""
        return self.trading_metrics_history
