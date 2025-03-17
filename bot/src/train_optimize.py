import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler

RATES_CSV_PATH = "../data/rates.csv"

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

df = pd.read_csv(RATES_CSV_PATH)
df.set_index('time', inplace=True)
#df.drop(columns=['EMA_fast', 'EMA_medium', 'EMA_slow', 'MACD', 'RSI', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV', 'VWAP'], inplace=True)

start_datetime = df.index[0]
end_datetime = df.index[-1]
print(f"Data collected from {start_datetime} to {end_datetime}")

df_scaled = df.copy()

scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
df_scaled['unscaled_close'] = df['close']

print(df_scaled.tail())

# Split train/test (80/20)
split_idx = int(len(df) * 0.8)
train_data = df_scaled.iloc[:split_idx]
test_data = df_scaled.iloc[split_idx:]

import optuna
from trade_environment import BitcoinTradingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

optuna.logging.set_verbosity(optuna.logging.INFO)

seed_value = np.random.randint(0, 100000)
seed_value = 65782
print(f"Using seed: {seed_value}")

def logging_callback(study, trial):
    print(f"Trial {trial.number} finished with value: {trial.value}", flush=True)

def optimize_ppo(trial):
  env = Monitor(BitcoinTradingEnv(train_data))
  env.action_space.seed(seed_value)
  vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed_value)

  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
  ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05, log=True)
  clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
  n_epochs = trial.suggest_int("n_epochs", 10, 40)
  n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])

  model = PPO("MlpPolicy", vec_env, verbose=0, n_epochs=n_epochs, learning_rate=learning_rate, ent_coef=ent_coef, gamma=0.95, clip_range=clip_range, n_steps=n_steps)

  model.learn(total_timesteps=100000, progress_bar=True)

  mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

  env.close()
  return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=20, callbacks=[logging_callback])

print("Best Hyperparameters: ", study.best_params)
print("Best Reward: ", study.best_value)