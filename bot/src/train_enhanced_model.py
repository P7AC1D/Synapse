"""
Training script for enhanced model with 34+ features.
This will train a new PPO-LSTM model using the enhanced feature set.
"""

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from trading.environment import TradingEnv
import warnings
warnings.filterwarnings('ignore')

def train_enhanced_model():
    """Train PPO model with enhanced features."""
    print("=" * 60)
    print("🚀 TRAINING ENHANCED MODEL WITH 34+ FEATURES")
    print("=" * 60)
    
    # Load data
    data_path = "../data/XAUUSDm_15min.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
        
    data = pd.read_csv(data_path)
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    
    # Use substantial amount of data for training
    total_bars = len(data)
    train_split = 0.8
    train_size = int(total_bars * train_split)
    
    train_data = data.iloc[:train_size].copy()
    val_data = data.iloc[train_size:].copy()
    
    print(f"📊 Data split:")
    print(f"   Total bars: {total_bars}")
    print(f"   Training: {len(train_data)} bars ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"   Validation: {len(val_data)} bars ({val_data.index[0]} to {val_data.index[-1]})")
    
    # Create environments
    print(f"\n🏗️ Creating environments...")
    train_env = TradingEnv(
        data=train_data,
        initial_balance=10000,
        random_start=True
    )
    
    val_env = TradingEnv(
        data=val_data,
        initial_balance=10000,
        random_start=False
    )
    
    # Wrap in vector environment
    train_env = DummyVecEnv([lambda: train_env])
    val_env = DummyVecEnv([lambda: val_env])
    
    print(f"✅ Environment created with {train_env.envs[0].raw_data.shape[1]} features")
    print(f"   Observation space: {train_env.observation_space}")
    print(f"   Action space: {train_env.action_space}")
    
    # Create model with enhanced architecture for more features
    print(f"\n🤖 Creating PPO model...")
    
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs=dict(
            lstm_hidden_size=256,  # Increased for more features
            n_lstm_layers=2,
            net_arch=dict(
                pi=[512, 256, 128],  # Increased policy network
                vf=[512, 256, 128]   # Increased value network
            ),
            activation_fn=torch.nn.ReLU,
            ortho_init=False
        )
    )
    
    print(f"✅ Model created with enhanced architecture")
    print(f"   Policy: MlpLstmPolicy")
    print(f"   LSTM hidden size: 256")
    print(f"   Network architecture: [512, 256, 128]")
    
    # Setup callbacks
    model_dir = "models/enhanced"
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="enhanced_model"
    )
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=model_dir,
        log_path=f"{model_dir}/logs",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    print(f"\n📚 Training configuration:")
    print(f"   Total timesteps: 200,000")
    print(f"   Checkpoint frequency: 10,000 steps")
    print(f"   Evaluation frequency: 5,000 steps")
    print(f"   Model save path: {model_dir}")
    
    # Train model
    print(f"\n🎯 Starting training...")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        model.learn(
            total_timesteps=200000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = f"{model_dir}/enhanced_model_final.zip"
        model.save(final_model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Final model saved: {final_model_path}")
        
        # Quick evaluation
        print(f"\n📊 Quick evaluation on validation data...")
        obs = val_env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(100):  # Quick 100-step test
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = val_env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if done[0]:
                break
                
        print(f"   Test steps: {steps}")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Average reward: {total_reward/steps:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function."""
    print("Enhanced Model Training")
    print("=" * 40)
    
    success = train_enhanced_model()
    
    if success:
        print(f"\n🎉 TRAINING SUCCESS!")
        print(f"✅ Enhanced model with 34+ features trained successfully")
        print(f"📈 Ready for performance comparison with baseline")
        print(f"")
        print(f"🔥 NEXT STEPS:")
        print(f"   1. Compare enhanced model vs baseline performance")
        print(f"   2. Analyze feature importance")
        print(f"   3. Fine-tune hyperparameters if needed")
        print(f"   4. Deploy for live testing")
    else:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Please check the error messages above and fix issues.")
        
    return success

if __name__ == "__main__":
    main()
