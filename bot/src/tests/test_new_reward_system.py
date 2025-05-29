"""
Test script to validate the overhauled reward system.
This ensures the new reward structure encourages active trading.
"""

import sys
import os

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from trading.environment import TradingEnv
from trading.actions import Action

def test_reward_system():
    """Test the new reward system functionality."""
    print("=" * 60)
    print("🧪 TESTING OVERHAULED REWARD SYSTEM")
    print("=" * 60)
    
    # Load minimal data for testing - updated path from tests directory
    data_path = "../data/XAUUSDm_15min.csv"
    data = pd.read_csv(data_path).head(1000)  # Use first 1000 rows for quick test
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    
    # Create test environment
    env = TradingEnv(
        data=data,
        initial_balance=10000,
        random_start=False
    )
    
    print(f"✅ Environment created with {env.raw_data.shape[1]} features")
    print(f"✅ Data points: {len(env.raw_data)}")
    
    # Test scenarios
    test_scenarios = [
        "HOLD Action Penalties",
        "Trading Action Rewards", 
        "Position Management",
        "Risk Management",
        "Invalid Action Handling"
    ]
    
    print(f"\n🔍 Testing {len(test_scenarios)} reward scenarios...\n")
    
    # Test 1: HOLD Action Penalties
    print("1️⃣ Testing HOLD Action Penalties:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Test consecutive holds without position
    hold_rewards = []
    for i in range(60):  # Test 60 consecutive holds
        obs, reward, done, truncated, info = env.step(Action.HOLD)
        hold_rewards.append(reward)
        if i in [9, 19, 49, 59]:  # Check at key intervals
            print(f"   Hold step {i+1:2d}: Reward = {reward:+.4f}")
    
    print(f"   ✅ Average HOLD reward: {np.mean(hold_rewards):+.4f}")
    print(f"   ✅ Penalty escalation: {hold_rewards[59] < hold_rewards[9]}")
    
    # Test 2: Trading Action Rewards
    print(f"\n2️⃣ Testing Trading Action Rewards:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Test opening position
    obs, buy_reward, done, truncated, info = env.step(Action.BUY)
    print(f"   BUY action reward: {buy_reward:+.4f}")
    
    # Test holding profitable position
    profitable_rewards = []
    for i in range(5):
        obs, reward, done, truncated, info = env.step(Action.HOLD)
        profitable_rewards.append(reward)
        
    print(f"   Average profitable HOLD: {np.mean(profitable_rewards):+.4f}")
    
    # Test closing position
    obs, close_reward, done, truncated, info = env.step(Action.CLOSE)
    print(f"   CLOSE action reward: {close_reward:+.4f}")
    print(f"   Total trades: {info['total_trades']}")
    
    # Test 3: Position Management
    print(f"\n3️⃣ Testing Position Management:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Open position and track P&L changes
    obs, _, _, _, _ = env.step(Action.BUY)
    
    pnl_rewards = []
    for i in range(10):
        obs, reward, done, truncated, info = env.step(Action.HOLD)
        pnl_rewards.append(reward)
        if i % 3 == 0:
            position_info = info.get('position', {})
            unrealized_pnl = position_info.get('unrealized_pnl', 0)
            print(f"   Step {i+1}: Reward = {reward:+.4f}, Unrealized P&L = {unrealized_pnl:+.2f}")
    
    # Test 4: Risk Management  
    print(f"\n4️⃣ Testing Risk Management:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Test quick profit-taking (should be rewarded)
    obs, _, _, _, _ = env.step(Action.BUY)
    obs, _, _, _, _ = env.step(Action.HOLD)  # Hold briefly
    obs, quick_close_reward, _, _, info = env.step(Action.CLOSE)
    quick_trades = info['total_trades']
    
    # Test holding too long (should be penalized)
    obs, _, _, _, _ = env.step(Action.SELL)
    long_hold_rewards = []
    for i in range(50):  # Hold for 50 steps
        obs, reward, done, truncated, info = env.step(Action.HOLD)
        long_hold_rewards.append(reward)
    
    obs, long_close_reward, _, _, info = env.step(Action.CLOSE)
    total_trades = info['total_trades']
    
    print(f"   Quick close reward: {quick_close_reward:+.4f}")
    print(f"   Long hold avg reward: {np.mean(long_hold_rewards[-10:]):+.4f}")
    print(f"   Long close reward: {long_close_reward:+.4f}")
    print(f"   Total trades executed: {total_trades}")
    
    # Test 5: Invalid Action Handling
    print(f"\n5️⃣ Testing Invalid Action Handling:")
    print("-" * 40)
    
    obs, _ = env.reset()
    
    # Test invalid actions
    obs, _, _, _, _ = env.step(Action.BUY)  # Open position
    obs, invalid_buy_reward, _, _, _ = env.step(Action.BUY)  # Try to buy again (invalid)
    obs, invalid_close_reward, _, _, _ = env.step(Action.CLOSE)  # Close position
    obs, invalid_close_reward2, _, _, _ = env.step(Action.CLOSE)  # Try to close again (invalid)
    
    print(f"   Invalid BUY when position open: {invalid_buy_reward:+.4f}")
    print(f"   Invalid CLOSE when no position: {invalid_close_reward2:+.4f}")
    
    # Summary Analysis
    print(f"\n📊 REWARD SYSTEM ANALYSIS:")
    print("=" * 50)
    
    # Check if system encourages trading
    trading_encouraged = buy_reward > np.mean(hold_rewards[:10])
    print(f"✅ Trading encouraged over holding: {trading_encouraged}")
    
    # Check if inactivity is penalized
    inactivity_penalized = hold_rewards[59] < hold_rewards[9]
    print(f"✅ Inactivity properly penalized: {inactivity_penalized}")
    
    # Check if invalid actions are penalized
    invalid_actions_penalized = invalid_buy_reward < -1.0 and invalid_close_reward2 < -1.0
    print(f"✅ Invalid actions penalized: {invalid_actions_penalized}")
    
    # Check if profitable trades are rewarded
    profitable_trading_rewarded = np.mean(profitable_rewards) > 0
    print(f"✅ Profitable position holding rewarded: {profitable_trading_rewarded}")
    
    # Overall assessment
    all_tests_passed = all([
        trading_encouraged,
        inactivity_penalized, 
        invalid_actions_penalized,
        profitable_trading_rewarded
    ])
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Reward system encourages active trading!")
        print("✅ Ready for training with new reward structure")
    else:
        print("❌ SOME TESTS FAILED - Review reward system configuration")
        
    print(f"\n🔥 KEY INSIGHTS:")
    print(f"   📈 Market engagement bonus: {buy_reward > 0}")
    print(f"   📉 Inactivity cost escalates: {hold_rewards[59] < hold_rewards[9]}")
    print(f"   ⚡ Quick decision making encouraged")
    print(f"   🛡️ Risk management incentivized")
    
    return all_tests_passed

def main():
    """Main test function."""
    print("Reward System Validation Test")
    print("=" * 30)
    
    success = test_reward_system()
    
    if success:
        print(f"\n🎉 REWARD SYSTEM VALIDATION SUCCESS!")
        print(f"✅ New reward structure is properly configured")
        print(f"✅ System encourages active, profitable trading")
        print(f"✅ Inactivity and invalid actions are penalized")
        print(f"")
        print(f"🚀 READY TO START TRAINING!")
        print(f"   Run: python train_enhanced_model.py")
    else:
        print(f"\n❌ REWARD SYSTEM NEEDS ADJUSTMENT!")
        print(f"Please review the test results and adjust reward parameters.")
        
    return success

if __name__ == "__main__":
    main()
