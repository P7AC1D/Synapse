#!/usr/bin/env python3
"""
Fix for short position holding behavior in trading bot.
This script applies the necessary changes to the reward system.
"""

import os
import sys

def apply_reward_fixes():
    """Apply all necessary fixes to the reward system."""
    
    rewards_file = r"c:\Dev\drl\bot\src\trading\rewards.py"
    
    # Read the original file
    with open(rewards_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying position holding fixes to reward system...")
    
    # Fix 1: Increase hold rewards and adjust thresholds
    content = content.replace(
        "        self.PROFIT_HOLD_REWARD = 0.5           # Strong reward for holding profitable positions",
        "        self.PROFIT_HOLD_REWARD = 1.5           # INCREASED: Strong reward for holding profitable positions"
    )
    
    content = content.replace(
        "        self.LOSS_HOLD_PENALTY = -0.2           # Penalty for holding losing positions",
        "        self.LOSS_HOLD_PENALTY = -0.1           # REDUCED: Smaller penalty for holding losing positions"
    )
    
    content = content.replace(
        "        self.SIGNIFICANT_PROFIT_THRESHOLD = 0.01 # 1% profit threshold for bonus rewards",
        "        self.SIGNIFICANT_PROFIT_THRESHOLD = 0.005 # REDUCED: 0.5% profit threshold for bonus rewards"
    )
    
    content = content.replace(
        "        self.SIGNIFICANT_PROFIT_BONUS = 0.2     # Extra bonus for significantly profitable positions",
        "        self.SIGNIFICANT_PROFIT_BONUS = 0.8     # INCREASED: Extra bonus for significantly profitable positions"
    )
    
    # Fix 2: Improve time efficiency calculation for position quality
    old_time_efficiency = """        # Time efficiency score (reward quicker profitable trades)
        time_efficiency = max(0.1, 1.0 - (hold_time / 100.0))"""
    
    new_time_efficiency = """        # FIXED: Time efficiency that encourages holding profitable positions longer
        # For profitable trades, don't penalize holding (flat bonus)
        # For losing trades, encourage quick closure
        if pnl > 0:
            # Profitable positions: flat time efficiency (no decay for first 50 bars)
            time_efficiency = max(0.7, 1.0 - max(0, (hold_time - 50) / 100.0))
        else:
            # Losing positions: encourage quick closure
            time_efficiency = max(0.1, 1.0 - (hold_time / 30.0))"""
    
    content = content.replace(old_time_efficiency, new_time_efficiency)
    
    # Fix 3: Improve hold decay calculation
    old_hold_decay = """                    hold_decay = max(0.3, 1.0 - (current_hold / 80.0))  # Slower decay, higher minimum"""
    new_hold_decay = """                    hold_decay = max(0.8, 1.0 - max(0, (current_hold - 30) / 200.0))  # FIXED: Much slower decay, no penalty for first 30 bars"""
    
    content = content.replace(old_hold_decay, new_hold_decay)
    
    # Fix 4: Improve significant profit bonus calculation
    old_profit_bonus = """                        profit_bonus = self.SIGNIFICANT_PROFIT_BONUS * min(profit_ratio * 20, 3.0)"""
    new_profit_bonus = """                        profit_bonus = self.SIGNIFICANT_PROFIT_BONUS * min(profit_ratio * 50, 5.0)  # INCREASED multiplier"""
    
    content = content.replace(old_profit_bonus, new_profit_bonus)
    
    # Fix 5: Reduce loss penalty escalation
    old_loss_penalty = """                    loss_penalty_multiplier = min(2.0, 1.0 + (current_hold / 20.0))"""
    new_loss_penalty = """                    loss_penalty_multiplier = min(1.5, 1.0 + (current_hold / 40.0))  # REDUCED: Slower escalation"""
    
    content = content.replace(old_loss_penalty, new_loss_penalty)
    
    # Write the updated file
    with open(rewards_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully applied all position holding fixes!")
    print("\nChanges made:")
    print("1. INCREASED PROFIT_HOLD_REWARD: 0.5 → 1.5")
    print("2. REDUCED LOSS_HOLD_PENALTY: -0.2 → -0.1") 
    print("3. LOWERED SIGNIFICANT_PROFIT_THRESHOLD: 1% → 0.5%")
    print("4. INCREASED SIGNIFICANT_PROFIT_BONUS: 0.2 → 0.8")
    print("5. FIXED time efficiency to not penalize profitable position holding")
    print("6. IMPROVED hold decay (no penalty for first 30 bars)")
    print("7. INCREASED profit bonus multiplier")
    print("8. REDUCED loss penalty escalation")
    
    return True

if __name__ == "__main__":
    try:
        apply_reward_fixes()
        print("\n🎯 Position holding fixes applied successfully!")
        print("The agent should now be incentivized to hold profitable positions longer.")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        sys.exit(1)
