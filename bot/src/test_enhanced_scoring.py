"""Test script to compare original vs enhanced scoring on your evaluation results.

This script shows how the enhanced scoring system would have selected different models
compared to the original simple average approach.
"""

def calculate_original_score(val_return, combined_return, profit_factor):
    """Original scoring: simple 50/50 average + profit factor bonus"""
    if val_return <= 0 or combined_return <= 0:
        return float('-inf')
    
    average_return = (val_return + combined_return) / 2
    pf_bonus = min(max(0, profit_factor - 1.0), 2.0) * 0.1 if profit_factor > 1.0 else 0
    return average_return + pf_bonus

def calculate_directional_balance_score(long_trades, short_trades):
    """Calculate directional balance score that rewards balanced long/short trading."""
    if long_trades == 0 and short_trades == 0:
        return 0.0
    if long_trades == 0 or short_trades == 0:
        return 0.3
    
    ratio = max(long_trades, short_trades) / min(long_trades, short_trades)
    return 1.0 / (1.0 + (ratio - 1.0) * 0.5)

def calculate_risk_reward_score(avg_win_points, avg_loss_points):
    """Calculate risk-reward score that rewards R:R ratios above 1.0."""
    if avg_loss_points >= 0 or avg_win_points <= 0:
        return 0.5
    
    rr_ratio = avg_win_points / abs(avg_loss_points)
    
    if rr_ratio >= 1.0:
        return min(1.0, 0.5 + (rr_ratio - 1.0) * 0.3)
    else:
        return max(0.1, rr_ratio * 0.5)

def calculate_enhanced_score(val_return, combined_return, val_performance):
    """Enhanced scoring with multiple factors."""
    # 80/20 validation weighting for base performance
    base_score = (val_return * 0.80 + combined_return * 0.20)
    
    if base_score <= 0:
        return float('-inf')
    
    # Risk-to-reward component
    rr_score = calculate_risk_reward_score(
        val_performance.get('avg_win_points', 0), 
        val_performance.get('avg_loss_points', 0)
    )
    
    # Directional balance component
    balance_score = calculate_directional_balance_score(
        val_performance.get('long_trades', 0),
        val_performance.get('short_trades', 0)
    )
    
    # Consistency component
    if val_return > 0 and combined_return > 0:
        consistency = min(val_return, combined_return) / max(val_return, combined_return)
    else:
        consistency = 0.0
    
    # Profit factor bonus
    pf_bonus = 0.0
    if val_performance.get('profit_factor', 1.0) > 1.0:
        pf_bonus = min(val_performance['profit_factor'] - 1.0, 2.0) * 0.05
    
    # Final weighted score
    final_score = (
        base_score * 0.60 +           # 60% performance (80/20 weighted)
        rr_score * 0.20 +             # 20% risk-reward
        balance_score * 0.15 +        # 15% directional balance  
        consistency * 0.05 +          # 5% consistency bonus
        pf_bonus                      # Small profit factor bonus
    )
    
    return final_score

def main():
    """Test enhanced scoring on your evaluation results."""
    print("=" * 70)
    print("🔬 ENHANCED SCORING SYSTEM TEST")
    print("=" * 70)
    
    # Your original evaluation results
    timesteps = [
        {
            'step': 20000,
            'combined_return': 0.2724,
            'validation_return': 0.0140,
            'val_performance': {
                'avg_win_points': 1981.2,
                'avg_loss_points': -1938.1,
                'profit_factor': 1.14,
                'long_trades': 39,
                'short_trades': 14
            }
        },
        {
            'step': 60000,
            'combined_return': 0.1211,
            'validation_return': 0.0326,
            'val_performance': {
                'avg_win_points': 1114.7,
                'avg_loss_points': -1267.0,
                'profit_factor': 1.40,
                'long_trades': 30,
                'short_trades': 53
            }
        }
    ]
    
    print("\n📊 COMPARISON OF SCORING METHODS:\n")
    
    best_original = None
    best_enhanced = None
    
    for ts in timesteps:
        step = ts['step']
        combined_ret = ts['combined_return']
        val_ret = ts['validation_return']
        val_perf = ts['val_performance']
        
        # Calculate scores
        original_score = calculate_original_score(val_ret, combined_ret, val_perf['profit_factor'])
        enhanced_score = calculate_enhanced_score(val_ret, combined_ret, val_perf)
        
        # Calculate components for enhanced score
        base_score = (val_ret * 0.80 + combined_ret * 0.20)
        rr_score = calculate_risk_reward_score(val_perf['avg_win_points'], val_perf['avg_loss_points'])
        balance_score = calculate_directional_balance_score(val_perf['long_trades'], val_perf['short_trades'])
        consistency = min(val_ret, combined_ret) / max(val_ret, combined_ret) if val_ret > 0 and combined_ret > 0 else 0.0
        rr_ratio = val_perf['avg_win_points'] / abs(val_perf['avg_loss_points'])
        directional_ratio = max(val_perf['long_trades'], val_perf['short_trades']) / min(val_perf['long_trades'], val_perf['short_trades'])
        
        print(f"Timestep {step:,d}:")
        print(f"  Returns: Combined {combined_ret*100:.2f}%, Validation {val_ret*100:.2f}%")
        print(f"  Risk-Reward: {rr_ratio:.2f} (Win: {val_perf['avg_win_points']:.1f} pts, Loss: {val_perf['avg_loss_points']:.1f} pts)")
        print(f"  Direction Balance: L:{val_perf['long_trades']} S:{val_perf['short_trades']} (Ratio: {directional_ratio:.1f})")
        print(f"  Profit Factor: {val_perf['profit_factor']:.2f}")
        print(f"")
        print(f"  📈 Original Score: {original_score:.4f}")
        print(f"     - Simple Average: {(val_ret + combined_ret) / 2 * 100:.2f}%")
        print(f"     - PF Bonus: {min(max(0, val_perf['profit_factor'] - 1.0), 2.0) * 0.1:.3f}")
        print(f"")
        print(f"  🚀 Enhanced Score: {enhanced_score:.4f}")
        print(f"     - Base Score (80/20): {base_score*100:.2f}%")
        print(f"     - Risk-Reward Score: {rr_score:.3f}")
        print(f"     - Balance Score: {balance_score:.3f}")
        print(f"     - Consistency: {consistency:.3f}")
        print(f"")
        
        # Track best models
        if best_original is None or original_score > best_original['score']:
            best_original = {'step': step, 'score': original_score}
        
        if best_enhanced is None or enhanced_score > best_enhanced['score']:
            best_enhanced = {'step': step, 'score': enhanced_score}
        
        print("-" * 50)
    
    print(f"\n🏆 BEST MODEL SELECTION:")
    print(f"")
    print(f"Original Scoring Selected: Timestep {best_original['step']:,d} (Score: {best_original['score']:.4f})")
    print(f"Enhanced Scoring Selects:  Timestep {best_enhanced['step']:,d} (Score: {best_enhanced['score']:.4f})")
    
    if best_original['step'] != best_enhanced['step']:
        print(f"")
        print(f"✅ ENHANCED SCORING FIXES THE ISSUE!")
        print(f"   - Original wrongly selected timestep {best_original['step']:,d}")
        print(f"   - Enhanced correctly selects timestep {best_enhanced['step']:,d}")
        print(f"   - Enhanced scoring favors validation performance and trading quality")
    else:
        print(f"")
        print(f"ℹ️  Both methods select the same model")
    
    print(f"\n💡 KEY IMPROVEMENTS:")
    print(f"   - 80/20 validation weighting (vs 50/50 original)")
    print(f"   - Rewards good risk-reward ratios (>1.0)")
    print(f"   - Rewards balanced long/short trading")
    print(f"   - Penalizes models that overfit to training data")
    print(f"   - Considers trading quality, not just returns")

if __name__ == "__main__":
    main()
