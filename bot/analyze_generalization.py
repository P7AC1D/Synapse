#!/usr/bin/env python3
"""
DRL Trading Bot Generalization Analysis

Analyzes training progression to determine if the model is:
1. Generalizing well
2. Overfitting to training data
3. Underfitting (insufficient learning)

Key indicators:
- Training vs validation performance gap
- Consistency of performance across iterations
- Policy entropy trends
- Trading behavior patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(results_path):
    """Load all training metrics from curr_best_metrics.json"""
    metrics_file = Path(results_path) / "curr_best_metrics.json"
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract iteration data
    iterations = []
    for key, value in data.items():
        if key.startswith("iteration_"):
            iter_num = int(key.split("_")[1])
            iterations.append((iter_num, value))
    
    # Sort by iteration number
    iterations.sort(key=lambda x: x[0])
    return iterations

def analyze_generalization(iterations):
    """Analyze generalization patterns"""
    
    # Extract key metrics
    iter_nums = []
    train_returns = []
    val_returns = []
    combined_returns = []
    entropies = []
    train_trades = []
    val_trades = []
    win_rates_train = []
    win_rates_val = []
    value_losses = []
    
    for iter_num, data in iterations:
        iter_nums.append(iter_num)
        
        # Performance metrics
        combined_returns.append(data['combined']['account']['return'])
        val_returns.append(data['validation']['account']['return'])
        
        # Training metrics
        entropies.append(data['training']['entropy_loss'])
        value_losses.append(data['training']['value_loss'])
        
        # Trading activity
        train_trades.append(data['combined']['trading']['total_trades'])
        val_trades.append(data['validation']['trading']['total_trades'])
        
        # Win rates
        win_rates_train.append(data['combined']['trading']['win_rate'])
        win_rates_val.append(data['validation']['trading']['win_rate'])
        
        # Calculate average return for "training"
        avg_return = (data['combined']['account']['return'] + data['validation']['account']['return']) / 2
        train_returns.append(avg_return)
    
    # Convert to numpy arrays
    iter_nums = np.array(iter_nums)
    train_returns = np.array(train_returns)
    val_returns = np.array(val_returns)
    combined_returns = np.array(combined_returns)
    entropies = np.array(entropies)
    
    # Analysis
    print("=== DRL TRADING BOT GENERALIZATION ANALYSIS ===\n")
    
    # 1. Performance Gap Analysis
    print("1. PERFORMANCE GAP ANALYSIS:")
    performance_gap = np.abs(combined_returns - val_returns)
    avg_gap = np.mean(performance_gap)
    gap_trend = np.polyfit(iter_nums, performance_gap, 1)[0]
    
    print(f"   Average performance gap: {avg_gap:.2f}%")
    print(f"   Gap trend (slope): {gap_trend:.4f} (negative = improving)")
    print(f"   Latest gap: {performance_gap[-1]:.2f}%")
    
    if avg_gap < 20:
        gap_status = "GOOD - Low bias between combined and validation"
    elif avg_gap < 50:
        gap_status = "MODERATE - Some overfitting tendency"
    else:
        gap_status = "HIGH - Potential overfitting"
    print(f"   Status: {gap_status}\n")
    
    # 2. Performance Trend Analysis
    print("2. PERFORMANCE TREND ANALYSIS:")
    val_trend = np.polyfit(iter_nums, val_returns, 1)[0]
    combined_trend = np.polyfit(iter_nums, combined_returns, 1)[0]
    
    print(f"   Validation trend: {val_trend:.2f}%/iteration")
    print(f"   Combined trend: {combined_trend:.2f}%/iteration")
    
    if val_trend > 0 and combined_trend > 0:
        trend_status = "HEALTHY - Both improving"
    elif val_trend < -1:
        trend_status = "CONCERNING - Validation declining"
    else:
        trend_status = "STABLE - Consistent performance"
    print(f"   Status: {trend_status}\n")
    
    # 3. Entropy Analysis (Exploration vs Exploitation)
    print("3. POLICY ENTROPY ANALYSIS:")
    entropy_trend = np.polyfit(iter_nums, entropies, 1)[0]
    latest_entropy = entropies[-1]
    
    print(f"   Latest entropy: {latest_entropy:.3f}")
    print(f"   Entropy trend: {entropy_trend:.5f}/iteration")
    print(f"   Entropy range: [{np.min(entropies):.3f}, {np.max(entropies):.3f}]")
    
    if latest_entropy > -1.0:
        entropy_status = "CONCERNING - May collapse to deterministic policy"
    elif latest_entropy > -1.2:
        entropy_status = "MODERATE - Good exploration/exploitation balance"
    else:
        entropy_status = "HEALTHY - Strong exploration maintained"
    print(f"   Status: {entropy_status}\n")
    
    # 4. Trading Activity Consistency
    print("4. TRADING ACTIVITY ANALYSIS:")
    train_trades = np.array(train_trades)
    val_trades = np.array(val_trades)
    
    avg_train_trades = np.mean(train_trades)
    avg_val_trades = np.mean(val_trades)
    trade_consistency = np.std(val_trades) / np.mean(val_trades)
    
    print(f"   Avg combined trades: {avg_train_trades:.0f}")
    print(f"   Avg validation trades: {avg_val_trades:.0f}")
    print(f"   Validation trade consistency (CV): {trade_consistency:.2f}")
    
    if trade_consistency < 0.3:
        trade_status = "CONSISTENT - Stable trading behavior"
    elif trade_consistency < 0.5:
        trade_status = "MODERATE - Some variability in trading"
    else:
        trade_status = "INCONSISTENT - High trading variability"
    print(f"   Status: {trade_status}\n")
    
    # 5. Overall Assessment
    print("5. OVERALL GENERALIZATION ASSESSMENT:")
    
    # Calculate generalization score
    gap_score = max(0, 1 - avg_gap/50)  # 0-1, higher is better
    trend_score = 1 if val_trend >= 0 else max(0, 1 + val_trend/10)
    entropy_score = 1 if latest_entropy < -1.2 else max(0, 1 - abs(latest_entropy + 1.0))
    consistency_score = max(0, 1 - trade_consistency)
    
    overall_score = (gap_score + trend_score + entropy_score + consistency_score) / 4
    
    print(f"   Performance Gap Score: {gap_score:.2f}/1.0")
    print(f"   Trend Score: {trend_score:.2f}/1.0")
    print(f"   Entropy Score: {entropy_score:.2f}/1.0")
    print(f"   Consistency Score: {consistency_score:.2f}/1.0")
    print(f"   Overall Generalization Score: {overall_score:.2f}/1.0")
    
    if overall_score > 0.75:
        assessment = "EXCELLENT - Model is generalizing very well"
    elif overall_score > 0.6:
        assessment = "GOOD - Model shows good generalization with minor concerns"
    elif overall_score > 0.4:
        assessment = "MODERATE - Some overfitting/underfitting signs"
    else:
        assessment = "POOR - Significant generalization issues"
    
    print(f"\n   FINAL ASSESSMENT: {assessment}")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS:")
    
    if gap_score < 0.5:
        print("   • HIGH PRIORITY: Address overfitting")
        print("     - Consider reducing model complexity")
        print("     - Increase regularization")
        print("     - Use more diverse training data")
    
    if entropy_score < 0.5:
        print("   • Monitor policy collapse risk")
        print("     - Current anti-collapse system is helping")
        print("     - Continue entropy monitoring")
    
    if trend_score < 0.5:
        print("   • Performance declining - investigate:")
        print("     - Market regime changes")
        print("     - Training instability")
        print("     - Hyperparameter tuning")
    
    if consistency_score < 0.5:
        print("   • Improve trading consistency:")
        print("     - Stabilize training process")
        print("     - Review reward function")
    
    if overall_score > 0.75:
        print("   • Model is ready for deployment consideration")
        print("   • Continue monitoring in production")
        print("   • Consider incremental improvements")
    
    return {
        'overall_score': overall_score,
        'gap_score': gap_score,
        'trend_score': trend_score,
        'entropy_score': entropy_score,
        'consistency_score': consistency_score,
        'latest_entropy': latest_entropy,
        'avg_gap': avg_gap,
        'val_trend': val_trend
    }

def create_visualization(iterations, save_path=None):
    """Create visualization of training progression"""
    
    iter_nums = []
    val_returns = []
    combined_returns = []
    entropies = []
    val_trades = []
    
    for iter_num, data in iterations:
        iter_nums.append(iter_num)
        val_returns.append(data['validation']['account']['return'])
        combined_returns.append(data['combined']['account']['return'])
        entropies.append(data['training']['entropy_loss'])
        val_trades.append(data['validation']['trading']['total_trades'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance comparison
    ax1.plot(iter_nums, combined_returns, 'b-', label='Combined Dataset', linewidth=2)
    ax1.plot(iter_nums, val_returns, 'r-', label='Validation Set', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Performance: Combined vs Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance gap
    gap = np.abs(np.array(combined_returns) - np.array(val_returns))
    ax2.plot(iter_nums, gap, 'g-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Performance Gap (%)')
    ax2.set_title('Overfitting Indicator (Performance Gap)')
    ax2.grid(True, alpha=0.3)
    
    # Entropy trend
    ax3.plot(iter_nums, entropies, 'purple', linewidth=2)
    ax3.axhline(y=-1.0, color='red', linestyle='--', alpha=0.7, label='Collapse Risk')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Policy Entropy')
    ax3.set_title('Policy Entropy (Exploration vs Exploitation)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Trading activity
    ax4.plot(iter_nums, val_trades, 'orange', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Validation Trades')
    ax4.set_title('Trading Activity Consistency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {save_path}")
    else:
        plt.show()

def main():
    results_path = "c:/Dev/drl/bot/results/1006"
    
    # Load and analyze data
    iterations = load_metrics(results_path)
    analysis_results = analyze_generalization(iterations)
    
    # Create visualization
    viz_path = Path(results_path) / "generalization_analysis.png"
    create_visualization(iterations, viz_path)
    
    # Save detailed results
    output_file = Path(results_path) / "generalization_report.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n   Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
