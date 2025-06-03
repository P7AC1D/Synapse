#!/usr/bin/env python3
"""
Deep Dive: Overfitting Analysis for DRL Trading Bot

This analysis specifically addresses the concerning performance gap between
combined dataset and validation set, providing actionable insights.
"""

import json
import numpy as np
from pathlib import Path

def analyze_overfitting_details(results_path):
    """Detailed overfitting analysis"""
    
    # Load latest iteration data
    latest_iter_file = Path(results_path) / "iterations" / "eval_results_iter_24.json"
    with open(latest_iter_file, 'r') as f:
        latest_data = json.load(f)[0]  # Get first (10k timesteps) entry
    
    print("=== DEEP DIVE: OVERFITTING ANALYSIS ===\n")
      # Extract key metrics
    combined_metrics = latest_data['combined']
    val_metrics = latest_data['validation']
    training_metrics = combined_metrics['training']  # Training metrics are under combined
    
    print("1. PERFORMANCE COMPARISON (Latest Iteration):")
    print(f"   Combined Dataset Return: {combined_metrics['account']['return']:.1f}%")
    print(f"   Validation Set Return: {val_metrics['account']['return']:.1f}%")
    print(f"   Performance Gap: {abs(combined_metrics['account']['return'] - val_metrics['account']['return']):.1f}%")
    
    # Dataset size analysis
    combined_steps = combined_metrics['trading']['steps_completed']
    val_steps = val_metrics['trading']['steps_completed']
    
    print(f"\n2. DATASET SIZE ANALYSIS:")
    print(f"   Combined Dataset Size: {combined_steps:,} steps")
    print(f"   Validation Dataset Size: {val_steps:,} steps")
    print(f"   Size Ratio: {combined_steps/val_steps:.1f}x larger")
    
    # Trading behavior comparison
    print(f"\n3. TRADING BEHAVIOR ANALYSIS:")
    
    combined_trades = combined_metrics['trading']['total_trades']
    val_trades = val_metrics['trading']['total_trades']
    
    print(f"   Combined Trades: {combined_trades}")
    print(f"   Validation Trades: {val_trades}")
    print(f"   Trade Density (trades/step):")
    print(f"     Combined: {combined_trades/combined_steps:.3f}")
    print(f"     Validation: {val_trades/val_steps:.3f}")
    
    # Win rate analysis
    combined_winrate = combined_metrics['trading']['win_rate']
    val_winrate = val_metrics['trading']['win_rate']
    
    print(f"   Win Rates:")
    print(f"     Combined: {combined_winrate:.1f}%")
    print(f"     Validation: {val_winrate:.1f}%")
    print(f"     Difference: {abs(combined_winrate - val_winrate):.1f}%")
    
    # Risk metrics
    combined_dd = combined_metrics['account']['max_dd']
    val_dd = val_metrics['account']['max_dd']
    
    print(f"   Maximum Drawdown:")
    print(f"     Combined: {combined_dd:.1f}%")
    print(f"     Validation: {val_dd:.1f}%")
    
    # Profit factor comparison
    combined_pf = combined_metrics['performance']['profit_factor']
    val_pf = val_metrics['performance']['profit_factor']
    
    print(f"   Profit Factor:")
    print(f"     Combined: {combined_pf:.3f}")
    print(f"     Validation: {val_pf:.3f}")
    
    print(f"\n4. ROOT CAUSE ANALYSIS:")
    
    # Check if it's memorization vs generalization
    trade_size_ratio = combined_trades / val_trades
    performance_ratio = combined_metrics['account']['return'] / max(val_metrics['account']['return'], 1)
    
    print(f"   Trade Volume Ratio: {trade_size_ratio:.1f}x")
    print(f"   Performance Ratio: {performance_ratio:.1f}x")
    
    if performance_ratio > trade_size_ratio * 2:
        print("   ⚠️  FINDING: Disproportionate performance gain suggests overfitting")
        print("      The model may be memorizing training patterns rather than learning generalizable strategies")
    else:
        print("   ✓ Performance gain proportional to opportunity size")
    
    # Entropy analysis
    entropy = training_metrics['entropy_loss']
    print(f"\n   Policy Entropy: {entropy:.3f}")
    if entropy > -1.0:
        print("   ⚠️  FINDING: High entropy may indicate insufficient convergence")
    elif entropy < -1.5:
        print("   ⚠️  FINDING: Low entropy suggests over-exploitation")
    else:
        print("   ✓ Entropy in healthy range")
    
    # Value loss analysis
    value_loss = training_metrics['value_loss']
    print(f"   Value Loss: {value_loss:.1f}")
    if value_loss > 50:
        print("   ⚠️  FINDING: High value loss indicates difficulty in value estimation")
        print("      This can lead to poor generalization")
    
    print(f"\n5. DETAILED RECOMMENDATIONS:")
    
    # Specific actionable recommendations
    print("   IMMEDIATE ACTIONS:")
    
    if performance_ratio > 3:
        print("   1. 🔴 CRITICAL: Implement stronger regularization")
        print("      - Add dropout layers (0.2-0.3)")
        print("      - Increase L2 regularization")
        print("      - Consider early stopping based on validation performance")
        
        print("   2. 🔴 CRITICAL: Expand validation dataset")
        print("      - Current validation set may be too small")
        print("      - Consider 80/20 split instead of current ratio")
        print("      - Ensure validation data covers different market conditions")
    
    print("   3. 🟡 MEDIUM: Review training process")
    print("      - Implement validation-based model selection")
    print("      - Consider ensemble methods")
    print("      - Add noise to training data")
    
    if value_loss > 50:
        print("   4. 🟡 MEDIUM: Improve value function approximation")
        print("      - Reduce learning rate")
        print("      - Increase network capacity for value function")
        print("      - Consider separate value and policy networks")
    
    print("   MONITORING:")
    print("   - Track validation performance trend")
    print("   - Monitor entropy for policy collapse")
    print("   - Implement cross-validation if possible")
    
    print(f"\n6. DEPLOYMENT READINESS:")
    
    if val_metrics['account']['return'] > 10 and val_pf > 1.0:
        print("   ✓ Validation performance shows profitability")
        print("   ✓ Positive profit factor on unseen data")
        deployment_risk = "MODERATE"
    elif val_metrics['account']['return'] > 0:
        print("   ⚠️  Marginal validation performance")
        deployment_risk = "HIGH"
    else:
        print("   🔴 Negative validation performance")
        deployment_risk = "VERY HIGH"
    
    print(f"   Deployment Risk: {deployment_risk}")
    
    if deployment_risk in ["HIGH", "VERY HIGH"]:
        print("   🚫 NOT RECOMMENDED for production deployment")
        print("   📋 Focus on improving generalization first")
    else:
        print("   ⚠️  Consider paper trading before live deployment")
        print("   📋 Implement strict risk management")
    
    return {
        'performance_gap': abs(combined_metrics['account']['return'] - val_metrics['account']['return']),
        'performance_ratio': performance_ratio,
        'validation_return': val_metrics['account']['return'],
        'validation_profit_factor': val_pf,
        'deployment_risk': deployment_risk,
        'entropy': entropy,
        'value_loss': value_loss
    }

def main():
    results_path = "c:/Dev/drl/bot/results/1006"
    
    analysis = analyze_overfitting_details(results_path)
    
    # Save analysis
    output_file = Path(results_path) / "overfitting_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n   Detailed overfitting analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
