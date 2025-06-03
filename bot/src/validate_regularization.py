#!/usr/bin/env python3
"""
Regularization Validation Script

This script validates that all critical overfitting fixes identified in the 
generalization analysis have been properly implemented.

Based on findings:
- 1,169% performance gap between training (+1,146%) and validation (-23.7%)
- 5 critical issues identified:
  1. Improper data splitting (90/10 → 70/20/10)
  2. Combined dataset model selection → Validation-only
  3. Insufficient regularization → Stronger constraints
  4. Oversized architecture → Reduced complexity  
  5. No early stopping → Validation-based stopping

Usage:
    python validate_regularization.py
"""

import os
import sys
import json
from datetime import datetime

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def validate_config_import():
    """Test that regularized configuration can be imported."""
    try:
        from configs.regularized_training_config import (
            REGULARIZED_TRAINING_CONFIG,
            REGULARIZED_POLICY_KWARGS,
            REGULARIZED_MODEL_KWARGS,
            REGULARIZED_DATA_CONFIG,
            REGULARIZED_VALIDATION_CONFIG
        )
        print("✅ Regularized configuration import successful")
        return True, (REGULARIZED_TRAINING_CONFIG, REGULARIZED_POLICY_KWARGS, 
                     REGULARIZED_MODEL_KWARGS, REGULARIZED_DATA_CONFIG, REGULARIZED_VALIDATION_CONFIG)
    except Exception as e:
        print(f"❌ Failed to import regularized configuration: {e}")
        return False, None

def validate_training_utils_import():
    """Test that regularized training utilities can be imported."""
    try:
        from utils.regularized_training_utils import (
            train_regularized_walk_forward,
            validate_regularization_implementation,
            RegularizedEvalCallback,
            create_regularized_data_splits
        )
        print("✅ Regularized training utilities import successful")
        return True
    except Exception as e:
        print(f"❌ Failed to import regularized training utilities: {e}")
        return False

def detailed_config_validation(configs):
    """Perform detailed validation of each configuration component."""
    (REGULARIZED_TRAINING_CONFIG, REGULARIZED_POLICY_KWARGS, 
     REGULARIZED_MODEL_KWARGS, REGULARIZED_DATA_CONFIG, REGULARIZED_VALIDATION_CONFIG) = configs
    
    print(f"\n📊 DETAILED CONFIGURATION VALIDATION:")
    
    issues = []
    fixes = []
    
    # 1. Data Splitting Validation
    print(f"\n1️⃣ DATA SPLITTING:")
    train_split = REGULARIZED_DATA_CONFIG.get('train_split', 0)
    val_split = REGULARIZED_DATA_CONFIG.get('validation_split', 0)
    test_split = REGULARIZED_DATA_CONFIG.get('test_split', 0)
    
    if train_split == 0.7 and val_split == 0.2 and test_split == 0.1:
        print(f"   ✅ Data splits: {train_split*100:.0f}%/{val_split*100:.0f}%/{test_split*100:.0f}% (FIXED from 90/10)")
        fixes.append("Data splitting improved to 70/20/10")
    else:
        print(f"   ❌ Data splits: {train_split*100:.0f}%/{val_split*100:.0f}%/{test_split*100:.0f}% (Should be 70/20/10)")
        issues.append("Data splitting not properly configured")
    
    # 2. Model Selection Validation  
    print(f"\n2️⃣ MODEL SELECTION:")
    selection_criterion = REGULARIZED_VALIDATION_CONFIG.get('selection_criterion', 'unknown')
    
    if selection_criterion == 'validation_return':
        print(f"   ✅ Selection criterion: '{selection_criterion}' (FIXED from combined)")
        fixes.append("Model selection changed to validation-only")
    else:
        print(f"   ❌ Selection criterion: '{selection_criterion}' (Should be 'validation_return')")
        issues.append("Model selection still using combined dataset")
    
    # 3. Regularization Validation
    print(f"\n3️⃣ REGULARIZATION:")
    learning_rate = REGULARIZED_MODEL_KWARGS.get('learning_rate', 0)
    grad_norm = REGULARIZED_MODEL_KWARGS.get('max_grad_norm', 0)
    weight_decay = REGULARIZED_MODEL_KWARGS.get('optimizer_kwargs', {}).get('weight_decay', 0)
    batch_size = REGULARIZED_MODEL_KWARGS.get('batch_size', 0)
    
    reg_checks = []
    
    if learning_rate <= 0.0005:
        print(f"   ✅ Learning rate: {learning_rate} (≤0.0005)")
        reg_checks.append(True)
    else:
        print(f"   ❌ Learning rate: {learning_rate} (Should be ≤0.0005)")
        reg_checks.append(False)
    
    if grad_norm <= 0.5:
        print(f"   ✅ Gradient clipping: {grad_norm} (≤0.5)")
        reg_checks.append(True)
    else:
        print(f"   ❌ Gradient clipping: {grad_norm} (Should be ≤0.5)")
        reg_checks.append(False)
    
    if weight_decay >= 1e-3:
        print(f"   ✅ Weight decay: {weight_decay} (≥1e-3)")
        reg_checks.append(True)
    else:
        print(f"   ❌ Weight decay: {weight_decay} (Should be ≥1e-3)")
        reg_checks.append(False)
    
    if batch_size <= 64:
        print(f"   ✅ Batch size: {batch_size} (≤64)")
        reg_checks.append(True)
    else:
        print(f"   ❌ Batch size: {batch_size} (Should be ≤64)")
        reg_checks.append(False)
    
    if all(reg_checks):
        fixes.append("Regularization parameters strengthened")
    else:
        issues.append("Regularization parameters insufficient")
    
    # 4. Architecture Validation
    print(f"\n4️⃣ ARCHITECTURE:")
    lstm_size = REGULARIZED_POLICY_KWARGS.get('lstm_hidden_size', 0)
    lstm_layers = REGULARIZED_POLICY_KWARGS.get('n_lstm_layers', 0)
    pi_arch = REGULARIZED_POLICY_KWARGS.get('net_arch', {}).get('pi', [])
    vf_arch = REGULARIZED_POLICY_KWARGS.get('net_arch', {}).get('vf', [])
    
    arch_checks = []
    
    if lstm_size <= 256:
        print(f"   ✅ LSTM hidden size: {lstm_size} (≤256, reduced from 512)")
        arch_checks.append(True)
    else:
        print(f"   ❌ LSTM hidden size: {lstm_size} (Should be ≤256)")
        arch_checks.append(False)
    
    if lstm_layers <= 2:
        print(f"   ✅ LSTM layers: {lstm_layers} (≤2, reduced from 4)")
        arch_checks.append(True)
    else:
        print(f"   ❌ LSTM layers: {lstm_layers} (Should be ≤2)")
        arch_checks.append(False)
    
    if all(size <= 128 for size in pi_arch):
        print(f"   ✅ Policy network: {pi_arch} (all ≤128, reduced from 256)")
        arch_checks.append(True)
    else:
        print(f"   ❌ Policy network: {pi_arch} (Should be all ≤128)")
        arch_checks.append(False)
    
    if all(size <= 128 for size in vf_arch):
        print(f"   ✅ Value network: {vf_arch} (all ≤128, reduced from 256)")
        arch_checks.append(True)
    else:
        print(f"   ❌ Value network: {vf_arch} (Should be all ≤128)")
        arch_checks.append(False)
    
    if all(arch_checks):
        fixes.append("Architecture complexity reduced")
    else:
        issues.append("Architecture still too complex")
    
    # 5. Early Stopping Validation
    print(f"\n5️⃣ EARLY STOPPING:")
    early_stopping = REGULARIZED_VALIDATION_CONFIG.get('early_stopping', {})
    es_enabled = early_stopping.get('enabled', False)
    es_metric = early_stopping.get('metric', 'unknown')
    es_patience = early_stopping.get('patience', 0)
    
    if es_enabled and es_metric == 'validation_return' and es_patience >= 10:
        print(f"   ✅ Early stopping: Enabled with '{es_metric}' metric, patience={es_patience}")
        fixes.append("Early stopping implemented with validation-based monitoring")
    else:
        print(f"   ❌ Early stopping: Enabled={es_enabled}, Metric='{es_metric}', Patience={es_patience}")
        issues.append("Early stopping not properly configured")
    
    return issues, fixes

def main():
    print(f"{'='*70}")
    print(f"🛡️ REGULARIZATION VALIDATION - OVERFITTING FIXES")
    print(f"{'='*70}")
    print(f"Purpose: Validate fixes for 1,169% performance gap")
    print(f"Target: Training +1,146% → Validation -23.7% FIXED")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # Test imports
    print(f"\n📦 IMPORT VALIDATION:")
    
    config_ok, configs = validate_config_import()
    utils_ok = validate_training_utils_import()
    
    if not (config_ok and utils_ok):
        print(f"\n❌ CRITICAL: Import failures detected. Cannot proceed with validation.")
        return 1
    
    # Detailed configuration validation
    issues, fixes = detailed_config_validation(configs)
    
    # Run programmatic validation
    print(f"\n🔍 PROGRAMMATIC VALIDATION:")
    try:
        from utils.regularized_training_utils import validate_regularization_implementation
        validation_results = validate_regularization_implementation()
        
        print(f"\nValidation Results:")
        for component, status in validation_results.items():
            if component != 'overall_status':
                icon = "✅" if status else "❌"
                print(f"   {icon} {component.replace('_', ' ').title()}: {'PASSED' if status else 'FAILED'}")
        
        overall_status = validation_results['overall_status']
        
    except Exception as e:
        print(f"❌ Programmatic validation failed: {e}")
        overall_status = 'ERROR'
    
    # Final assessment
    print(f"\n{'='*70}")
    print(f"🎯 FINAL ASSESSMENT:")
    print(f"{'='*70}")
    
    if overall_status == 'PASSED' and not issues:
        print(f"✅ VALIDATION PASSED - All overfitting fixes implemented correctly!")
        print(f"\n🛡️ Implemented Fixes:")
        for fix in fixes:
            print(f"   ✅ {fix}")
        
        print(f"\n🚀 READY FOR REGULARIZED TRAINING:")
        print(f"   1. Run: python train_ppo_regularized.py --seed 1007")
        print(f"   2. Monitor validation performance improvement")
        print(f"   3. Compare with previous overfitted results")
        print(f"   4. Proceed to production if validation positive")
        
        return 0
    else:
        print(f"❌ VALIDATION FAILED - Issues found that need fixing:")
        
        if issues:
            print(f"\n🔧 Issues to Fix:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        if fixes:
            print(f"\n✅ Fixes Already Implemented:")
            for fix in fixes:
                print(f"   ✅ {fix}")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Fix the issues listed above")
        print(f"   2. Re-run this validation script")
        print(f"   3. Proceed with training once all checks pass")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
