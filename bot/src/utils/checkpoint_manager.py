#!/usr/bin/env python3
"""
Checkpoint Management Utility for Walk-Forward Optimization Training

This utility provides command-line tools for managing checkpoint models:
- List all available checkpoints
- Analyze performance trends across iterations
- Load specific checkpoint models
- Clean up old checkpoints to save disk space
- Restore training from specific checkpoints

Usage:
    python checkpoint_manager.py --results-dir ../results/1002 --command list
    python checkpoint_manager.py --results-dir ../results/1002 --command analyze
    python checkpoint_manager.py --results-dir ../results/1002 --command cleanup --keep-every 10
    python checkpoint_manager.py --results-dir ../results/1002 --command load --iteration 25
"""

import argparse
import os
import sys
import json
from typing import Optional

# Add src to path to import training utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Checkpoint Management for WFO Training')
    parser.add_argument('--results-dir', required=True, help='Path to results directory (e.g., ../results/1002)')
    parser.add_argument('--command', required=True, choices=['list', 'analyze', 'cleanup', 'load', 'summary'],
                        help='Command to execute')
    parser.add_argument('--iteration', type=int, help='Specific iteration for load command')
    parser.add_argument('--keep-every', type=int, default=10, help='Keep every Nth checkpoint during cleanup')
    parser.add_argument('--keep-last', type=int, default=5, help='Keep last N checkpoints during cleanup')
    parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    args = parser.parse_args()
    
    # Import checkpoint utilities
    try:
        # First try regularized utilities
        try:
            from regularized_training_utils import (
                list_checkpoints, analyze_checkpoint_performance,
                cleanup_checkpoints, load_checkpoint_model, create_checkpoint_summary
            )
            print("✅ Using regularized training utilities")
        except ImportError:
            # Fall back to non-regularized utilities
            from training_utils_no_early_stopping import (
                list_checkpoints, analyze_checkpoint_performance,
                cleanup_checkpoints, load_checkpoint_model, create_checkpoint_summary
            )
            print("⚠️ Using non-regularized training utilities")
    except ImportError as e:
        print(f"❌ Error importing checkpoint utilities: {e}")
        print("Make sure you're running this from the src/utils directory")
        return 1
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        return 1
    
    try:
        if args.command == 'list':
            list_command(args.results_dir, args.format)
        elif args.command == 'analyze':
            analyze_command(args.results_dir, args.format)
        elif args.command == 'cleanup':
            cleanup_command(args.results_dir, args.keep_every, args.keep_last)
        elif args.command == 'load':
            load_command(args.results_dir, args.iteration)
        elif args.command == 'summary':
            summary_command(args.results_dir)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return 1

def list_command(results_dir: str, format_type: str):
    """List all available checkpoints"""
    try:
        from regularized_training_utils import list_checkpoints
    except ImportError:
        from training_utils_no_early_stopping import list_checkpoints
    
    checkpoints = list_checkpoints(results_dir)
    
    if not checkpoints:
        print("📝 No checkpoints found")
        return
    
    if format_type == 'json':
        print(json.dumps(checkpoints, indent=2))
    else:
        print(f"\n📂 Found {len(checkpoints)} checkpoints in {results_dir}")
        print(f"{'Iteration':<10} {'Size (MB)':<10} {'Model Path':<50}")
        print("-" * 80)
        
        for cp in checkpoints:
            size_str = f"{cp['file_size_mb']:.1f}"
            print(f"{cp['iteration']:<10} {size_str:<10} {os.path.basename(cp['model_path']):<50}")

def analyze_command(results_dir: str, format_type: str):
    """Analyze performance trends across checkpoints"""
    try:
        from regularized_training_utils import analyze_checkpoint_performance
    except ImportError:
        from training_utils_no_early_stopping import analyze_checkpoint_performance
    
    analysis = analyze_checkpoint_performance(results_dir)
    
    if format_type == 'json':
        print(json.dumps(analysis, indent=2))
    else:
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print(f"\n📊 Performance Analysis for {results_dir}")
        print("=" * 60)
        print(f"Total checkpoints: {analysis['total_checkpoints']}")
        print(f"Iterations analyzed: {analysis['iterations_analyzed']}")
        print(f"Iteration range: {analysis['iteration_range']}")
        
        print(f"\n📈 Validation Performance:")
        val_perf = analysis['validation_performance']
        print(f"  Mean: {val_perf['mean']:.2f}%")
        print(f"  Std:  {val_perf['std']:.2f}%")
        print(f"  Range: {val_perf['min']:.2f}% to {val_perf['max']:.2f}%")
        print(f"  Latest: {val_perf['latest']:.2f}%")
        
        print(f"\n📈 Combined Performance:")
        comb_perf = analysis['combined_performance']
        print(f"  Mean: {comb_perf['mean']:.2f}%")
        print(f"  Std:  {comb_perf['std']:.2f}%")
        print(f"  Range: {comb_perf['min']:.2f}% to {comb_perf['max']:.2f}%")
        print(f"  Latest: {comb_perf['latest']:.2f}%")
        
        print(f"\n🏆 Best Iterations:")
        best = analysis['best_iteration']
        if best['validation'] is not None:
            print(f"  Validation: {best['validation']} ({val_perf['max']:.2f}%)")
        if best['combined'] is not None:
            print(f"  Combined: {best['combined']} ({comb_perf['max']:.2f}%)")

def cleanup_command(results_dir: str, keep_every: int, keep_last: int):
    """Clean up old checkpoints"""
    try:
        from regularized_training_utils import cleanup_checkpoints, list_checkpoints
    except ImportError:
        from training_utils_no_early_stopping import cleanup_checkpoints, list_checkpoints
    
    checkpoints_before = list_checkpoints(results_dir)
    removed = cleanup_checkpoints(results_dir, keep_every, keep_last)
    checkpoints_after = list_checkpoints(results_dir)
    
    print(f"\n🧹 Checkpoint Cleanup Complete")
    print(f"Checkpoints before: {len(checkpoints_before)}")
    print(f"Checkpoints removed: {removed}")
    print(f"Checkpoints remaining: {len(checkpoints_after)}")
    print(f"Strategy: Keep every {keep_every}th + last {keep_last} iterations")

def load_command(results_dir: str, iteration: Optional[int]):
    """Load a specific checkpoint model and restore its state"""
    try:
        from regularized_training_utils import load_checkpoint_model, RegularizedTrainingManager
        is_regularized = True
    except ImportError:
        from training_utils_no_early_stopping import load_checkpoint_model
        is_regularized = False

    if iteration is None:
        print("❌ Please specify --iteration for load command")
        return
    
    # Load the model
    model = load_checkpoint_model(results_dir, iteration)
    if model is None:
        print(f"❌ Failed to load checkpoint from iteration {iteration}")
        return
    
    # Restore validation state for regularized models
    if is_regularized:
        validation_state_path = os.path.join(results_dir, "validation_state.json")
        if os.path.exists(validation_state_path):
            try:
                with open(validation_state_path, 'r') as f:
                    validation_state = json.load(f)
                print("\n📊 Restored validation state:")
                print(f"Best validation score: {validation_state.get('best_validation_score', -float('inf'))*100:.2f}%")
                print(f"Iteration: {validation_state.get('iteration', 'N/A')}")
                print(f"No improvement count: {validation_state.get('no_improvement_count', 0)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not restore validation state - {e}")
    
    print(f"\n✅ Successfully loaded checkpoint from iteration {iteration}")
    print(f"Model type: {type(model).__name__}")
    try:
        print(f"Device: {model.device}")
    except:
        print(f"Device: Not available")
    
    # Load training state
    training_state_path = os.path.join(results_dir, "regularized_training_state.json")
    if os.path.exists(training_state_path):
        try:
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            print("\n📈 Training state loaded:")
            print(f"Best model path: {training_state.get('best_model_path', 'N/A')}")
            print(f"Training metrics:")
            metrics = training_state.get('training_metrics', {})
            print(f"  - Iterations completed: {metrics.get('iterations_completed', 0)}")
            print(f"  - Best validation score: {metrics.get('best_validation_score', -float('inf'))*100:.2f}%")
            print(f"  - Validation improvements: {metrics.get('validation_improvements', 0)}")
            print(f"  - Early stops triggered: {metrics.get('early_stops_triggered', 0)}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load training state - {e}")

def summary_command(results_dir: str):
    """Create comprehensive checkpoint summary"""
    try:
        from regularized_training_utils import create_checkpoint_summary
    except ImportError:
        from training_utils_no_early_stopping import create_checkpoint_summary
    
    create_checkpoint_summary(results_dir)
    print(f"✅ Checkpoint summary created for {results_dir}")

if __name__ == "__main__":
    exit(main())
