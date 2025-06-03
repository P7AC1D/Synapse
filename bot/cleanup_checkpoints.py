#!/usr/bin/env python3
"""
Checkpoint Cleanup Script for DRL Trading Bot
Keeps only essential checkpoints to save disk space while preserving important models.
"""

import os
import shutil
from pathlib import Path

def cleanup_checkpoints(checkpoint_dir, keep_iterations=None):
    """
    Clean up checkpoint files, keeping only specified iterations.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        keep_iterations: List of iteration numbers to keep (None = keep latest, baseline, and key milestones)
    """
    
    checkpoint_path = Path(checkpoint_dir)
    
    if keep_iterations is None:
        # Default: Keep baseline (0), milestones (every 5th), and latest (22)
        keep_iterations = [0, 5, 10, 15, 20, 22]
    
    print(f"Checkpoint cleanup for: {checkpoint_path}")
    print(f"Keeping iterations: {keep_iterations}")
    
    # Create backup directory
    backup_dir = checkpoint_path.parent / "checkpoints_backup"
    backup_dir.mkdir(exist_ok=True)
    
    total_saved = 0
    kept_files = []
    removed_files = []
    
    # Process all files in checkpoint directory
    for file_path in checkpoint_path.glob("*"):
        if file_path.is_file():
            filename = file_path.name
            
            # Extract iteration number from filename
            if filename.startswith("model_iter_") and filename.endswith(".zip"):
                try:
                    iter_num = int(filename.replace("model_iter_", "").replace(".zip", ""))
                    
                    if iter_num in keep_iterations:
                        kept_files.append(filename)
                        print(f"✓ Keeping: {filename}")
                    else:
                        # Move to backup instead of deleting (safer)
                        backup_path = backup_dir / filename
                        shutil.move(str(file_path), str(backup_path))
                        removed_files.append(filename)
                        total_saved += file_path.stat().st_size if file_path.exists() else 0
                        print(f"📦 Moved to backup: {filename}")
                        
                except ValueError:
                    print(f"⚠️  Skipping invalid filename: {filename}")
                    
            elif filename.startswith("no_model_iter_"):
                # Keep these - they show when anti-collapse system worked
                kept_files.append(filename)
                print(f"✓ Keeping no-model indicator: {filename}")
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"   Files kept: {len(kept_files)}")
    print(f"   Files moved to backup: {len(removed_files)}")
    print(f"   Space saved: {total_saved / (1024**3):.2f} GB")
    print(f"   Backup location: {backup_dir}")
    
    return kept_files, removed_files

if __name__ == "__main__":
    # Default cleanup - keeps key checkpoints
    checkpoint_dir = "c:/Dev/drl/bot/results/1006/checkpoints"
    
    print("🧹 DRL Trading Bot Checkpoint Cleanup")
    print("=" * 50)
    
    # Option 1: Conservative cleanup (keeps more models)
    conservative_keep = [0, 2, 5, 8, 10, 12, 15, 18, 20, 21, 22]
    
    # Option 2: Aggressive cleanup (keeps only essential)
    aggressive_keep = [0, 10, 20, 22]
    
    print("\nChoose cleanup strategy:")
    print("1. Conservative (keep ~11 models)")
    print("2. Aggressive (keep ~4 models)")
    print("3. Custom (specify iterations)")
    print("4. Cancel")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        cleanup_checkpoints(checkpoint_dir, conservative_keep)
    elif choice == "2":
        cleanup_checkpoints(checkpoint_dir, aggressive_keep)
    elif choice == "3":
        custom_iterations = input("Enter iteration numbers to keep (comma-separated): ")
        try:
            keep_list = [int(x.strip()) for x in custom_iterations.split(",")]
            cleanup_checkpoints(checkpoint_dir, keep_list)
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
    else:
        print("✅ Cleanup cancelled.")
