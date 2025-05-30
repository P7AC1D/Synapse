"""
Fixed version of fast evaluation with proper sequential processing for GPU deadlock prevention.
"""

import os
import json
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.ppo_recurrent import RecurrentPPO
from trading.environment import TradingEnv
from trading.enhanced_features import EnhancedFeatureProcessor
from trading.metrics import MetricsTracker
from utils.progress import show_progress_continuous, stop_progress_indicator

def compare_models_parallel_fixed(model_paths: List[str], data: pd.DataFrame, args,
                                max_workers: int = None,
                                batch_size: int = 1000) -> Dict[str, Dict[str, Any]]:
    """
    FIXED: Compare multiple models with proper sequential processing for 1-2 models.
    
    Args:
        model_paths: List of model file paths
        data: Dataset for evaluation
        args: Training arguments
        max_workers: Number of parallel processes (default: CPU count)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary mapping model paths to evaluation results
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(model_paths))
    
    print(f"Evaluating {len(model_paths)} models using {max_workers} parallel processes...")
    start_time = time.time()
    
    results = {}
    
    # SAFETY FIX: Use sequential processing for 1-2 models to avoid GPU deadlock
    if len(model_paths) <= 2:
        print(f"🔧 Using SEQUENTIAL processing to avoid GPU deadlock (models: {len(model_paths)})")
        
        # Import the optimized evaluation function
        from utils.fast_evaluation import evaluate_model_on_dataset_optimized
        
        for i, model_path in enumerate(model_paths):
            print(f"📊 [{i+1}/{len(model_paths)}] Evaluating {os.path.basename(model_path)}...")
            result = evaluate_model_on_dataset_optimized(model_path, data, args, batch_size, show_progress=True)
            results[model_path] = result
            
            if result:
                print(f"✓ Score: {result['score']:.4f}, Time: {result.get('evaluation_time', 0):.1f}s")
            else:
                print(f"✗ Evaluation failed")
        
        total_time = time.time() - start_time
        print(f"\nSequential evaluation completed in {total_time:.2f} seconds")
        return results
    
    # Use parallel processing for 3+ models
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Import the optimized evaluation function
        from utils.fast_evaluation import evaluate_model_on_dataset_optimized
        
        # Submit all evaluation tasks
        future_to_model = {
            executor.submit(
                evaluate_model_on_dataset_optimized, 
                model_path, data, args, batch_size, True, False  # use_cache=True, show_progress=False
            ): model_path
            for model_path in model_paths
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_model):
            model_path = future_to_model[future]
            try:
                result = future.result()
                results[model_path] = result
                completed += 1
                
                if result:
                    print(f"✓ [{completed}/{len(model_paths)}] {os.path.basename(model_path)}: "
                          f"Score={result['score']:.4f}, Time={result.get('evaluation_time', 0):.1f}s")
                else:
                    print(f"✗ [{completed}/{len(model_paths)}] {os.path.basename(model_path)}: Failed")
                    
            except Exception as e:
                print(f"✗ [{completed}/{len(model_paths)}] {os.path.basename(model_path)}: Error - {e}")
                results[model_path] = None
                completed += 1
    
    total_time = time.time() - start_time
    print(f"\nParallel evaluation completed in {total_time:.2f} seconds")
    print(f"Average time per model: {total_time/len(model_paths):.2f} seconds")
    
    return results
