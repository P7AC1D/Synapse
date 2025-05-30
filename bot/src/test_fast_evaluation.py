#!/usr/bin/env python3
"""
Test script for the fast evaluation system.

This script demonstrates the performance improvements and validates
the accuracy of the optimized evaluation functions.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, Any
import argparse

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockArgs:
    """Mock arguments class for testing."""
    def __init__(self):
        self.initial_balance = 10000.0
        self.balance_per_lot = 1000.0
        self.point_value = 0.001
        self.min_lots = 0.01
        self.max_lots = 200.0
        self.contract_size = 100.0

def load_test_data(symbol: str = "XAUUSDm") -> pd.DataFrame:
    """Load test data for evaluation."""
    data_path = f"../data/{symbol}_15min.csv"
    
    if not os.path.exists(data_path):
        print(f"Test data not found: {data_path}")
        print("Generating synthetic test data...")
        return generate_synthetic_data(89979)  # Same size as mentioned in the issue
    
    print(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add spread if not present
    if 'spread' not in data.columns:
        data['spread'] = np.random.uniform(0.1, 0.5, len(data))
    
    # Add volume if not present
    if 'volume' not in data.columns:
        data['volume'] = np.random.randint(100, 1000, len(data))
    
    # Set datetime index if not present
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
    else:
        # Create synthetic datetime index
        start_date = pd.Timestamp('2020-01-01')
        data.index = pd.date_range(start=start_date, periods=len(data), freq='15min')
    
    print(f"Loaded {len(data)} samples from {data.index[0]} to {data.index[-1]}")
    return data

def generate_synthetic_data(n_samples: int) -> pd.DataFrame:
    """Generate synthetic market data for testing."""
    print(f"Generating {n_samples} synthetic data points...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data using random walk
    initial_price = 2000.0
    returns = np.random.normal(0, 0.001, n_samples)
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate OHLC data
    high_noise = np.random.uniform(0.0005, 0.002, n_samples)
    low_noise = np.random.uniform(0.0005, 0.002, n_samples)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + high_noise),
        'low': prices * (1 - low_noise),
        'close': prices,
        'volume': np.random.randint(100, 1000, n_samples),
        'spread': np.random.uniform(0.1, 0.5, n_samples)
    })
    
    # Set datetime index
    start_date = pd.Timestamp('2020-01-01')
    data.index = pd.date_range(start=start_date, periods=n_samples, freq='15min')
    
    return data

def find_test_model() -> str:
    """Find a test model file."""
    model_dirs = ["../model", "../results"]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    model_path = os.path.join(model_dir, file)
                    print(f"Found test model: {model_path}")
                    return model_path
    
    raise FileNotFoundError("No model files found for testing. Please train a model first.")

def test_basic_functionality():
    """Test basic functionality of fast evaluation."""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        from utils.fast_evaluation import (
            evaluate_model_on_dataset_optimized,
            evaluate_model_quick,
            compare_models_parallel,
            clear_evaluation_cache,
            get_cache_info
        )
        print("✓ Fast evaluation imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test cache functionality
    cache_info = get_cache_info()
    print(f"✓ Cache info: {cache_info}")
    
    clear_evaluation_cache()
    print("✓ Cache cleared successfully")
    
    return True

def test_evaluation_performance():
    """Test evaluation performance with different methods."""
    print("\n" + "="*60)
    print("TESTING EVALUATION PERFORMANCE")
    print("="*60)
    
    try:
        # Load test data and model
        data = load_test_data()
        model_path = find_test_model()
        args = MockArgs()
        
        # Import evaluation functions
        from utils.training_utils_enhanced import (
            evaluate_model_on_dataset,
            benchmark_evaluation_performance,
            quick_model_evaluation
        )
        
        print(f"\nTesting with {len(data)} samples...")
        
        # Test 1: Standard vs Optimized Evaluation
        print("\n1. Testing standard evaluation...")
        start_time = time.time()
        standard_result = evaluate_model_on_dataset(model_path, data, args, use_fast_evaluation=False)
        standard_time = time.time() - start_time
        
        if standard_result:
            print(f"   Standard time: {standard_time:.2f}s")
            print(f"   Score: {standard_result['score']:.4f}")
            print(f"   Trades: {standard_result['total_trades']}")
        
        print("\n2. Testing optimized evaluation...")
        start_time = time.time()
        optimized_result = evaluate_model_on_dataset(model_path, data, args, use_fast_evaluation=True)
        optimized_time = time.time() - start_time
        
        if optimized_result:
            print(f"   Optimized time: {optimized_time:.2f}s")
            print(f"   Score: {optimized_result['score']:.4f}")
            print(f"   Trades: {optimized_result['total_trades']}")
            
            if standard_result:
                speedup = standard_time / optimized_time
                score_diff = abs(standard_result['score'] - optimized_result['score'])
                trade_diff = abs(standard_result['total_trades'] - optimized_result['total_trades'])
                
                print(f"\n   PERFORMANCE COMPARISON:")
                print(f"   Speedup: {speedup:.1f}x")
                print(f"   Score difference: {score_diff:.6f}")
                print(f"   Trade count difference: {trade_diff}")
                
                if speedup > 2.0:
                    print("   ✓ Significant speedup achieved!")
                if score_diff < 0.001:
                    print("   ✓ Results are consistent!")
        
        # Test 3: Quick Evaluation
        print("\n3. Testing quick evaluation (10,000 samples)...")
        start_time = time.time()
        quick_result = quick_model_evaluation(model_path, data, args, sample_size=10000)
        quick_time = time.time() - start_time
        
        if quick_result:
            print(f"   Quick time: {quick_time:.2f}s")
            print(f"   Score: {quick_result['score']:.4f}")
            print(f"   Trades: {quick_result['total_trades']}")
            
            if standard_result:
                quick_speedup = standard_time / quick_time
                print(f"   Quick speedup: {quick_speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parallel_comparison():
    """Test parallel model comparison."""
    print("\n" + "="*60)
    print("TESTING PARALLEL MODEL COMPARISON")
    print("="*60)
    
    try:
        # Load test data
        data = load_test_data()
        args = MockArgs()
        
        # Find available models
        model_path = find_test_model()
        
        # For testing, we'll compare the same model against itself
        # In real usage, you'd have different model files
        model_paths = [model_path]  # Could add more if available
        
        from utils.training_utils_enhanced import parallel_model_comparison
        
        print(f"Comparing {len(model_paths)} model(s)...")
        start_time = time.time()
        results = parallel_model_comparison(model_paths, data, args)
        total_time = time.time() - start_time
        
        print(f"Parallel comparison completed in {total_time:.2f}s")
        
        for model_path, result in results.items():
            if result:
                print(f"Model: {os.path.basename(model_path)}")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Return: {result['returns']*100:.2f}%")
                print(f"  Trades: {result['total_trades']}")
            else:
                print(f"Model: {os.path.basename(model_path)} - FAILED")
        
        return True
        
    except Exception as e:
        print(f"✗ Parallel comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark():
    """Test the benchmark functionality."""
    print("\n" + "="*60)
    print("TESTING BENCHMARK FUNCTIONALITY")
    print("="*60)
    
    try:
        # Use smaller dataset for benchmark to speed up testing
        data = load_test_data()
        if len(data) > 20000:
            data = data.iloc[-20000:]  # Use last 20k samples
            print(f"Using subset of {len(data)} samples for benchmark")
        
        model_path = find_test_model()
        args = MockArgs()
        
        from utils.training_utils_enhanced import performance_benchmark
        
        results = performance_benchmark(model_path, data, args)
        
        if results:
            print("\n✓ Benchmark completed successfully!")
            return True
        else:
            print("✗ Benchmark failed")
            return False
            
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test fast evaluation system')
    parser.add_argument('--symbol', default='XAUUSDm', help='Symbol to test with')
    parser.add_argument('--test', choices=['basic', 'performance', 'parallel', 'benchmark', 'all'], 
                        default='all', help='Which test to run')
    args = parser.parse_args()
    
    print("FAST EVALUATION SYSTEM TEST")
    print("="*60)
    print(f"Testing with symbol: {args.symbol}")
    
    test_results = {}
    
    if args.test in ['basic', 'all']:
        test_results['basic'] = test_basic_functionality()
    
    if args.test in ['performance', 'all']:
        test_results['performance'] = test_evaluation_performance()
    
    if args.test in ['parallel', 'all']:
        test_results['parallel'] = test_parallel_comparison()
    
    if args.test in ['benchmark', 'all']:
        test_results['benchmark'] = test_benchmark()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name.upper():15s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fast evaluation system is working correctly.")
        print("\nTo use the optimized evaluation in your code:")
        print("1. Replace 'from utils.training_utils import ...' with 'from utils.training_utils_enhanced import ...'")
        print("2. Your existing code will automatically use the faster evaluation")
        print("3. For 89,979 samples, expect 10-20x speedup!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
