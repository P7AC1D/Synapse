#!/usr/bin/env python3

"""
Test TorchScript model against original model to ensure identical results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path
from sb3_contrib.ppo_recurrent import RecurrentPPO

def setup_python_path():
    """Add bot/src to Python path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bot_src_dir = os.path.abspath(os.path.join(script_dir, "../../bot/src"))
    if not os.path.exists(bot_src_dir):
        raise RuntimeError(f"Bot source directory not found at {bot_src_dir}")
    sys.path.append(bot_src_dir)
    print(f"Added {bot_src_dir} to Python path")

def verify_imports():
    """Verify all required modules can be imported."""
    try:
        from trading.environment import TradingEnv
        from trading.features import FeatureProcessor
        print("Successfully loaded trading modules")
        return TradingEnv, FeatureProcessor
    except ImportError as e:
        raise RuntimeError(f"Failed to import trading modules: {e}")

def load_original_model(model_path: str) -> RecurrentPPO:
    """Load the original PPO-LSTM model."""
    print(f"Loading original model from {model_path}")
    try:
        model = RecurrentPPO.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load original model: {e}")

def load_torchscript_model(model_path: str) -> torch.jit.ScriptModule:
    """Load the exported TorchScript model."""
    print(f"Loading TorchScript model from {model_path}")
    try:
        return torch.jit.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load TorchScript model: {e}")

def load_config(config_path: str) -> dict:
    """Load model configuration from JSON file."""
    print(f"Loading config from {config_path}")
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

def generate_test_data() -> pd.DataFrame:
    """Generate synthetic test data for model testing with realistic patterns."""
    np.random.seed(42)
    n_samples = 500  # More samples to account for preprocessing losses
    
    # Generate base price with a realistic trend
    base = 1000
    noise = np.random.normal(0, 5, n_samples)
    trend = np.cumsum(np.random.normal(0, 1, n_samples)) * 0.5  # Add a random walk trend
    close = base + trend + noise
    
    # Generate realistic OHLC data
    high_offsets = np.abs(np.random.normal(0, 10, n_samples))
    low_offsets = np.abs(np.random.normal(0, 10, n_samples))
    
    data = pd.DataFrame({
        'close': close,
        'open': close + np.random.normal(0, 5, n_samples),
        'high': close + high_offsets,
        'low': close - low_offsets,
        'volume': np.abs(np.random.normal(1000, 100, n_samples) + np.random.exponential(500, n_samples)),
        'spread': np.abs(np.random.normal(2, 0.1, n_samples))
    })
    
    # Ensure OHLC relationship is valid
    data['high'] = np.maximum.reduce([data['high'], data['open'], data['close']])
    data['low'] = np.minimum.reduce([data['low'], data['open'], data['close']])
    
    # Set datetime index with consistent spacing
    data.index = pd.date_range(start='2025-01-01', periods=n_samples, freq='15min')
    
    return data

def main():
    # Setup Python path and verify imports
    setup_python_path()
    TradingEnv, FeatureProcessor = verify_imports()
    
    parser = argparse.ArgumentParser(description="Test LibTorch model implementation")
    parser.add_argument("--original-model", required=True, help="Path to original .zip model")
    parser.add_argument("--torchscript-model", required=True, help="Path to exported .pt model")
    parser.add_argument("--config", required=True, help="Path to model_config.json")
    parser.add_argument("--match-threshold", type=float, default=0.7, 
                       help="Threshold for action matching rate to consider test successful (default: 0.7)")
    args = parser.parse_args()
    
    # Load models and config
    original_model = load_original_model(args.original_model)
    torchscript_model = load_torchscript_model(args.torchscript_model)
    config = load_config(args.config)
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data()
    
    # Create environment and process features
    env = TradingEnv(data=data, initial_balance=10000)
    features_df, _ = env.feature_processor.preprocess_data(data)
    
    # Initialize states
    hidden_size = config['hidden_size']
    num_layers = 2  # SB3 RecurrentPPO uses 2 LSTM layers
    
    # Initialize with correct dimensions for TorchScript model
    zero_hidden = torch.zeros(num_layers, 1, hidden_size)
    zero_cell = torch.zeros(num_layers, 1, hidden_size)
    lstm_states = (zero_hidden, zero_cell)  # Use PyTorch tensors from the start
    
    # Print model dimensions for debugging
    feature_count = config['feature_count']
    print(f"Model expects {feature_count} features")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
    
    # Test sequence
    print("\nTesting inference...")
    
    # Track metrics
    total_steps = 0
    matching_actions = 0
    
    # Run several inference steps to compare outputs
    max_steps = min(20, len(features_df))  # Test with more steps
    for i in range(max_steps):
        try:
            # Get features
            features = features_df.iloc[i].values
            features = np.append(features, [0, 0])  # Add position features
            
            # Original model prediction
            obs = torch.FloatTensor(features).reshape(1, -1)
            
            # Print original model's hidden state dimensions on first step
            if i == 0:
                print(f"Original model observation shape: {obs.shape}")
                print(f"Original LSTM hidden state shape: {lstm_states[0].shape}")
                print(f"Original LSTM cell state shape: {lstm_states[1].shape}")
                
            with torch.no_grad():
                original_action, lstm_states_np = original_model.predict(
                    obs,
                    state=lstm_states,
                    deterministic=True
                )
                
                # Ensure states are PyTorch tensors
                if isinstance(lstm_states_np[0], np.ndarray):
                    hidden_state = torch.FloatTensor(lstm_states_np[0])
                    cell_state = torch.FloatTensor(lstm_states_np[1])
                    lstm_states = (hidden_state, cell_state)
                else:
                    lstm_states = lstm_states_np
                
            # TorchScript model expects different tensor shapes - add sequence dimension
            features_tensor = torch.FloatTensor(features).reshape(1, 1, -1)
            
            if i == 0:
                print(f"TorchScript input shape: {features_tensor.shape}")
                print(f"Prepared hidden state shape: {lstm_states[0].shape}")
                print(f"Prepared cell state shape: {lstm_states[1].shape}")
            
            # Get TorchScript prediction
            with torch.no_grad():
                torchscript_probs, new_hidden, new_cell = torchscript_model(
                    features_tensor,
                    lstm_states[0],
                    lstm_states[1]
                )
                
            # Get action from probabilities
            ts_action = torch.argmax(torchscript_probs).item()
            orig_action = original_action.item() if isinstance(original_action, torch.Tensor) else original_action
            
            # Print comparison results
            print(f"\nStep {i}:")
            print(f"Original model action: {orig_action}")
            print(f"TorchScript model action: {ts_action}")
            
            # Track matches vs mismatches
            total_steps += 1
            if orig_action == ts_action:
                matching_actions += 1
                print("✓ Actions match")
            else:
                print("✗ Actions don't match")
                print(f"TorchScript probabilities: {torchscript_probs}")
            
            # Use TorchScript model's state for next iteration
            lstm_states = (new_hidden, new_cell)
            
        except Exception as e:
            print(f"Error during inference step {i}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Calculate and report match rate
    match_rate = matching_actions / total_steps if total_steps > 0 else 0
    print(f"\nTest Summary:")
    print(f"Total steps tested: {total_steps}")
    print(f"Matching actions: {matching_actions}")
    print(f"Match rate: {match_rate:.2%}")
    
    # Determine overall success based on threshold
    success = match_rate >= args.match_threshold
    
    if success:
        print(f"\nTest PASSED! Match rate {match_rate:.2%} meets or exceeds threshold of {args.match_threshold:.2%}")
        return 0
    else:
        print(f"\nTest FAILED. Match rate {match_rate:.2%} below threshold of {args.match_threshold:.2%}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
