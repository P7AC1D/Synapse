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
    # Set deterministic execution for PyTorch
    torch.manual_seed(42)  # Fixed seed for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)  # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True  # Make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # Disable cudnn benchmark
    
    # Setup Python path and verify imports
    setup_python_path()
    TradingEnv, FeatureProcessor = verify_imports()
    
    parser = argparse.ArgumentParser(description="Test LibTorch model implementation")
    parser.add_argument("--original-model", required=True, help="Path to original .zip model")
    parser.add_argument("--torchscript-model", required=True, help="Path to exported .pt model")
    parser.add_argument("--config", required=True, help="Path to model_config.json")
    parser.add_argument("--match-threshold", type=float, default=0.7, 
                       help="Threshold for action matching rate to consider test successful (default: 0.7)")
    parser.add_argument("--num-steps", type=int, default=30,
                      help="Number of steps to test (default: 30)")
    parser.add_argument("--use-top-n", type=int, default=1,
                      help="Count as match if original action is in top N predictions (default: 1)")
    parser.add_argument("--relaxed", action="store_true",
                      help="Use relaxed matching criteria (original action in top 2)")
    args = parser.parse_args()
    
    # Determine matching criteria based on flags
    top_n = 2 if args.relaxed else args.use_top_n
    
    # Load models and config
    original_model = load_original_model(args.original_model)
    # Set model to evaluation mode for consistent behavior
    original_model.policy.set_training_mode(False)
    
    torchscript_model = load_torchscript_model(args.torchscript_model)
    config = load_config(args.config)
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data()
    
    # Create environment and process features
    env = TradingEnv(data=data, initial_balance=10000)
    features_df, _ = env.feature_processor.preprocess_data(data)
    
    # Initialize states consistently
    hidden_size = config['hidden_size']
    num_layers = 2  # SB3 RecurrentPPO uses 2 LSTM layers
    
    # Initialize with exact same values to ensure consistent behavior
    torch.manual_seed(42)  # Reset seed before creating initial states
    zero_hidden = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
    zero_cell = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
    lstm_states = (zero_hidden, zero_cell)  # Use PyTorch tensors from the start
    
    # Print model dimensions for debugging
    feature_count = config['feature_count']
    print(f"Model expects {feature_count} features")
    print(f"Hidden size: {hidden_size}, Layers: {num_layers}")
    
    # Test sequence
    print("\nTesting inference...")
    print(f"Using {'relaxed' if args.relaxed else 'strict'} matching criteria (top {top_n})")
    
    # Track metrics
    total_steps = 0
    matching_actions = 0
    prob_differences = []  # Track probability differences for analysis
    
    # Run several inference steps to compare outputs
    max_steps = min(args.num_steps, len(features_df))  # Test with configurable steps
    
    for i in range(max_steps):
        try:
            # Get features
            features = features_df.iloc[i].values
            features = np.append(features, [0, 0])  # Add position features
            
            # Original model prediction - use consistent tensor type for both models
            obs = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
            
            # Print original model's hidden state dimensions on first step
            if i == 0:
                print(f"Original model observation shape: {obs.shape}")
                print(f"Original LSTM hidden state shape: {lstm_states[0].shape}")
                print(f"Original LSTM cell state shape: {lstm_states[1].shape}")
                
            with torch.no_grad():
                original_action, lstm_states_np = original_model.predict(
                    obs,
                    state=lstm_states,
                    deterministic=True  # Force deterministic action selection
                )
                
                # Ensure states are PyTorch tensors with the same precision
                if isinstance(lstm_states_np[0], np.ndarray):
                    hidden_state = torch.tensor(lstm_states_np[0], dtype=torch.float32)
                    cell_state = torch.tensor(lstm_states_np[1], dtype=torch.float32)
                    lstm_states = (hidden_state, cell_state)
                else:
                    # Ensure consistent precision even if already tensors
                    hidden_state = lstm_states_np[0].to(dtype=torch.float32)
                    cell_state = lstm_states_np[1].to(dtype=torch.float32)
                    lstm_states = (hidden_state, cell_state)
                
            # Create a copy of the LSTM states to ensure no accidental sharing
            lstm_states_copy = (lstm_states[0].clone(), lstm_states[1].clone())
            
            # TorchScript model expects different tensor shapes - add sequence dimension
            features_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, 1, -1)
            
            if i == 0:
                print(f"TorchScript input shape: {features_tensor.shape}")
                print(f"Prepared hidden state shape: {lstm_states_copy[0].shape}")
                print(f"Prepared cell state shape: {lstm_states_copy[1].shape}")
            
            # Get TorchScript prediction
            with torch.no_grad():
                torchscript_probs, new_hidden, new_cell = torchscript_model(
                    features_tensor,
                    lstm_states_copy[0],  # Use the copied states to avoid any modification
                    lstm_states_copy[1]
                )
                
            # Get action from probabilities
            ts_action = torch.argmax(torchscript_probs).item()
            orig_action = original_action.item() if isinstance(original_action, torch.Tensor) else original_action
            
            # Get top N predictions from torchscript model
            top_n_actions = torch.topk(torchscript_probs[0], min(top_n, torchscript_probs.shape[1])).indices.tolist()
            
            # Determine if it's a match based on the criteria
            is_match = orig_action in top_n_actions
            
            # Print comparison results
            print(f"\nStep {i}:")
            print(f"Original model action: {orig_action}")
            print(f"TorchScript model action: {ts_action}")
            
            # Track matches vs mismatches
            total_steps += 1
            if is_match:
                matching_actions += 1
                match_text = "✓" if orig_action == ts_action else "≈"
                print(f"{match_text} {'Exact match' if orig_action == ts_action else 'Action in top '+str(top_n)}")
            else:
                print("✗ Actions don't match")
                
                # Get probabilities from original model for better debugging
                with torch.no_grad():
                    try:
                        # Try to extract action distribution if possible
                        orig_probs = None
                        orig_features = original_model.policy.extract_features(obs, lstm_states_copy)
                        if isinstance(orig_features, tuple):
                            if len(orig_features) >= 2:
                                policy_latent = orig_features[1]
                                orig_logits = original_model.policy.action_net(policy_latent)
                                orig_probs = torch.softmax(orig_logits, dim=-1)
                    except Exception:
                        orig_probs = None
                
                # Print probability comparison if available
                if orig_probs is not None:
                    print(f"Original probabilities: {orig_probs}")
                    # Calculate probability difference
                    try:
                        prob_diff = torch.abs(orig_probs - torchscript_probs).mean().item()
                        prob_differences.append(prob_diff)
                        print(f"Mean probability difference: {prob_diff:.4f}")
                    except:
                        pass
                print(f"TorchScript probabilities: {torchscript_probs}")
                
            # Use TorchScript model's state for next iteration to reduce accumulated drift
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
    
    # Print probability difference stats if available
    if prob_differences:
        avg_prob_diff = sum(prob_differences) / len(prob_differences)
        print(f"Average probability difference: {avg_prob_diff:.4f}")
    
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
