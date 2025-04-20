#!/usr/bin/env python3

"""
Export RecurrentPPO model to TorchScript format with feature preprocessing parameters.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sb3_contrib.ppo_recurrent import RecurrentPPO
import torch
from typing import Any, Dict, List, Union

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

def convert_to_native_types(value: Any) -> Union[float, int, List, Dict]:
    """Convert numpy/torch types to Python native types."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, (list, tuple)):
        return [convert_to_native_types(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_to_native_types(v) for k, v in value.items()}
    elif isinstance(value, np.ndarray):
        return convert_to_native_types(value.tolist())
    return value

class LSTMWrapper(torch.nn.Module):
    """Wrapper class to make the LSTM model compatible with TorchScript.
    
    This wrapper handles the specific architecture of SB3's RecurrentPPO with:
    - 2 LSTM layers with hidden size 256
    - Policy MLP extractor [128, 64]
    - Action network from 64 dimensions to action space
    """
    
    def __init__(self, policy):
        super().__init__()
        # Extract essential components from the policy
        self.lstm = policy.lstm_actor
        self.mlp_extractor = policy.mlp_extractor.policy_net  # Policy network after LSTM
        self.action_net = policy.action_net
        self.hidden_size = policy.lstm_actor.hidden_size
        self.num_layers = policy.lstm_actor.num_layers
        
    def forward(self, features, hidden_state, cell_state):
        """
        Forward pass through the model.
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            hidden_state: LSTM hidden state [num_layers, batch_size, hidden_size]
            cell_state: LSTM cell state [num_layers, batch_size, hidden_size]
            
        Returns:
            tuple: (action_probs, new_hidden_state, new_cell_state)
        """
        # Run LSTM
        lstm_out, (new_hidden, new_cell) = self.lstm(
            features, (hidden_state, cell_state)
        )
        
        # Extract features from the LSTM output and reshape
        # The LSTM output is [batch_size, seq_len, hidden_size]
        # We need to flatten it to [batch_size * seq_len, hidden_size]
        lstm_features = lstm_out.view(-1, self.hidden_size)
        
        # Pass through policy network layers
        policy_features = self.mlp_extractor(lstm_features)
        
        # Get action probabilities
        action_logits = self.action_net(policy_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return action_probs, new_hidden, new_cell

def get_feature_names(FeatureProcessor) -> List[str]:
    """Get feature names directly from FeatureProcessor."""
    # Create feature processor instance
    feature_processor = FeatureProcessor()
    
    # Get feature names from the feature processor's own method
    feature_names = feature_processor.get_feature_names()
    
    return feature_names

def generate_dummy_data() -> pd.DataFrame:
    """Generate dummy data for feature names extraction."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base prices
    close = np.random.normal(1000, 10, n_samples)
    
    # Generate consistent OHLCV data
    data = pd.DataFrame({
        'close': close,
        'open': close + np.random.normal(0, 5, n_samples),
        'high': close + abs(np.random.normal(0, 10, n_samples)),
        'low': close - abs(np.random.normal(0, 10, n_samples)),
        'volume': abs(np.random.normal(1000, 100, n_samples)),
        'spread': abs(np.random.normal(2, 0.1, n_samples))
    })
    
    # Ensure OHLC relationship is valid
    data['high'] = np.maximum.reduce([data['high'], data['open'], data['close']])
    data['low'] = np.minimum.reduce([data['low'], data['open'], data['close']])
    
    # Set index as datetime
    data.index = pd.date_range(start='2025-01-01', periods=n_samples, freq='15min')
    
    return data

def validate_config(config: Dict) -> None:
    """Validate configuration before saving."""
    required_keys = [
        "feature_count",
        "hidden_size",
        "action_count",
        "feature_names"
    ]
    
    # Check for required keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")
    
    # Validate types and values
    if not isinstance(config["feature_count"], int):
        raise TypeError("feature_count must be an integer")
    if not isinstance(config["hidden_size"], int):
        raise TypeError("hidden_size must be an integer")
    if not isinstance(config["action_count"], int):
        raise TypeError("action_count must be an integer")
    
    # Validate lists
    if len(config["feature_names"]) != config["feature_count"]:
        raise ValueError("feature_names length doesn't match feature_count")

def main():
    # Setup Python path
    setup_python_path()
    
    # Verify imports
    TradingEnv, FeatureProcessor = verify_imports()
    
    parser = argparse.ArgumentParser(description="Export model to TorchScript format")
    parser.add_argument("--model-path", required=True, help="Path to the trained model")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    try:
        model = RecurrentPPO.load(args.model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Debug model architecture
    print("\nDebugging model architecture:")
    policy = model.policy
    print(f"LSTM Actor: {policy.lstm_actor}")
    print(f"Action Net: {policy.action_net}")
    
    # Check input/output dimensions
    print("\nAnalyzing network dimensions:")
    lstm_input_size = policy.lstm_actor.input_size
    lstm_hidden_size = policy.lstm_actor.hidden_size
    lstm_num_layers = policy.lstm_actor.num_layers
    print(f"LSTM Input Size: {lstm_input_size}")
    print(f"LSTM Hidden Size: {lstm_hidden_size}")
    print(f"LSTM Num Layers: {lstm_num_layers}")
    
    # Get action network dimensions
    action_net_in_features = policy.action_net.in_features
    action_net_out_features = policy.action_net.out_features
    print(f"Action Net Input Features: {action_net_in_features}")
    print(f"Action Net Output Features: {action_net_out_features}")
    
    # Create LSTM wrapper
    print("\nConverting to TorchScript...")
    try:
        lstm_wrapper = LSTMWrapper(model.policy)
        lstm_wrapper.eval()
        scripted_module = torch.jit.script(lstm_wrapper)
    except Exception as e:
        raise RuntimeError(f"Failed to convert model to TorchScript: {e}")
    
    # Save TorchScript model
    model_path = output_dir / "model.pt"
    try:
        scripted_module.save(str(model_path))
        print(f"Saved TorchScript model to {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save TorchScript model: {e}")
    
    # Get feature names
    print("Getting feature names...")
    try:
        feature_names = get_feature_names(FeatureProcessor)
        
        # Verify feature count matches model expectations
        expected_feature_count = model.policy.observation_space.shape[0]
        if len(feature_names) != expected_feature_count:
            print(f"Warning: Feature count mismatch. Model expects {expected_feature_count} features, but FeatureProcessor provides {len(feature_names)}.")
    except Exception as e:
        raise RuntimeError(f"Failed to get feature names: {e}")
    
    # Create config with proper type conversion
    config = {
        "feature_count": int(model.policy.observation_space.shape[0]),
        "hidden_size": int(model.policy.lstm_actor.hidden_size),
        "action_count": int(model.policy.action_space.n),
        "feature_names": feature_names
    }
    
    # Validate config
    try:
        validate_config(config)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Invalid configuration: {e}")
    
    # Save config
    config_path = output_dir / "model_config.json"
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved model config to {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model config: {e}")
    
    print("Export completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)
