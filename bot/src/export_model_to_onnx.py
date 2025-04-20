#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from stable_baselines3.common.preprocessing import preprocess_obs
from collections import namedtuple

def export_recurrent_ppo_to_onnx(model_path, output_path, input_shape=(1, 500, 11)):
    """
    Export a RecurrentPPO model to ONNX format for use in MQL5.
    
    Args:
        model_path: Path to the saved RecurrentPPO model (.zip file)
        output_path: Path where to save the ONNX model
        input_shape: Shape of the input tensor (batch_size, sequence_length, features)
                    Default is (1, 500, 11) for 11 features:
                    - returns, rsi, atr, volume_change, volatility_breakout, trend_strength,
                    - candle_pattern, sin_time, cos_time, position_type, unrealized_pnl
    """
    print(f"Loading model from {model_path}...")
    model = RecurrentPPO.load(model_path)
    
    # Print model structure to help debug
    policy = model.policy
    print("Model loaded. Policy structure:")
    print(f"- Feature extractor type: {type(policy.features_extractor).__name__ if hasattr(policy, 'features_extractor') else 'None'}")
    print(f"- MLP extractor type: {type(policy.mlp_extractor).__name__ if hasattr(policy, 'mlp_extractor') else 'None'}")
    print(f"- LSTM layer info: hidden_size={policy.lstm_actor.hidden_size}, num_layers={policy.lstm_actor.num_layers}")
    print(f"- Action distribution type: {type(policy.action_dist).__name__ if hasattr(policy, 'action_dist') else 'None'}")
    
    # Get important dimensions
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    feature_dim = input_shape[2]
    hidden_size = policy.lstm_actor.hidden_size
    num_layers = policy.lstm_actor.num_layers
    
    # Create a wrapper class that mimics only the needed parts of the policy
    class SimpleRecurrentWrapper(nn.Module):
        def __init__(self, policy):
            super(SimpleRecurrentWrapper, self).__init__()
            # Extract the needed components from the policy
            self.features_extractor = policy.features_extractor
            self.lstm = policy.lstm_actor
            self.action_net = policy.action_net
            self.mlp_extractor = policy.mlp_extractor
            self.hidden_size = policy.lstm_actor.hidden_size
            self.num_layers = policy.lstm_actor.num_layers
            
            # Print details for debugging
            print(f"LSTM Details - input_size: {self.lstm.input_size}, hidden_size: {self.hidden_size}")
            if hasattr(self.action_net, 'in_features'):
                print(f"Action net - in_features: {self.action_net.in_features}, out_features: {self.action_net.out_features}")
            
        def forward(self, obs, lstm_h, lstm_c):
            # obs: [batch_size, seq_length, feature_dim]
            batch_size = obs.shape[0]
            seq_length = obs.shape[1]
            feature_dim = obs.shape[2]
            
            # Ensure LSTM states have the right shape
            if lstm_h.shape != (self.num_layers, batch_size, self.hidden_size):
                lstm_h = lstm_h.reshape(self.num_layers, batch_size, self.hidden_size)
            
            if lstm_c.shape != (self.num_layers, batch_size, self.hidden_size):
                lstm_c = lstm_c.reshape(self.num_layers, batch_size, self.hidden_size)
            
            # LSTM states tuple
            lstm_states = (lstm_h, lstm_c)
            
            # Step 1: Process the observations through LSTM first
            # Reshape for LSTM input [seq_len, batch, features]
            lstm_input = obs.permute(1, 0, 2)
            
            # Run through LSTM - this gives all outputs for each timestep
            lstm_out, (h_n, c_n) = self.lstm(lstm_input, lstm_states)
            
            # Take only the last output from the sequence
            last_output = lstm_out[-1]  # Shape: [batch_size, hidden_size]
            
            # Pass through the policy network from MLP extractor
            policy_latent = self.mlp_extractor.policy_net(last_output)
            
            # Get action logits
            action_logits = self.action_net(policy_latent)
            
            # Apply softmax to get probabilities
            action_probs = torch.softmax(action_logits, dim=-1)
            
            return action_probs, h_n, c_n
    
    # Create the wrapper
    wrapper = SimpleRecurrentWrapper(policy)
    wrapper.eval()  # Set to evaluation mode
    
    # Create dummy inputs with the expected shapes
    dummy_obs = torch.zeros(input_shape, dtype=torch.float32)
    lstm_h = torch.zeros((num_layers, batch_size, hidden_size), dtype=torch.float32)
    lstm_c = torch.zeros((num_layers, batch_size, hidden_size), dtype=torch.float32)
    
    # Test the forward pass
    print("\nTesting wrapper forward pass...")
    try:
        action_probs, new_h, new_c = wrapper(dummy_obs, lstm_h, lstm_c)
        print(f"Forward pass successful!")
        print(f"Output shapes: action_probs={action_probs.shape}, new_h={new_h.shape}, new_c={new_c.shape}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        print(f"\nExtracting more detailed network information:")
        
        # Print more information about the network components
        for name, module in policy.named_modules():
            print(f"Module: {name}, Type: {type(module).__name__}")
            if isinstance(module, nn.Linear):
                print(f"  Linear layer: in_features={module.in_features}, out_features={module.out_features}")
            elif isinstance(module, nn.LSTM):
                print(f"  LSTM layer: input_size={module.input_size}, hidden_size={module.hidden_size}, num_layers={module.num_layers}")
        raise
    
    # Export the model to ONNX
    print(f"Exporting model to {output_path}...")
    try:
        torch.onnx.export(
            wrapper,
            (dummy_obs, lstm_h, lstm_c),
            output_path,
            export_params=True,
            opset_version=12,  # Using higher opset version for better compatibility
            do_constant_folding=True,
            input_names=['observation', 'lstm_h', 'lstm_c'],
            output_names=['action_probs', 'new_lstm_h', 'new_lstm_c'],
            dynamic_axes={
                'observation': {0: 'batch', 1: 'sequence'},
                'lstm_h': {1: 'batch'},
                'lstm_c': {1: 'batch'},
                'action_probs': {0: 'batch'},
                'new_lstm_h': {1: 'batch'},
                'new_lstm_c': {1: 'batch'},
            }
        )
        print("Model exported successfully!")
    except Exception as e:
        print(f"Export failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RecurrentPPO model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved RecurrentPPO model")
    parser.add_argument("--output_path", type=str, required=True, help="Path where to save the ONNX model")
    parser.add_argument("--seq_length", type=int, default=500, help="Sequence length (number of bars)")
    parser.add_argument("--features", type=int, default=11, help="Number of features per bar (default: 11)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    export_recurrent_ppo_to_onnx(
        args.model_path, 
        args.output_path,
        input_shape=(1, args.seq_length, args.features)
    )