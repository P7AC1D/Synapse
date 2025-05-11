#!/usr/bin/env python3

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import argparse
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import torch.nn as nn
from stable_baselines3.common.preprocessing import preprocess_obs
from collections import namedtuple

def export_ppo_to_onnx(model_path, output_path, input_shape=(1, 11), is_recurrent=False):
    """
    Export a PPO or RecurrentPPO model to ONNX format for use in cTrader or MQL5.
    
    Args:
        model_path: Path to the saved PPO model (.zip file)
        output_path: Path where to save the ONNX model
        input_shape: Shape of the input tensor
                    - For standard PPO: (batch_size, features)
                    - For RecurrentPPO: (batch_size, sequence_length, features)
                    Default features include:
                    - returns, rsi, atr, volume_change, volatility_breakout, trend_strength,
                    - candle_pattern, sin_time, cos_time, position_type, unrealized_pnl
        is_recurrent: Whether the model is a RecurrentPPO model or standard PPO model
    """
    print(f"Loading model from {model_path}...")
    
    if is_recurrent:
        model = RecurrentPPO.load(model_path)
        print("Loading RecurrentPPO model...")
        return export_recurrent_ppo(model, output_path, input_shape)
    else:
        model = PPO.load(model_path)
        print("Loading standard PPO model...")
        return export_standard_ppo(model, output_path, input_shape)

def export_standard_ppo(model, output_path, input_shape=(1, 11)):
    """Export a standard PPO model to ONNX format."""
    policy = model.policy
    print("Model loaded. Policy structure:")
    print(f"- Feature extractor type: {type(policy.features_extractor).__name__ if hasattr(policy, 'features_extractor') else 'None'}")
    print(f"- MLP extractor type: {type(policy.mlp_extractor).__name__ if hasattr(policy, 'mlp_extractor') else 'None'}")
    print(f"- Action distribution type: {type(policy.action_dist).__name__ if hasattr(policy, 'action_dist') else 'None'}")
    
    # Create a wrapper class for standard PPO policy
    class SimplePPOWrapper(nn.Module):
        def __init__(self, policy):
            super(SimplePPOWrapper, self).__init__()
            # Extract the needed components from the policy
            self.features_extractor = policy.features_extractor
            self.action_net = policy.action_net
            self.mlp_extractor = policy.mlp_extractor
            
            # Print details for debugging
            if hasattr(self.action_net, 'in_features'):
                print(f"Action net - in_features: {self.action_net.in_features}, out_features: {self.action_net.out_features}")
            
        def forward(self, obs):
            # Extract features if applicable
            if self.features_extractor is not None:
                features = self.features_extractor(obs)
            else:
                features = obs
            
            # Pass through the policy network from MLP extractor
            policy_latent = self.mlp_extractor.policy_net(features)
            
            # Get raw logits without softmax
            action_logits = self.action_net(policy_latent)
            
            # Return raw logits for consistent action processing
            return action_logits
    
    # Create the wrapper
    wrapper = SimplePPOWrapper(policy)
    wrapper.eval()  # Set to evaluation mode
    
    # Create dummy inputs with the expected shapes
    dummy_obs = torch.zeros(input_shape, dtype=torch.float32)
    
    # Test the forward pass
    print("\nTesting wrapper forward pass...")
    try:
        action_logits = wrapper(dummy_obs)
        print(f"Forward pass successful!")
        print(f"Output shapes: action_logits={action_logits.shape}")
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
        print(f"\nExtracting more detailed network information:")
        
        # Print more information about the network components
        for name, module in policy.named_modules():
            print(f"Module: {name}, Type: {type(module).__name__}")
            if isinstance(module, nn.Linear):
                print(f"  Linear layer: in_features={module.in_features}, out_features={module.out_features}")
        raise
    
    # Export the model to ONNX
    print(f"Exporting model to {output_path}...")
    try:
        torch.onnx.export(
            wrapper,
            dummy_obs,
            output_path,
            export_params=True,
            opset_version=12,  # Using higher opset version for better compatibility
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action_logits'],
            dynamic_axes={
                'observation': {0: 'batch'},
                'action_logits': {0: 'batch'},
            }
        )
        print("Model exported successfully!")
    except Exception as e:
        print(f"Export failed: {str(e)}")
        raise

def export_recurrent_ppo(model, output_path, input_shape=(1, 500, 11)):
    """Export a RecurrentPPO model to ONNX format."""
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
            
            # Get raw action logits (no softmax)
            action_logits = self.action_net(policy_latent)
            
            # Return raw logits for consistent processing
            return action_logits, h_n, c_n
    
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
        action_logits, new_h, new_c = wrapper(dummy_obs, lstm_h, lstm_c)
        print(f"Forward pass successful!")
        print(f"Output shapes: action_logits={action_logits.shape}, new_h={new_h.shape}, new_c={new_c.shape}")
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
            output_names=['action_logits', 'new_lstm_h', 'new_lstm_c'],
            dynamic_axes={
                'observation': {0: 'batch', 1: 'sequence'},
                'lstm_h': {1: 'batch'},
                'lstm_c': {1: 'batch'},
                'action_logits': {0: 'batch'},
                'new_lstm_h': {1: 'batch'},
                'new_lstm_c': {1: 'batch'},
            }
        )
        print("Model exported successfully!")
        
        # Validate the exported model
        print("\nValidating exported ONNX model...")
        import onnx
        from onnx import checker, shape_inference
        
        # Load and check the model
        model = onnx.load(output_path)
        checker.check_model(model)
        
        # Print model information
        print("\nModel Input Details:")
        for input in model.graph.input:
            print(f"- Name: {input.name}")
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in input.type.tensor_type.shape.dim]
            print(f"  Shape: {shape}")
        
        print("\nModel Output Details:")
        for output in model.graph.output:
            print(f"- Name: {output.name}")
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]
            print(f"  Shape: {shape}")
        
        print("\nModel validation successful!")
        
        # Compare PyTorch and ONNX outputs
        print("\nComparing PyTorch and ONNX outputs...")
        import onnxruntime as ort
        
        # Create ONNX inference session
        sess = ort.InferenceSession(output_path)
        
        if is_recurrent:
            # Run PyTorch model
            with torch.no_grad():
                torch_out = wrapper(dummy_obs, lstm_h, lstm_c)
            
            # Run ONNX model
            onnx_out = sess.run(
                ['action_logits', 'new_lstm_h', 'new_lstm_c'],
                {
                    'observation': dummy_obs.numpy(),
                    'lstm_h': lstm_h.numpy(),
                    'lstm_c': lstm_c.numpy()
                }
            )
            
            # Compare outputs
            torch_logits = torch_out[0].numpy()
            onnx_logits = onnx_out[0]
            max_diff = np.abs(torch_logits - onnx_logits).max()
            print(f"Max difference between PyTorch and ONNX logits: {max_diff}")
            if max_diff > 1e-6:
                print("WARNING: Large difference between PyTorch and ONNX outputs!")
            else:
                print("PyTorch and ONNX outputs match!")
        else:
            # Run PyTorch model
            with torch.no_grad():
                torch_out = wrapper(dummy_obs)
            
            # Run ONNX model
            onnx_out = sess.run(['action_logits'], {'observation': dummy_obs.numpy()})
            
            # Compare outputs
            torch_logits = torch_out.numpy()
            onnx_logits = onnx_out[0]
            max_diff = np.abs(torch_logits - onnx_logits).max()
            print(f"Max difference between PyTorch and ONNX logits: {max_diff}")
            if max_diff > 1e-6:
                print("WARNING: Large difference between PyTorch and ONNX outputs!")
            else:
                print("PyTorch and ONNX outputs match!")
        
    except Exception as e:
        print(f"Export/validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PPO or RecurrentPPO model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved PPO model")
    parser.add_argument("--output_path", type=str, required=True, help="Path where to save the ONNX model")
    parser.add_argument("--is_recurrent", action="store_true", help="Whether the model is RecurrentPPO")
    parser.add_argument("--seq_length", type=int, default=500, help="Sequence length for RecurrentPPO (number of bars)")
    parser.add_argument("--features", type=int, default=11, help="Number of features per bar (default: 11)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.is_recurrent:
        export_ppo_to_onnx(
            args.model_path, 
            args.output_path,
            input_shape=(1, args.seq_length, args.features),
            is_recurrent=True
        )
    else:
        export_ppo_to_onnx(
            args.model_path, 
            args.output_path,
            input_shape=(1, args.features),
            is_recurrent=False
        )
