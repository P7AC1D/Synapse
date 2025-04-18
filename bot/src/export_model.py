#!/usr/bin/env python3

"""
Export PPO-LSTM model to MQL5 format.
Extracts model architecture, weights, and generates verification test cases.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
from sb3_contrib.ppo_recurrent import RecurrentPPO

from trade_model import TradeModel
from trading.dummy_processor import DummyFeatureProcessor

import argparse

def generate_test_cases(trade_model: TradeModel, feature_processor: DummyFeatureProcessor) -> List[Dict[str, Any]]:
    """Generate test cases for model verification."""
    test_cases = []
    
    # Initialize empty LSTM state
    lstm_state = np.zeros(trade_model.model.policy.lstm_actor.hidden_size, dtype=np.float32)
    
    # Generate base features using feature processor
    base_features = np.zeros(9)  # Same size as in ProcessFeatures
    
    # Add position features
    features = np.concatenate([base_features, [0.0, 0.0]])  # Add position size and profit
    
    # Create test case
    test_case = {
        'features': features.tolist(),
        'lstm_state': lstm_state.tolist(),
        'expected_action': 0  # Default to HOLD action
    }
    
    test_cases.append(test_case)
    return test_cases

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export PPO-LSTM model to MQL5 format.")
    parser.add_argument("--model-path", 
                       default="bot/model/XAUUSDm.zip",
                       help="Path to the model file (default: bot/model/XAUUSDm.zip)")
    return parser.parse_args()

def create_export_dirs() -> Tuple[Path, Path]:
    """Create directories for MQL5 export files."""
    # Create mql5 directory structure
    mql5_dir = Path('mql5')
    include_dir = mql5_dir / 'Include' / 'DRL'
    experts_dir = mql5_dir / 'Experts'
    
    for dir_path in [include_dir, experts_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return include_dir, experts_dir

def extract_lstm_params(model: RecurrentPPO) -> Dict[str, np.ndarray]:
    """Extract LSTM layer parameters from PPO model."""
    try:
        # Access LSTM layers from the model
        lstm_actor = model.policy.lstm_actor
        lstm_critic = model.policy.lstm_critic
        
        # Extract weights and biases
        # Extract and print shapes before converting
        actor_input_w = lstm_actor.weight_ih_l0.detach().numpy()
        actor_hidden_w = lstm_actor.weight_hh_l0.detach().numpy()
        actor_output_w = model.policy.action_net.weight.detach().numpy()
        
        print("PyTorch shapes:")
        print(f"actor_input_weight: {actor_input_w.shape}")
        print(f"actor_hidden_weight: {actor_hidden_w.shape}")
        print(f"actor_output_weight: {actor_output_w.shape}")
        
        # Create weights dictionary
        weights = {
            # Actor LSTM parameters
            'actor_input_weight': actor_input_w,
            'actor_hidden_weight': actor_hidden_w,
            'actor_input_bias': lstm_actor.bias_ih_l0.detach().numpy(),
            'actor_hidden_bias': lstm_actor.bias_hh_l0.detach().numpy(),
            
            # Critic LSTM parameters
            'critic_input_weight': lstm_critic.weight_ih_l0.detach().numpy(),
            'critic_hidden_weight': lstm_critic.weight_hh_l0.detach().numpy(),
            'critic_input_bias': lstm_critic.bias_ih_l0.detach().numpy(),
            'critic_hidden_bias': lstm_critic.bias_hh_l0.detach().numpy(),
            
            # Output layer parameters
            'actor_output_weight': actor_output_w,
            'actor_output_bias': model.policy.action_net.bias.detach().numpy(),
            'critic_output_weight': model.policy.value_net.weight.detach().numpy(),
            'critic_output_bias': model.policy.value_net.bias.detach().numpy()
        }
        
        return weights
        
    except Exception as e:
        raise ValueError(f"Failed to extract LSTM parameters: {e}")

def generate_mql5_array(arr: np.ndarray, name: str, const: bool = True) -> str:
    """Generate MQL5 array declaration with proper 2D dimensions."""
    shape = arr.shape
    
    # Special handling for actor/critic weight matrices
    is_weight_matrix = any(x in name for x in ["input_weight", "hidden_weight", "output_weight", "fc_weight"])
    
    if len(shape) > 1 and is_weight_matrix:
        # Transpose the weight matrices to match required dimensions
        if "input_weight" in name:
            # PyTorch: [1024][11] -> MQL5: [11][1024]
            arr = arr.T
        elif "hidden_weight" in name:
            # PyTorch: [1024][256] -> MQL5: [256][1024]
            arr = arr.T
        elif "fc_weight" in name:
            # PyTorch: [64][256] -> MQL5: [256][64]
            arr = arr.T
        elif "output_weight" in name:
            # PyTorch: [4][64] -> MQL5: [64][4]
            arr = arr.T
            
        shape = arr.shape  # Get new shape after transpose
        lines = [f"{'const ' if const else ''}double {name}[{shape[0]}][{shape[1]}] = {{"]
        
        # Format each row
        for i in range(shape[0]):
            row = [f"{x:.10f}" for x in arr[i]]
            if i < shape[0] - 1:
                lines.append("    {" + ", ".join(row) + "},")
            else:
                lines.append("    {" + ", ".join(row) + "}")
    else:
        # 1D array for bias vectors and other arrays
        flat = arr.flatten()
        formatted = [f"{x:.10f}" for x in flat]
        
        lines = [f"{'const ' if const else ''}double {name}[] = {{"]
        lines.append("    " + ", ".join(formatted))
        
    lines.append("};")
    lines.append(f"// Original shape before transpose: {arr.T.shape if len(shape) > 1 and is_weight_matrix else shape}")
    lines.append("")  # Empty line for readability
    
    return "\n".join(lines)

def export_weights_mqh(weights: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Export model weights to weights.mqh."""
    content = [
        "// Auto-generated LSTM model weights",
        "// Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "#property copyright \"Copyright 2024, DRL Trading Bot\"",
        "#property link      \"https://github.com/your-repo\"",
        "#property version   \"1.00\"",
        "",
        "#ifndef _DRL_WEIGHTS_H_",
        "#define _DRL_WEIGHTS_H_",
        "",
        "// LSTM Layer Weights and Biases",
        ""
    ]
    
    # Export each weight matrix/vector
    for name, arr in weights.items():
        content.append(generate_mql5_array(arr, name))
        
    content.extend([
        "#endif  // _DRL_WEIGHTS_H_",
        ""
    ])
    
    with open(output_dir / 'weights.mqh', 'w') as f:
        f.write('\n'.join(content))

def export_model_mqh(model: RecurrentPPO, output_dir: Path) -> None:
    """Export model architecture to model.mqh."""
    content = [
        "// Auto-generated LSTM model architecture",
        "// Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "#include <Trade/Trade.mqh>",
        "#include <Math/Stat/Math.mqh>",
        "",
        "#property copyright \"Copyright 2024, DRL Trading Bot\"",
        "#property link      \"https://github.com/your-repo\"",
        "#property version   \"1.00\"",
        "",
        "#ifndef _DRL_MODEL_H_",
        "#define _DRL_MODEL_H_",
        "",
        "// Model Architecture Constants",
        f"#define FEATURE_COUNT {model.policy.observation_space.shape[0]}",
        f"#define LSTM_UNITS {model.policy.lstm_actor.hidden_size}",
        f"#define ACTION_COUNT {model.policy.action_space.n}",
        "",
        "// Matrix Dimensions Constants",
        "#define INPUT_WEIGHT_COLS (LSTM_UNITS * 4)  // 1024",
        "#define HIDDEN_WEIGHT_COLS (LSTM_UNITS * 4) // 1024",
        "#define OUTPUT_WEIGHT_COLS ACTION_COUNT     // 4",
        "#define OUTPUT_WEIGHT_ROWS LSTM_UNITS      // 256",
        "",
        "// Activation Functions",
        "double custom_tanh(const double x) {",
        "    const double ep = MathExp(x);",
        "    const double em = MathExp(-x);",
        "    return (ep - em) / (ep + em);",
        "}",
        "",
        "double sigmoid(const double x) {",
        "    return 1.0 / (1.0 + MathExp(-x));",
        "}",
        "",
        "double relu(const double x) {",
        "    return x > 0.0 ? x : 0.0;",
        "}",
        "",
        "#endif  // _DRL_MODEL_H_",
        ""
    ]
    
    with open(output_dir / 'model.mqh', 'w') as f:
        f.write('\n'.join(content))

def export_test_cases_mqh(test_cases: List[Dict[str, Any]], output_dir: Path) -> None:
    """Export test cases to test_cases.mqh."""
    content = [
        "// Auto-generated test cases for model verification",
        "// Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "#include <Trade/Trade.mqh>",
        "#include <Math/Math.mqh>",
        "#include <Arrays/ArrayDouble.mqh>",
        "",
        "#property copyright \"Copyright 2024, DRL Trading Bot\"",
        "#property link      \"https://github.com/your-repo\"",
        "#property version   \"1.00\"",
        "",
        "#ifndef _DRL_TEST_CASES_H_",
        "#define _DRL_TEST_CASES_H_",
        "",
        "// Test Case Structure",
        "struct TestCase {",
        "    double features[];",
        "    double lstm_state[];",
        "    int expected_action;",
        "};",
        "",
        f"#define TEST_CASE_COUNT {len(test_cases)}",
        "",
        "// Test Cases",
    ]
    
    # Generate test case arrays
    for i, case in enumerate(test_cases):
        content.extend([
            f"// Test Case {i}",
            generate_mql5_array(np.array(case['features']), f"test_case_{i}_features"),
            generate_mql5_array(np.array(case['lstm_state']), f"test_case_{i}_lstm_state"),
            f"const int test_case_{i}_action = {case['expected_action']};",
            ""
        ])
    
    # Generate test case initialization function
    content.extend([
        "// Initialize test cases",
        "void InitTestCases(TestCase &cases[]) {",
        f"    ArrayResize(cases, {len(test_cases)});",
        ""
    ])
    
    for i in range(len(test_cases)):
        content.extend([
            f"    // Initialize test case {i}",
            f"    ArrayCopy(cases[{i}].features, test_case_{i}_features);",
            f"    ArrayCopy(cases[{i}].lstm_state, test_case_{i}_lstm_state);",
            f"    cases[{i}].expected_action = test_case_{i}_action;",
            ""
        ])
    
    content.extend([
        "}",
        "",
        "#endif  // _DRL_TEST_CASES_H_",
        ""
    ])
    
    with open(output_dir / 'test_cases.mqh', 'w') as f:
        f.write('\n'.join(content))

def export_matrix_mqh(output_dir: Path) -> None:
    """Export matrix operations to matrix.mqh."""
    content = [
        "// Matrix operations for LSTM computations",
        "// Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "#property copyright \"Copyright 2024, DRL Trading Bot\"",
        "#property link      \"https://github.com/your-repo\"",
        "#property version   \"1.00\"",
        "",
        "#ifndef _DRL_MATRIX_H_",
        "#define _DRL_MATRIX_H_",
        "",
        "// Matrix multiplication: C = A * B",
        "void MatrixMultiply(const double &a[], const double &b[], double &c[],",
        "                   const int a_rows, const int a_cols,",
        "                   const int b_rows, const int b_cols) {",
        "    if(a_cols != b_rows)",
        "        return;",
        "",
        "    ArrayResize(c, a_rows * b_cols);",
        "    ArrayInitialize(c, 0);",
        "",
        "    for(int i=0; i<a_rows; i++) {",
        "        for(int j=0; j<b_cols; j++) {",
        "            for(int k=0; k<a_cols; k++) {",
        "                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];",
        "            }",
        "        }",
        "    }",
        "}",
        "",
        "// Vector addition: C = A + B",
        "void VectorAdd(const double &a[], const double &b[], double &c[], const int size) {",
        "    ArrayResize(c, size);",
        "    for(int i=0; i<size; i++) {",
        "        c[i] = a[i] + b[i];",
        "    }",
        "}",
        "",
        "// Apply activation function element-wise",
        "void ApplyActivation(const double &in_values[], double &output[],", 
        "                    const int size, const string activation) {",
        "    ArrayResize(output, size);",
        "    for(int i=0; i<size; i++) {",
        "        if(activation == \"tanh\")",
        "            output[i] = custom_tanh(in_values[i]);",
        "        else if(activation == \"sigmoid\")",
        "            output[i] = sigmoid(in_values[i]);",
        "        else if(activation == \"relu\")",
        "            output[i] = relu(in_values[i]);",
        "        else",
        "            output[i] = in_values[i];  // Linear activation",
        "    }",
        "}",
        "",
        "#endif  // _DRL_MATRIX_H_",
        ""
    ]
    
    with open(output_dir / 'matrix.mqh', 'w') as f:
        f.write('\n'.join(content))

def export_features_mqh(feature_processor: DummyFeatureProcessor, output_dir: Path) -> None:
    """Export feature processing to features.mqh."""
    content = [
        "// Feature processing for DRL model",
        "// Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "#include <Trade/Trade.mqh>",
        "#include <Arrays/ArrayDouble.mqh>",
        "#include <Math/Stat/Math.mqh>",
        "",
        "#property copyright \"Copyright 2024, DRL Trading Bot\"",
        "#property link      \"https://github.com/your-repo\"",
        "#property version   \"1.00\"",
        "",
        "#ifndef _DRL_FEATURES_H_",
        "#define _DRL_FEATURES_H_",
        "",
        "// Indicator Parameters",
        f"#define RSI_PERIOD {feature_processor.rsi_period}",
        f"#define ATR_PERIOD {feature_processor.atr_period}",
        f"#define BOLL_PERIOD {feature_processor.boll_period}",
        "",
        "// Time constants",
        "#define MINUTES_IN_DAY 1440",
        "",
        "class CFeatureProcessor {",
        "private:",
        "    int m_atr_handle;",
        "    int m_rsi_handle;",
        "    int m_bb_handle;",
        "    int m_adx_handle;",
        "",
        "public:",
        "    void Init(const string symbol, const ENUM_TIMEFRAMES timeframe) {",
        "        m_atr_handle = iATR(symbol, timeframe, ATR_PERIOD);",
        "        m_rsi_handle = iRSI(symbol, timeframe, RSI_PERIOD, PRICE_CLOSE);",
        "        m_bb_handle = iBands(symbol, timeframe, BOLL_PERIOD, 0, 2, PRICE_CLOSE);",
        "        m_adx_handle = iADX(symbol, timeframe, ATR_PERIOD);",
        "    }",
        "",
        "    void ProcessFeatures(double& features[]) {",
        "        double close[];",
        "        double open[];", 
        "        double high[];",
        "        double low[];",
        "        long volume[];",
        "        ArraySetAsSeries(close, true);",
        "        ArraySetAsSeries(open, true);",
        "        ArraySetAsSeries(high, true);",
        "        ArraySetAsSeries(low, true);",
        "        ArraySetAsSeries(volume, true);",
        "",
        "",
        "        // Get price data",
        "        CopyClose(_Symbol, _Period, 0, 2, close);",
        "        CopyOpen(_Symbol, _Period, 0, 1, open);",
        "        CopyHigh(_Symbol, _Period, 0, 1, high);",
        "        CopyLow(_Symbol, _Period, 0, 1, low);",
        "        CopyTickVolume(_Symbol, _Period, 0, 2, volume);",
        "",
        "        // Calculate returns",
        "        double returns = (close[0] - close[1]) / close[1];",
        "        returns = MathMax(MathMin(returns, 0.1), -0.1);",
        "",
        "        // Get indicators",
        "        double atr[], rsi[], bb_upper[], bb_lower[], adx[];",
        "        ArraySetAsSeries(atr, true);",
        "        ArraySetAsSeries(rsi, true);",
        "        ArraySetAsSeries(bb_upper, true);",
        "        ArraySetAsSeries(bb_lower, true);",
        "        ArraySetAsSeries(adx, true);",
        "",
        "        CopyBuffer(m_atr_handle, 0, 0, 1, atr);",
        "        CopyBuffer(m_rsi_handle, 0, 0, 1, rsi);",
        "        CopyBuffer(m_bb_handle, 1, 0, 1, bb_upper);",
        "        CopyBuffer(m_bb_handle, 2, 0, 1, bb_lower);",
        "        CopyBuffer(m_adx_handle, 0, 0, 1, adx);",
        "",
        "        // Normalize RSI to [-1, 1]",
        "        double norm_rsi = rsi[0] / 50.0 - 1.0;",
        "",
        "        // Normalize ATR",
        "        double norm_atr = 2.0 * (atr[0] / close[0]) - 1.0;",
        "",
        "        // Calculate volatility breakout",
        "        double band_range = bb_upper[0] - bb_lower[0];",
        "        double position = close[0] - bb_lower[0];",
        "        double volatility_breakout = position / (band_range + 1e-8);",
        "        volatility_breakout = MathMax(MathMin(volatility_breakout, 1.0), 0.0);",
        "",
        "        // Calculate trend strength",
        "        double trend_strength = MathMax(MathMin(adx[0]/25.0 - 1.0, 1.0), -1.0);",
        "",
        "        // Calculate candle pattern",
        "        double body = close[0] - open[0];",
        "        double upper_wick = high[0] - MathMax(close[0], open[0]);",
        "        double lower_wick = MathMin(close[0], open[0]) - low[0];",
        "        double range = high[0] - low[0] + 1e-8;",
        "        double candle_pattern = (body/range + ",
        "                               (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2.0;",
        "        candle_pattern = MathMax(MathMin(candle_pattern, 1.0), -1.0);",
        "",
        "        // Calculate time features",
        "        MqlDateTime time;",
        "        TimeToStruct(TimeCurrent(), time);",
        "        int minutes = time.hour * 60 + time.min;",
        "        double sin_time = MathSin(2.0 * M_PI * minutes / MINUTES_IN_DAY);",
        "        double cos_time = MathCos(2.0 * M_PI * minutes / MINUTES_IN_DAY);",
        "",
        "        // Calculate volume change",
        "        double volume_change = 0.0;",
        "        if(volume[1] > 0) {",
        "            volume_change = ((double)volume[0] - (double)volume[1]) / (double)volume[1];",
        "            volume_change = MathMax(MathMin(volume_change, 1.0), -1.0);",
        "        }",
        "",
        "        // Set features array",
        "        ArrayResize(features, 9);  // Base features (position features added separately)",
        "        features[0] = returns;",
        "        features[1] = norm_rsi;",
        "        features[2] = norm_atr;",
        "        features[3] = volatility_breakout;",
        "        features[4] = trend_strength;",
        "        features[5] = candle_pattern;",
        "        features[6] = sin_time;",
        "        features[7] = cos_time;",
        "        features[8] = volume_change;",
        "    }",
        "",
        "    void Deinit() {",
        "        IndicatorRelease(m_atr_handle);",
        "        IndicatorRelease(m_rsi_handle);",
        "        IndicatorRelease(m_bb_handle);",
        "        IndicatorRelease(m_adx_handle);",
        "    }",
        "};",
        "",
        "#endif  // _DRL_FEATURES_H_",
        ""
    ]
    
    with open(output_dir / 'features.mqh', 'w') as f:
        f.write('\n'.join(content))

def main():
    """Main export function."""
    print("Starting model export process...")
    args = parse_args()
    
    try:
        # Create export directories
        include_dir, experts_dir = create_export_dirs()
        print(f"Created export directories at {include_dir}")
        
        # Load model
        print(f"Loading model from {args.model_path}...")
        trade_model = TradeModel(args.model_path)
        model = trade_model.model
        if model is None:
            raise ValueError("Failed to load model")
            
        # Extract LSTM parameters
        print("Extracting LSTM parameters...")
        weights = extract_lstm_params(model)
        
        # Create feature processor
        feature_processor = DummyFeatureProcessor()
        
        # Generate test cases
        print("Generating test cases...")
        test_cases = generate_test_cases(trade_model, feature_processor)
        
        # Export MQL5 files
        print("Exporting MQL5 files...")
        export_weights_mqh(weights, include_dir)
        export_model_mqh(model, include_dir)
        export_test_cases_mqh(test_cases, include_dir)
        export_matrix_mqh(include_dir)
        export_features_mqh(feature_processor, include_dir)
        
        print("Export completed successfully!")
        print(f"Files exported to: {include_dir}")
        
    except Exception as e:
        print(f"Error during export: {e}")
        raise

if __name__ == "__main__":
    main()
