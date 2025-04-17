// Auto-generated LSTM model architecture
// Generated on: 2025-04-17 18:44:26

#include <Trade/Trade.mqh>
#include <Math/Stat/Math.mqh>

#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"
#property version   "1.00"

#ifndef _DRL_MODEL_H_
#define _DRL_MODEL_H_

// Model Architecture Constants
#define FEATURE_COUNT 11
#define LSTM_UNITS 256
#define FC_UNITS 64           // Fully connected layer size
#define ACTION_COUNT 4

// Matrix Dimensions Constants
#define INPUT_WEIGHT_COLS (LSTM_UNITS * 4)  // 1024
#define HIDDEN_WEIGHT_COLS (LSTM_UNITS * 4) // 1024
#define FC_WEIGHT_COLS FC_UNITS            // 64
#define FC_WEIGHT_ROWS LSTM_UNITS          // 256
#define FC_BIAS_SIZE FC_UNITS              // 64
#define OUTPUT_WEIGHT_COLS ACTION_COUNT     // 4
#define OUTPUT_WEIGHT_ROWS FC_UNITS        // 64

// Activation Functions
double custom_tanh(const double x) {
    const double ep = MathExp(x);
    const double em = MathExp(-x);
    return (ep - em) / (ep + em);
}

double sigmoid(const double x) {
    return 1.0 / (1.0 + MathExp(-x));
}

double relu(const double x) {
    return x > 0.0 ? x : 0.0;
}

#endif  // _DRL_MODEL_H_
