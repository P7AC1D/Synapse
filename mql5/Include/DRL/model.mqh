// Auto-generated LSTM model architecture
// Generated on: 2025-04-18 04:45:45

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
#define ACTION_COUNT 4

// LSTM Gate Constants
#define GATES_PER_UNIT 4  // input, forget, cell, output gates

// Matrix Dimensions Constants
#define INPUT_WEIGHT_COLS (LSTM_UNITS * GATES_PER_UNIT)   // input weights: [feature_count][lstm_units * 4]
#define HIDDEN_WEIGHT_COLS (LSTM_UNITS * GATES_PER_UNIT)  // hidden weights: [lstm_units][lstm_units * 4]
#define OUTPUT_WEIGHT_COLS ACTION_COUNT           // output weights: [lstm_units][action_count]
#define OUTPUT_WEIGHT_ROWS LSTM_UNITS            // output directly from LSTM to actions

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
