// Auto-generated LSTM model architecture
// Generated on: 2025-04-17 15:26:23

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

// Activation Functions
double custom_tanh(const double x) {
    const double ep = MathExp(x);
    const double em = MathExp(-x);
    return (ep - em) / (ep + em);
}

double sigmoid(const double x) {
    return 1.0 / (1.0 + MathExp(-x));
}

#endif  // _DRL_MODEL_H_
