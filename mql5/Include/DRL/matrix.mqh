// Matrix operations for LSTM computations
// Generated on: 2025-04-17 11:31:03

#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"
#property version   "1.00"

#ifndef _DRL_MATRIX_H_
#define _DRL_MATRIX_H_

// Matrix multiplication: C = A * B
void MatrixMultiply(const double& a[], const double& b[], double& c[],
                    const int a_rows, const int a_cols,
                    const int b_rows, const int b_cols) {
    if(a_cols != b_rows)
        return;

    ArrayResize(c, a_rows * b_cols);
    ArrayInitialize(c, 0);

    for(int i=0; i<a_rows; i++) {
        for(int j=0; j<b_cols; j++) {
            for(int k=0; k<a_cols; k++) {
                c[i*b_cols + j] += a[i*a_cols + k] * b[k*b_cols + j];
            }
        }
    }
}

// Vector addition: C = A + B
void VectorAdd(const double& a[], const double& b[], double& c[], const int size) {
    ArrayResize(c, size);
    for(int i=0; i<size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Apply activation function element-wise
void ApplyActivation(const double& input[], double& output[],
                     const int size, const string activation) {
    ArrayResize(output, size);
    for(int i=0; i<size; i++) {
        if(activation == "tanh")
            output[i] = custom_tanh(input[i]);
        else if(activation == "sigmoid")
            output[i] = sigmoid(input[i]);
        else
            output[i] = input[i];  // Linear activation
    }
}

#endif  // _DRL_MATRIX_H_
