//+------------------------------------------------------------------+
//|                                                    DRLTester.mq5    |
//|                                   Copyright 2024, DRL Trading Bot   |
//|                                     https://github.com/your-repo    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link "https://github.com/your-repo"
#property version "1.00"
#property strict

#include <DRL/model.mqh>
#include <DRL/test_cases.mqh>
#include <DRL/features.mqh>

// State and test case arrays
double CurrentState[];
TestCase TestCases[];

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
  Print("Starting LSTM inference test...");

  // Initialize test cases
  InitTestCases(TestCases);
  Print("Loaded ", ArraySize(TestCases), " test cases");

  // Initialize LSTM state array
  ArrayResize(CurrentState, 2 * LSTM_UNITS); // [hidden_state, cell_state]

  for (int i = 0; i < TEST_CASE_COUNT; i++)
  {
    Print("Testing case ", i);

    // Copy state from test case
    ArrayCopy(CurrentState, TestCases[i].lstm_state);

    // Get features from test case
    double features[];
    ArrayCopy(features, TestCases[i].features);

    // Run LSTM inference
    double output[];
    RunLSTMInference(features, CurrentState, output);

    // Get predicted action
    int predicted_action = 0;
    double max_prob = output[0];

    for (int j = 1; j < ACTION_COUNT; j++)
    {
      if (output[j] > max_prob)
      {
        max_prob = output[j];
        predicted_action = j;
      }
    }

    // Compare with expected action
    bool passed = predicted_action == TestCases[i].expected_action;
    Print("Test case ", i,
          ": Expected=", TestCases[i].expected_action,
          ", Got=", predicted_action,
          ", Result=", (passed ? "PASS" : "FAIL"));

    // Log state transitions for debugging
    string state_debug = "Final state samples: ";
    for (int k = 0; k < 3; k++)
    {
      state_debug += DoubleToString(CurrentState[k], 8);
      if (k < 2)
        state_debug += ", ";
    }
    Print(state_debug);

    // Log output probabilities
    string probs = "Action probabilities: ";
    for (int k = 0; k < ACTION_COUNT; k++)
    {
      probs += DoubleToString(output[k], 4);
      if (k < ACTION_COUNT - 1)
        probs += ", ";
    }
    Print(probs);

    if (!passed)
    {
      Print("WARNING: Test case ", i, " failed!");
    }
  }

  Print("LSTM inference test completed");
  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  // Cleanup
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick(){
    // This is a test EA, no trading functionality needed
}

    < / write_file >

    <write_file> mql5 / Experts / DRLTester.mq5

//+------------------------------------------------------------------+
//|                                                    DRLTester.mq5    |
//|                                   Copyright 2024, DRL Trading Bot   |
//|                                     https://github.com/your-repo    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link "https://github.com/your-repo"
#property version "1.00"
#property strict

#include <DRL/model.mqh>
#include <DRL/test_cases.mqh>
#include <DRL/features.mqh>

                                  // State and test case arrays
                                  double CurrentState[];
TestCase TestCases[];

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
  Print("Starting LSTM inference test...");

  // Initialize test cases
  InitTestCases(TestCases);
  Print("Loaded ", ArraySize(TestCases), " test cases");

  // Initialize LSTM state array
  ArrayResize(CurrentState, 2 * LSTM_UNITS); // [hidden_state, cell_state]

  for (int i = 0; i < TEST_CASE_COUNT; i++)
  {
    Print("Testing case ", i);

    // Copy state from test case
    ArrayCopy(CurrentState, TestCases[i].lstm_state);

    // Get features from test case
    double features[];
    ArrayCopy(features, TestCases[i].features);

    // Run LSTM inference
    double output[];
    RunLSTMInference(features, CurrentState, output);

    // Get predicted action
    int predicted_action = 0;
    double max_prob = output[0];

    for (int j = 1; j < ACTION_COUNT; j++)
    {
      if (output[j] > max_prob)
      {
        max_prob = output[j];
        predicted_action = j;
      }
    }

    // Compare with expected action
    bool passed = predicted_action == TestCases[i].expected_action;
    Print("Test case ", i,
          ": Expected=", TestCases[i].expected_action,
          ", Got=", predicted_action,
          ", Result=", (passed ? "PASS" : "FAIL"));

    // Log state transitions for debugging
    string state_debug = "Final state samples: ";
    for (int k = 0; k < 3; k++)
    {
      state_debug += DoubleToString(CurrentState[k], 8);
      if (k < 2)
        state_debug += ", ";
    }
    Print(state_debug);

    // Log output probabilities
    string probs = "Action probabilities: ";
    for (int k = 0; k < ACTION_COUNT; k++)
    {
      probs += DoubleToString(output[k], 4);
      if (k < ACTION_COUNT - 1)
        probs += ", ";
    }
    Print(probs);

    if (!passed)
    {
      Print("WARNING: Test case ", i, " failed!");
      // Log detailed feature values
      string feature_debug = "Features: ";
      for (int k = 0; k < ArraySize(features); k++)
      {
        feature_debug += DoubleToString(features[k], 6);
        if (k < ArraySize(features) - 1)
          feature_debug += ", ";
      }
      Print(feature_debug);
    }
  }

  Print("LSTM inference test completed");
  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  // Cleanup
  ArrayFree(CurrentState);
  ArrayFree(TestCases);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
  // This is a test EA, no trading functionality needed
}