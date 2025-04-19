mql5 / Experts / DRLTester.mq5

//+------------------------------------------------------------------+
//|                                                    DRLTester.mq5    |
//|                                   Copyright 2024, DRL Trading Bot   |
//|                                     https://github.com/your-repo    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link "https://github.com/your-repo"
#property version "1.00"
#property strict

#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>
#include <DRL/model.mqh>
#include <DRL/test_cases.mqh>
#include <DRL/features.mqh>
#include <DRL/matrix.mqh>
#include <DRL/weights.mqh>

                 // Global variables for test tracking
                 double g_CurrentState[];
TestCase g_TestCases[];
int g_PassedTests = 0;
int g_TotalTests = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
  Print("Starting LSTM inference test...");

  // Initialize test cases
  InitTestCases(g_TestCases);
  g_TotalTests = TEST_CASE_COUNT;
  Print("Loaded ", g_TotalTests, " test cases");

  // Initialize LSTM state array
  if (ArrayResize(g_CurrentState, 2 * LSTM_UNITS) != 2 * LSTM_UNITS)
  {
    Print("ERROR: Failed to initialize LSTM state array");
    return INIT_FAILED;
  }
  ArrayInitialize(g_CurrentState, 0);

  // Run test cases
  for (int i = 0; i < TEST_CASE_COUNT; i++)
  {
    Print("\n=== Testing case ", i, " ===");

    // Copy test case data
    double features[];
    ArrayCopy(features, g_TestCases[i].features);
    ArrayCopy(g_CurrentState, g_TestCases[i].lstm_state);

    // Log input state
    Print("Initial state sample: ",
          DoubleToString(g_CurrentState[0], 8), ", ",
          DoubleToString(g_CurrentState[1], 8));

    // Run inference
    double output[];
    RunLSTMInference(features, g_CurrentState, output);

    // Find predicted action
    int predicted = 0;
    double max_prob = output[0];
    for (int j = 1; j < ACTION_COUNT; j++)
    {
      if (output[j] > max_prob)
      {
        max_prob = output[j];
        predicted = j;
      }
    }

    // Compare with expected
    bool passed = (predicted == g_TestCases[i].expected_action);
    if (passed)
      g_PassedTests++;

    // Log results
    Print("Results: Expected=", g_TestCases[i].expected_action,
          ", Got=", predicted,
          ", ", (passed ? "PASS" : "FAIL"));

    // Log final state
    Print("Final state sample: ",
          DoubleToString(g_CurrentState[0], 8), ", ",
          DoubleToString(g_CurrentState[1], 8));

    // On failure, log full feature vector
    if (!passed)
    {
      Print("Features:");
      for (int k = 0; k < ArraySize(features); k++)
      {
        Print("  [", k, "]: ", DoubleToString(features[k], 8));
      }
    }
  }

  // Print summary
  Print("\n=== Summary ===");
  Print("Total tests: ", g_TotalTests);
  Print("Passed: ", g_PassedTests);
  Print("Success rate: ",
        DoubleToString(100.0 * g_PassedTests / g_TotalTests, 2), "%");

  return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  ArrayFree(g_CurrentState);
  ArrayFree(g_TestCases);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
  // Test EA - no trading functionality needed
}
