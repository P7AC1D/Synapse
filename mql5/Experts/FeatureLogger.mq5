//+------------------------------------------------------------------+
//|                                                FeatureLogger.mq5 |
//|                                       Feature Processing Debugger |
//+------------------------------------------------------------------+
#property copyright "DRL Trader"
#property version "1.00"
#property strict
#property description "Logs processed features for comparison with Python implementation"

// Include standard libraries
#include <Arrays\ArrayObj.mqh>
// Include NumPy-like functions
#include <MQL5\Numpy\Numpy.mqh>

// Input parameters
input int ATR_PERIOD = 14;     // ATR period
input int RSI_PERIOD = 14;     // RSI period
input int BOLL_PERIOD = 20;    // Bollinger Bands period
input int ATR_SMA_PERIOD = 20; // ATR SMA period
input bool DEBUG_MODE = true;  // Enable debug logging

// Constants
const int LOOKBACK = 20;                   // Maximum lookback period for indicators
const int MIN_BARS = 100;                  // Minimum bars needed for valid features
const double MIN_EXPECTED_ATR_RATIO = 0.5; // Min expected ATR/SMA ratio
const double MAX_EXPECTED_ATR_RATIO = 2.0; // Max expected ATR/SMA ratio

// Global variables
int handle_rsi;     // RSI indicator handle
int handle_atr;     // ATR indicator handle
int handle_bb;      // Bollinger Bands indicator handle
int handle_adx;     // ADX indicator handle
int handle_atr_sma; // ATR SMA indicator handle

// Create a numpy object
CNumpy np;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
  // Initialize indicators
  handle_rsi = iRSI(_Symbol, PERIOD_CURRENT, RSI_PERIOD, PRICE_CLOSE);
  handle_atr = iATR(_Symbol, PERIOD_CURRENT, ATR_PERIOD);
  handle_bb = iBands(_Symbol, PERIOD_CURRENT, BOLL_PERIOD, 2, 0, PRICE_CLOSE);
  handle_adx = iADX(_Symbol, PERIOD_CURRENT, ATR_PERIOD);
  handle_atr_sma = iMA(_Symbol, PERIOD_CURRENT, ATR_SMA_PERIOD, 0, MODE_SMA, handle_atr);

  if (handle_rsi == INVALID_HANDLE || handle_atr == INVALID_HANDLE ||
      handle_bb == INVALID_HANDLE || handle_adx == INVALID_HANDLE ||
      handle_atr_sma == INVALID_HANDLE)
  {
    Print("Failed to create indicator handles");
    return INIT_FAILED;
  }

  Print("FeatureLogger initialized successfully.");
  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  // Clean up indicator handles
  if (handle_rsi != INVALID_HANDLE)
    IndicatorRelease(handle_rsi);
  if (handle_atr != INVALID_HANDLE)
    IndicatorRelease(handle_atr);
  if (handle_bb != INVALID_HANDLE)
    IndicatorRelease(handle_bb);
  if (handle_adx != INVALID_HANDLE)
    IndicatorRelease(handle_adx);
  if (handle_atr_sma != INVALID_HANDLE)
    IndicatorRelease(handle_atr_sma);

  Print("FeatureLogger deinitialized.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
  // Only process features on completed bars
  static datetime last_bar_time = 0;
  datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);

  if (last_bar_time == current_bar_time)
  {
    return; // Skip if not a new bar
  }

  last_bar_time = current_bar_time;

  // Check if enough bars are available
  int bars = Bars(_Symbol, PERIOD_CURRENT);
  if (bars < MIN_BARS)
  {
    Print("Not enough historical bars available: ", bars, " (need at least ", MIN_BARS, ")");
    return;
  }

  // Process features
  ProcessAndLogFeatures();
}

//+------------------------------------------------------------------+
//| Process features and log them                                    |
//+------------------------------------------------------------------+
void ProcessAndLogFeatures()
{
  // Copy indicator values with sufficient buffer for calculations
  int num_bars = MathMin(100, Bars(_Symbol, PERIOD_CURRENT)); // Limit to 100 bars for efficiency

  // Prepare arrays
  double close[], high[], low[], open[];
  long volume[]; // Changed from double[] to long[]
  double rsi[], atr[], atr_sma[], upper_bb[], lower_bb[], adx[];

  ArraySetAsSeries(close, true);
  ArraySetAsSeries(high, true);
  ArraySetAsSeries(low, true);
  ArraySetAsSeries(open, true);
  ArraySetAsSeries(volume, true);
  ArraySetAsSeries(rsi, true);
  ArraySetAsSeries(atr, true);
  ArraySetAsSeries(atr_sma, true);
  ArraySetAsSeries(upper_bb, true);
  ArraySetAsSeries(lower_bb, true);
  ArraySetAsSeries(adx, true);

  // Copy price data
  if (CopyClose(_Symbol, PERIOD_CURRENT, 0, num_bars, close) <= 0)
    return;
  if (CopyHigh(_Symbol, PERIOD_CURRENT, 0, num_bars, high) <= 0)
    return;
  if (CopyLow(_Symbol, PERIOD_CURRENT, 0, num_bars, low) <= 0)
    return;
  if (CopyOpen(_Symbol, PERIOD_CURRENT, 0, num_bars, open) <= 0)
    return;
  if (CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, num_bars, volume) <= 0)
    return;

  // Copy indicator values
  if (CopyBuffer(handle_rsi, 0, 0, num_bars, rsi) <= 0)
    return;
  if (CopyBuffer(handle_atr, 0, 0, num_bars, atr) <= 0)
    return;
  if (CopyBuffer(handle_atr_sma, 0, 0, num_bars, atr_sma) <= 0)
    return;
  if (CopyBuffer(handle_bb, 1, 0, num_bars, upper_bb) <= 0)
    return; // Upper band = 1
  if (CopyBuffer(handle_bb, 2, 0, num_bars, lower_bb) <= 0)
    return; // Lower band = 2
  if (CopyBuffer(handle_adx, 0, 0, num_bars, adx) <= 0)
    return; // ADX line = 0

  // Use the last completed bar (index 1), not the current incomplete bar (index 0)
  int idx = 1;

  // Make sure we have enough bars
  if (num_bars <= idx)
  {
    Print("Not enough bars for processing (need at least ", idx + 1, ")");
    return;
  }

  // ===== Using NumPy-like functions for calculations =====

  // Calculate returns - MATCHING PYTHON EXACTLY
  double returns = 0.0;
  if (idx < num_bars - 1)
  {
    // Python: returns = np.diff(close) / close[:-1]
    // In MQL5 with reversed arrays: (close[idx-1] - close[idx]) / close[idx]
    returns = (close[idx - 1] - close[idx]) / close[idx];
  }
  // Clip returns to [-0.1, 0.1] range
  returns = MathMin(MathMax(returns, -0.1), 0.1);

  // Normalize RSI from [0, 100] to [-1, 1] just like Python
  double rsi_norm = rsi[idx] / 50.0 - 1.0;

  // Normalize ATR (relative to its own moving average)
  double atr_ratio = atr[idx] / (atr_sma[idx] + 1e-8);
  double atr_norm = 2.0 * (atr_ratio - MIN_EXPECTED_ATR_RATIO) / (MAX_EXPECTED_ATR_RATIO - MIN_EXPECTED_ATR_RATIO) - 1.0;
  atr_norm = MathMin(MathMax(atr_norm, -1.0), 1.0); // clip to [-1, 1]

  // Volatility breakout feature - MATCHING PYTHON
  double band_range = upper_bb[idx] - lower_bb[idx];
  band_range = band_range < 1e-8 ? 1e-8 : band_range;
  double position = close[idx] - lower_bb[idx];
  double volatility_breakout = position / band_range;
  volatility_breakout = MathMin(MathMax(volatility_breakout, 0.0), 1.0); // clip to [0, 1]
  // Convert to [-1, 1] range for comparison purposes
  double volatility_breakout_norm = volatility_breakout * 2.0 - 1.0;
  volatility_breakout_norm = MathMin(MathMax(volatility_breakout_norm, -1.0), 1.0);

  // Trend strength from ADX - MATCHING PYTHON
  double trend_strength = MathMin(MathMax(adx[idx] / 25.0 - 1.0, -1.0), 1.0);

  // Candle pattern - MATCHING PYTHON EXACTLY
  double body = close[idx] - open[idx];
  double upper_wick = high[idx] - MathMax(close[idx], open[idx]);
  double lower_wick = MathMin(close[idx], open[idx]) - low[idx];
  double range = MathMax(high[idx] - low[idx], 1e-8);

  double body_ratio = body / range;
  double wick_ratio = 0.0;
  if (upper_wick + lower_wick > 1e-8)
  {
    wick_ratio = (upper_wick - lower_wick) / (upper_wick + lower_wick);
  }
  double candle_pattern = (body_ratio + wick_ratio) / 2.0;
  candle_pattern = MathMin(MathMax(candle_pattern, -1.0), 1.0); // clip to [-1, 1]

  // Time encoding features - MATCHING PYTHON EXACTLY
  datetime time = iTime(_Symbol, PERIOD_CURRENT, idx);
  MqlDateTime dt;
  TimeToStruct(time, dt);
  int minutes_in_day = 24 * 60;
  int time_index = dt.hour * 60 + dt.min;
  double sin_time = MathSin(2.0 * M_PI * time_index / minutes_in_day);
  double cos_time = MathCos(2.0 * M_PI * time_index / minutes_in_day);

  // Volume change - MATCHING PYTHON EXACTLY
  double volume_pct = 0;
  if (idx < num_bars - 1 && volume[idx + 1] > 0)
  {
    volume_pct = ((double)volume[idx] - (double)volume[idx + 1]) / (double)volume[idx + 1];
  }
  volume_pct = MathMin(MathMax(volume_pct, -1.0), 1.0); // clip to [-1, 1]

  // Log feature values with timestamp for easy comparison
  Print("==== Features at ", TimeToString(time, TIME_DATE | TIME_MINUTES | TIME_SECONDS), " ====");
  Print("returns = ", DoubleToString(returns, 8), " | [-0.1, 0.1]");
  Print("rsi = ", DoubleToString(rsi_norm, 8), " | [-1, 1] (raw: ", DoubleToString(rsi[idx], 2), ")");
  Print("atr = ", DoubleToString(atr_norm, 8), " | [-1, 1] (raw: ", DoubleToString(atr[idx], 8), ", ratio: ", DoubleToString(atr_ratio, 8), ")");
  Print("volatility_breakout = ", DoubleToString(volatility_breakout, 8), " | [0, 1]");
  Print("volatility_breakout_norm = ", DoubleToString(volatility_breakout_norm, 8), " | [-1, 1]");
  Print("trend_strength = ", DoubleToString(trend_strength, 8), " | [-1, 1] (raw ADX: ", DoubleToString(adx[idx], 2), ")");
  Print("candle_pattern = ", DoubleToString(candle_pattern, 8), " | [-1, 1]");
  Print("sin_time = ", DoubleToString(sin_time, 8), " | [-1, 1]");
  Print("cos_time = ", DoubleToString(cos_time, 8), " | [-1, 1]");
  Print("volume_change = ", DoubleToString(volume_pct, 8), " | [-1, 1]");

  // Print comparisons to Python features
  datetime current_time = TimeCurrent();
  Print("\nMQL5 vs Python comparison at ", TimeToString(current_time, TIME_DATE | TIME_MINUTES | TIME_SECONDS));
  Print("Feature         | MQL5      | Python    | Status");
  Print("----------------|-----------|-----------|-------");

  // Get Python values from most recent run
  double py_returns = -0.000788;
  double py_rsi = -0.095425;
  double py_atr = 1.000000;
  double py_vol_bo = -1.000000;
  double py_trend = -0.833975;
  double py_candle = 0.946930;
  double py_sin = 0.321439;
  double py_cos = 0.128239;
  double py_vol_chg = 0.366211;

  // Calculate differences
  double diff_returns = MathAbs(returns - py_returns);
  double diff_rsi = MathAbs(rsi_norm - py_rsi);
  double diff_atr = MathAbs(atr_norm - py_atr);
  double diff_vol_bo = MathAbs(volatility_breakout_norm - py_vol_bo);
  double diff_trend = MathAbs(trend_strength - py_trend);
  double diff_candle = MathAbs(candle_pattern - py_candle);
  double diff_sin = MathAbs(sin_time - py_sin);
  double diff_cos = MathAbs(cos_time - py_cos);
  double diff_vol_chg = MathAbs(volume_pct - py_vol_chg);

  // Status check (✓ if difference < 0.05, otherwise ✗)
  Print("returns        | ", DoubleToString(returns, 6), " | ", DoubleToString(py_returns, 6), " | ", (diff_returns < 0.05 ? "✓" : "✗"));
  Print("rsi            | ", DoubleToString(rsi_norm, 6), " | ", DoubleToString(py_rsi, 6), " | ", (diff_rsi < 0.05 ? "✓" : "✗"));
  Print("atr            | ", DoubleToString(atr_norm, 6), " | ", DoubleToString(py_atr, 6), " | ", (diff_atr < 0.05 ? "✓" : "✗"));
  Print("volatility_bo  | ", DoubleToString(volatility_breakout_norm, 6), " | ", DoubleToString(py_vol_bo, 6), " | ", (diff_vol_bo < 0.05 ? "✓" : "✗"));
  Print("trend_strength | ", DoubleToString(trend_strength, 6), " | ", DoubleToString(py_trend, 6), " | ", (diff_trend < 0.05 ? "✓" : "✗"));
  Print("candle_pattern | ", DoubleToString(candle_pattern, 6), " | ", DoubleToString(py_candle, 6), " | ", (diff_candle < 0.05 ? "✓" : "✗"));
  Print("sin_time       | ", DoubleToString(sin_time, 6), " | ", DoubleToString(py_sin, 6), " | ", (diff_sin < 0.05 ? "✓" : "✗"));
  Print("cos_time       | ", DoubleToString(cos_time, 6), " | ", DoubleToString(py_cos, 6), " | ", (diff_cos < 0.05 ? "✓" : "✗"));
  Print("volume_change  | ", DoubleToString(volume_pct, 6), " | ", DoubleToString(py_vol_chg, 6), " | ", (diff_vol_chg < 0.05 ? "✓" : "✗"));

  // Log feature values in comma-separated format for easy copying
  string csv_format = StringFormat("%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f",
                                   TimeToString(time, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                                   returns, rsi_norm, atr_norm, volatility_breakout,
                                   trend_strength, candle_pattern, sin_time, cos_time, volume_pct);

  Print("CSV Format: ", csv_format);
}