//+------------------------------------------------------------------+
//|                                                FeatureLogger.mq5 |
//|                                       Feature Processing Debugger |
//+------------------------------------------------------------------+
#property copyright "DRL Trader"
#property version   "1.00"
#property strict
#property description "Logs processed features for comparison with Python implementation"

// Include standard libraries
#include <Arrays\ArrayObj.mqh>

// Input parameters
input int ATR_PERIOD = 14;             // ATR period
input int RSI_PERIOD = 14;             // RSI period
input int BOLL_PERIOD = 20;            // Bollinger Bands period
input int ATR_SMA_PERIOD = 20;         // ATR SMA period
input bool DEBUG_MODE = true;          // Enable debug logging

// Constants
const int LOOKBACK = 20;               // Maximum lookback period for indicators
const int MIN_BARS = 100;              // Minimum bars needed for valid features
const double MIN_EXPECTED_ATR_RATIO = 0.5;  // Min expected ATR/SMA ratio
const double MAX_EXPECTED_ATR_RATIO = 2.0;  // Max expected ATR/SMA ratio

// Global variables
int handle_rsi;                        // RSI indicator handle
int handle_atr;                        // ATR indicator handle
int handle_bb;                         // Bollinger Bands indicator handle  
int handle_adx;                        // ADX indicator handle
int handle_atr_sma;                    // ATR SMA indicator handle

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
   
   if(handle_rsi == INVALID_HANDLE || handle_atr == INVALID_HANDLE || 
      handle_bb == INVALID_HANDLE || handle_adx == INVALID_HANDLE ||
      handle_atr_sma == INVALID_HANDLE) {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   Print("FeatureLogger initialized successfully.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Clean up indicator handles
   if(handle_rsi != INVALID_HANDLE) IndicatorRelease(handle_rsi);
   if(handle_atr != INVALID_HANDLE) IndicatorRelease(handle_atr);
   if(handle_bb != INVALID_HANDLE) IndicatorRelease(handle_bb);
   if(handle_adx != INVALID_HANDLE) IndicatorRelease(handle_adx);
   if(handle_atr_sma != INVALID_HANDLE) IndicatorRelease(handle_atr_sma);
   
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
   
   if(last_bar_time == current_bar_time) {
      return;  // Skip if not a new bar
   }
   
   last_bar_time = current_bar_time;
   
   // Check if enough bars are available
   int bars = Bars(_Symbol, PERIOD_CURRENT);
   if(bars < MIN_BARS) {
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
   long volume[];    // Changed from double[] to long[]
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
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, num_bars, close) <= 0) return;
   if(CopyHigh(_Symbol, PERIOD_CURRENT, 0, num_bars, high) <= 0) return;
   if(CopyLow(_Symbol, PERIOD_CURRENT, 0, num_bars, low) <= 0) return;
   if(CopyOpen(_Symbol, PERIOD_CURRENT, 0, num_bars, open) <= 0) return;
   if(CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, num_bars, volume) <= 0) return;
   
   // Copy indicator values
   if(CopyBuffer(handle_rsi, 0, 0, num_bars, rsi) <= 0) return;
   if(CopyBuffer(handle_atr, 0, 0, num_bars, atr) <= 0) return;
   if(CopyBuffer(handle_atr_sma, 0, 0, num_bars, atr_sma) <= 0) return;
   if(CopyBuffer(handle_bb, 1, 0, num_bars, upper_bb) <= 0) return;  // Upper band = 1
   if(CopyBuffer(handle_bb, 2, 0, num_bars, lower_bb) <= 0) return;  // Lower band = 2
   if(CopyBuffer(handle_adx, 0, 0, num_bars, adx) <= 0) return;      // ADX line = 0
   
   // Use the last completed bar (index 1), not the current incomplete bar (index 0)
   int idx = 1;
   
   // Make sure we have enough bars
   if(num_bars <= idx) {
      Print("Not enough bars for processing (need at least ", idx + 1, ")");
      return;
   }
   
   // Calculate returns - IMPORTANT: Use the right formula that matches Python
   // Python: returns = np.diff(close) / close[:-1]
   double returns = 0.0;
   if(idx < num_bars-1) {
      // In Python: returns[i] = (close[i+1] - close[i]) / close[i]
      // With ArraySetAsSeries=true, idx=1 is the last completed bar
      returns = (close[idx] - close[idx+1]) / close[idx+1];
   }
   returns = MathMin(MathMax(returns, -0.1), 0.1); // clip to [-0.1, 0.1]
   
   // Normalize RSI from [0, 100] to [-1, 1] - match Python exactly
   double rsi_norm = rsi[idx] / 50.0 - 1.0;
   
   // Normalize ATR (relative to its own moving average)
   double atr_ratio = atr[idx] / (atr_sma[idx] + 1e-8);
   // Fix scaling to match Python using the MIN/MAX_EXPECTED_ATR_RATIO constants
   double atr_norm = 2.0 * (atr_ratio - MIN_EXPECTED_ATR_RATIO) / (MAX_EXPECTED_ATR_RATIO - MIN_EXPECTED_ATR_RATIO) - 1.0;
   atr_norm = MathMin(MathMax(atr_norm, -1.0), 1.0); // clip to [-1, 1]
   
   // Volume change - match Python's calculation exactly
   // Python: volume_pct[1:] = np.diff(volume) / volume[:-1]
   double volume_pct = 0;
   if(idx < num_bars-1 && volume[idx+1] > 0) {
      // Calculate volume change exactly like in Python
      // Cast to double before division to prevent loss of precision
      volume_pct = ((double)volume[idx] - (double)volume[idx+1]) / (double)volume[idx+1];
   }
   volume_pct = MathMin(MathMax(volume_pct, -1.0), 1.0); // clip to [-1, 1]
   
   // Volatility breakout feature - match Python exactly
   // In Python: position = close - lower_band; volatility_breakout = position / band_range
   double band_range = upper_bb[idx] - lower_bb[idx];
   band_range = band_range < 1e-8 ? 1e-8 : band_range;
   double position = close[idx] - lower_bb[idx];
   double volatility_breakout = position / band_range;
   volatility_breakout = MathMin(MathMax(volatility_breakout, 0.0), 1.0); // clip to [0, 1] to match Python
   
   // Trend strength from ADX - match Python exactly
   // Python: trend_strength = np.clip(adx/25 - 1, -1, 1)
   double trend_strength = MathMin(MathMax(adx[idx]/25.0 - 1.0, -1.0), 1.0); // clip to [-1, 1]
   
   // Candle pattern - match Python's calculation exactly
   double body = close[idx] - open[idx];
   double upper_wick = high[idx] - MathMax(close[idx], open[idx]);
   double lower_wick = MathMin(close[idx], open[idx]) - low[idx];
   double range = MathMax(high[idx] - low[idx], 1e-8);
   
   // Fix: Use separate components then average them
   double body_ratio = body / range;
   double wick_ratio = 0.0;
   if(upper_wick + lower_wick > 1e-8) {
      wick_ratio = (upper_wick - lower_wick) / (upper_wick + lower_wick);
   }
   double candle_pattern = (body_ratio + wick_ratio) / 2.0;
   candle_pattern = MathMin(MathMax(candle_pattern, -1.0), 1.0); // clip to [-1, 1]
   
   // Time encoding features
   datetime time = iTime(_Symbol, PERIOD_CURRENT, idx);
   MqlDateTime dt;
   TimeToStruct(time, dt);
   int minutes_in_day = 24 * 60;
   int time_index = dt.hour * 60 + dt.min;
   double sin_time = MathSin(2 * M_PI * time_index / minutes_in_day);
   double cos_time = MathCos(2 * M_PI * time_index / minutes_in_day);
   
   // Log feature values with timestamp for easy comparison
   Print("==== Features at ", TimeToString(time, TIME_DATE|TIME_MINUTES|TIME_SECONDS), " ====");
   // Match exactly the order in Python's get_feature_names() method:
   Print("returns = ", DoubleToString(returns, 8), " | [-0.1, 0.1]");
   Print("rsi = ", DoubleToString(rsi_norm, 8), " | [-1, 1] (raw: ", DoubleToString(rsi[idx], 2), ")");
   Print("atr = ", DoubleToString(atr_norm, 8), " | [-1, 1] (raw: ", DoubleToString(atr[idx], 8), ", ratio: ", DoubleToString(atr_ratio, 8), ")");
   Print("volume_change = ", DoubleToString(volume_pct, 8), " | [-1, 1]");
   Print("volatility_breakout = ", DoubleToString(volatility_breakout, 8), " | [0, 1]");
   Print("trend_strength = ", DoubleToString(trend_strength, 8), " | [-1, 1] (raw ADX: ", DoubleToString(adx[idx], 2), ")");
   Print("candle_pattern = ", DoubleToString(candle_pattern, 8), " | [-1, 1]");
   Print("sin_time = ", DoubleToString(sin_time, 8), " | [-1, 1]");
   Print("cos_time = ", DoubleToString(cos_time, 8), " | [-1, 1]");
}