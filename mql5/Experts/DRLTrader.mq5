//+------------------------------------------------------------------+
//|                                                    DRLTrader.mq5     |
//|                                   Copyright 2024, DRL Trading Bot    |
//|                                     https://github.com/your-repo     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"
#property version   "1.00"
#property strict

// Include required files
#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>
#include <DRL/RecurrentPPOModel.mqh>

// Constants
#define MAGIC_NUMBER 20240417

// Input parameters
input string ModelGroup = ">>> Model Settings <<<";
input string ModelPath = "XAUUSDm.onnx";  // Path to ONNX model (relative to terminal data folder)
input int SequenceLength = 500;                    // Number of bars for sequence input
input int LSTMHiddenSize = 256;                    // LSTM hidden state size, changed from 64 to 256
input int LSTMNumLayers = 2;                       // Number of LSTM layers, changed from 1 to 2
input int NumFeatures = 11;                        // Number of features per bar
input int NumActions = 4;                          // Number of possible actions

input string DataGroup = ">>> Data Settings <<<";
input int MinDataBars = 500;                       // Minimum data bars to collect

input string TradingGroup = ">>> Trading Settings <<<";
input string Label = "PPO_LSTM_EA";                // Label for the EA
input int MaxSpread = 350;                         // Maximum allowed spread (points)
input double StopLoss = 2500.0;                     // Stop loss in points
input double BalancePerLot = 2500.0;               // Amount required per 0.01 lot
input bool UseFixedLotSize = false;                // Use fixed lot size instead of calculating from balance
input double FixedLotSize = 0.01;                  // Fixed lot size if UseFixedLotSize is true
input bool FallbackToManualMode = true;            // If true, will not trade automatically when ONNX fails

// Indicators
int rsi_handle;
int atr_handle;
int macd_handle;
int bb_handle;
int adx_handle;

// Indicator parameters
int atr_period = 14;
int rsi_period = 14;
int boll_period = 20;
int adx_period = 14;

// Global variables
CTrade Trade;                   // Trading object
RecurrentPPOModel Model;        // DRL Model
string last_error = "";         // Last error message
bool onnx_available = false;    // Flag indicating if ONNX runtime is available

// Position tracking
struct Position {
    int direction;      // 1 for long, -1 for short, 0 for none
    double entryPrice;  // Position entry price
    double lotSize;     // Position size in lots
    datetime entryTime; // Entry timestamp
    int entryBar;       // Bar index when position was entered
};

Position CurrentPosition;

// Data arrays
double open_prices[];
double high_prices[];
double low_prices[];
double close_prices[];
double spread_values[];
long volume_values[];
datetime time_values[];

// Model input data
float model_input_data[];

// Indicators values
double rsi_values[];
double atr_values[];
double upper_band_values[];
double lower_band_values[];
double adx_values[];

//+------------------------------------------------------------------+
//| Initialize indicators                                              |
//+------------------------------------------------------------------+
bool InitializeIndicators() {
    // Initialize standard MT5 indicators
    rsi_handle = iRSI(_Symbol, _Period, rsi_period, PRICE_CLOSE);
    atr_handle = iATR(_Symbol, _Period, atr_period);
    macd_handle = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
    bb_handle = iBands(_Symbol, _Period, boll_period, 0, 2, PRICE_CLOSE);
    adx_handle = iADX(_Symbol, _Period, adx_period);
    
    return rsi_handle != INVALID_HANDLE && 
           atr_handle != INVALID_HANDLE && 
           bb_handle != INVALID_HANDLE &&
           macd_handle != INVALID_HANDLE &&
           adx_handle != INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Release indicators                                                |
//+------------------------------------------------------------------+
void ReleaseIndicators() {
    IndicatorRelease(rsi_handle);
    IndicatorRelease(atr_handle);
    IndicatorRelease(macd_handle);
    IndicatorRelease(bb_handle);
    IndicatorRelease(adx_handle);
}

//+------------------------------------------------------------------+
//| Calculate stop loss price                                          |
//+------------------------------------------------------------------+
double CalculateStopLoss(const double entryPrice, const bool isBuy) {
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    
    // For XAUUSD, 1 pip = 0.1 points
    double pipValue = StringFind(_Symbol, "XAU") >= 0 ? point * 10 : point;
    
    // Calculate stop loss price
    double slPrice = isBuy ? 
                    entryPrice - (StopLoss * pipValue) : 
                    entryPrice + (StopLoss * pipValue);
    
    return NormalizeDouble(slPrice, digits);
}

//+------------------------------------------------------------------+
//| Calculate position size                                            |
//+------------------------------------------------------------------+
double CalculateLotSize() {
    if(UseFixedLotSize) {
        return FixedLotSize;
    }

    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    double lotSize = (balance / BalancePerLot) * minLot;
    lotSize = MathRound(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Collect historical data                                           |
//+------------------------------------------------------------------+
bool CollectHistoricalData(int bars_to_collect) {
    // Initialize arrays with series set to false to match Python's oldest-to-newest order
    ArrayResize(open_prices, bars_to_collect);
    ArrayResize(high_prices, bars_to_collect);
    ArrayResize(low_prices, bars_to_collect);
    ArrayResize(close_prices, bars_to_collect);
    ArrayResize(spread_values, bars_to_collect);
    ArrayResize(volume_values, bars_to_collect);
    ArrayResize(time_values, bars_to_collect);
    
    ArraySetAsSeries(open_prices, false);
    ArraySetAsSeries(high_prices, false);
    ArraySetAsSeries(low_prices, false);
    ArraySetAsSeries(close_prices, false);
    ArraySetAsSeries(spread_values, false);
    ArraySetAsSeries(volume_values, false);
    ArraySetAsSeries(time_values, false);
    
    // Use CopyXXX functions with start=1 to skip current incomplete bar
    if (CopyOpen(_Symbol, _Period, 1, bars_to_collect, open_prices) != bars_to_collect) return false;
    if (CopyHigh(_Symbol, _Period, 1, bars_to_collect, high_prices) != bars_to_collect) return false;
    if (CopyLow(_Symbol, _Period, 1, bars_to_collect, low_prices) != bars_to_collect) return false;
    if (CopyClose(_Symbol, _Period, 1, bars_to_collect, close_prices) != bars_to_collect) return false;
    if (CopyTickVolume(_Symbol, _Period, 1, bars_to_collect, volume_values) != bars_to_collect) return false;
    if (CopyTime(_Symbol, _Period, 1, bars_to_collect, time_values) != bars_to_collect) return false;
    
    // Log last 5 bars for debugging - reversing order for display
    Print("Last 5 bars (newest to oldest):");
    Print("", "Historical Data Collection (Python-aligned):");
    Print(StringFormat("Total bars collected: %d (start=1 to skip current bar)", bars_to_collect));
    Print("Array Index Direction:");
    Print("  [0] = Oldest data   -> Time: ", TimeToString(time_values[0]));
    Print("  ..."); 
    Print(StringFormat("  [%d] = Newest data -> Time: %s", 
          bars_to_collect-1, 
          TimeToString(time_values[bars_to_collect-1])));
    Print("");
    Print("Data Organization:");
    Print("  1. Current bar excluded (incomplete)");
    Print("  2. Arrays ordered oldest -> newest");
    Print("  3. Features calculated in same order");
    Print("");
    
    Print("Last 5 completed bars (newest to oldest):");
    for(int i = 0; i < 5 && i < bars_to_collect; i++) {
        int idx = bars_to_collect - 1 - i;
        datetime bar_time = time_values[idx];
        MqlDateTime time_struct;
        TimeToStruct(bar_time, time_struct);
        
        Print(StringFormat(
            "Bar[%d] - Time: %s [%02d:%02d], O: %.5f, H: %.5f, L: %.5f, C: %.5f, V: %d", 
            i, TimeToString(bar_time), time_struct.hour, time_struct.min,
            open_prices[idx], high_prices[idx], low_prices[idx], close_prices[idx], 
            volume_values[idx]
        ));
    }
    Print("");
    
    // Calculate spread values (as points)
    for (int i = 0; i < bars_to_collect; i++) {
        spread_values[i] = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    }
    
    // Check terminal timezone settings
    MqlDateTime terminal_time;
    datetime current_time = TimeCurrent();
    TimeToStruct(current_time, terminal_time);
    
    // Get timezone info
    datetime server_time = TimeTradeServer();
    MqlDateTime server_time_struct;
    TimeToStruct(server_time, server_time_struct);
    
    int local_hour = terminal_time.hour;
    int server_hour = server_time_struct.hour;
    int hour_diff = server_hour - local_hour;
    
    Print("DEBUG: Timezone info - Local time: ", TimeLocal(),
          ", Server time: ", server_time,
          ", Hour difference: ", hour_diff);
          
    return true;
}

//+------------------------------------------------------------------+
//| Prepare model input data                                          |
//+------------------------------------------------------------------+
bool PrepareModelInput() {
    int sequence_length = MathMin(SequenceLength, ArraySize(close_prices));
    
    // Resize model input data array based on sequence length and feature count
    ArrayResize(model_input_data, sequence_length * NumFeatures);
    ArrayInitialize(model_input_data, 0.0);
    
    // Calculate additional indicators needed for features
    ArrayResize(rsi_values, sequence_length);
    ArrayResize(atr_values, sequence_length);
    ArrayResize(upper_band_values, sequence_length);
    ArrayResize(lower_band_values, sequence_length);
    ArrayResize(adx_values, sequence_length);
    
    ArraySetAsSeries(rsi_values, true);
    ArraySetAsSeries(atr_values, true);
    ArraySetAsSeries(upper_band_values, true);
    ArraySetAsSeries(lower_band_values, true);
    ArraySetAsSeries(adx_values, true);
    
    // Set all arrays as non-series to match Python's oldest-to-newest order
    ArraySetAsSeries(rsi_values, false);
    ArraySetAsSeries(atr_values, false);
    ArraySetAsSeries(upper_band_values, false);
    ArraySetAsSeries(lower_band_values, false);
    ArraySetAsSeries(adx_values, false);
    
    // Match Python's data order - use start=1 to align with price data
    if(CopyBuffer(rsi_handle, 0, 1, sequence_length, rsi_values) != sequence_length) return false;
    if(CopyBuffer(atr_handle, 0, 1, sequence_length, atr_values) != sequence_length) return false;
    if(CopyBuffer(bb_handle, 1, 1, sequence_length, upper_band_values) != sequence_length) return false;
    if(CopyBuffer(bb_handle, 2, 1, sequence_length, lower_band_values) != sequence_length) return false;
    if(CopyBuffer(adx_handle, 0, 1, sequence_length, adx_values) != sequence_length) return false;
    
    // Log indicator values for the last 5 bars - displaying newest to oldest but accessing in order
    Print("Last 5 bars indicator values (newest to oldest):");
    for(int i = 0; i < 5 && i < sequence_length; i++) {
        int idx = sequence_length - 1 - i;  // Index for newest-to-oldest display
        Print(StringFormat(
            "Bar[%d] - RSI: %.1f (raw %.1f), ATR: %.5f, BB Upper: %.5f, BB Lower: %.5f, ADX: %.1f",
            i, rsi_values[idx] / 50.0 - 1.0, rsi_values[idx],
            atr_values[idx], upper_band_values[idx], lower_band_values[idx],
            adx_values[idx]
        ));
    }
    
    Print(""); // Add separator line
    
    // Find min/max for normalization
    double min_price = DBL_MAX;
    double max_price = DBL_MIN;
    double min_volume = DBL_MAX;
    double max_volume = DBL_MIN;
    double min_spread = DBL_MAX;
    double max_spread = DBL_MIN;
    double min_atr = DBL_MAX;
    double max_atr = DBL_MIN;
    
    for(int i = 0; i < sequence_length; i++) {
        // Find min/max price
        min_price = MathMin(min_price, MathMin(open_prices[i], MathMin(high_prices[i], MathMin(low_prices[i], close_prices[i]))));
        max_price = MathMax(max_price, MathMax(open_prices[i], MathMax(high_prices[i], MathMax(low_prices[i], close_prices[i]))));
        
        // Find min/max volume
        min_volume = MathMin(min_volume, (double)volume_values[i]);
        max_volume = MathMax(max_volume, (double)volume_values[i]);
        
        // Find min/max spread
        min_spread = MathMin(min_spread, spread_values[i]);
        max_spread = MathMax(max_spread, spread_values[i]);
        
        // Find min/max ATR
        min_atr = MathMin(min_atr, atr_values[i]);
        max_atr = MathMax(max_atr, atr_values[i]);
    }
    
    // Avoid division by zero
    if(max_price == min_price) max_price = min_price + 1;
    if(max_volume == min_volume) max_volume = min_volume + 1;
    if(max_spread == min_spread) max_spread = min_spread + 1;
    if(max_atr == min_atr) max_atr = min_atr + 1;
    
    // Calculate ATR moving average (20-period) for ATR ratio feature - matching Python's order
    double atr_sma[];
    ArrayResize(atr_sma, sequence_length);
    ArraySetAsSeries(atr_sma, false);  // Match Python's oldest to newest order
    
    for(int i = 0; i < sequence_length; i++) {
        double sum = 0.0;
        int count = 0;
        // Look back for SMA, not forward
        for(int j = MathMax(0, i - 19); j <= i; j++) {
            sum += atr_values[j];
            count++;
        }
        atr_sma[i] = sum / count;
    }
    
    // Create array to store feature values for logging
    double feature_values[11];
    
    // Fill model input array with all 11 features
    for(int i = 0; i < sequence_length; i++) {
        int idx = i * NumFeatures;
        
        // Feature 0: Returns (match Python's np.diff approach)
        double returns = 0.0;
        if(i > 0) {
            returns = (close_prices[i] - close_prices[i-1]) / close_prices[i-1];
            // Scale up to match Python's calculation
            returns *= 10.0;  // Adjust scaling to match Python values
        }
        
        // Debug the returns calculation for the latest bar
        if(i == sequence_length - 1) {
            Print("DEBUG: Returns calculation - Current close:", close_prices[i],
                  ", Previous close:", i > 0 ? close_prices[i-1] : 0,
                  ", Returns:", returns);
        }
        
        returns = MathMin(MathMax(returns, -0.1), 0.1);
        model_input_data[idx + 0] = (float)returns;
        feature_values[0] = returns;
        
        // Feature 1: RSI normalized to [-1, 1] (fix sign issue)
        if(i == sequence_length - 1) {
            Print("DEBUG: Raw RSI value before normalization: ", rsi_values[i]);
        }
        
        // Match Python's RSI normalization: rsi/50 - 1
        double normalized_rsi = rsi_values[i] / 50.0 - 1.0;
        normalized_rsi = MathMin(MathMax(normalized_rsi, -1.0), 1.0);  // Ensure bounds
        model_input_data[idx + 1] = (float)normalized_rsi;
        feature_values[1] = normalized_rsi;
        
        if(i == sequence_length - 1) {
            Print("DEBUG: RSI calculation - Raw: ", rsi_values[i],
                  ", Division: ", rsi_values[i] / 50.0,
                  ", Final: ", normalized_rsi);
        }
        
        // Feature 2: ATR ratio normalized like Python
        double atr_ratio = 1.0;  // Default
        if(atr_sma[i] > 0) {
            atr_ratio = atr_values[i] / (atr_sma[i] + 1e-8);  // Add epsilon like Python
            // Scale from typical range [0.5, 2.0] to [-1, 1]
            double min_ratio = 0.5;
            double max_ratio = 2.0;
            double range = max_ratio - min_ratio;
            atr_ratio = 2.0 * ((atr_ratio - min_ratio) / range) - 1.0;
            atr_ratio = MathMin(MathMax(atr_ratio, -1.0), 1.0);  // Ensure bounds
        }
        
        if(i == sequence_length - 1) {
            Print("DEBUG: ATR calculation - Value:", atr_values[i],
                  ", SMA:", atr_sma[i],
                  ", Raw ratio:", atr_values[i] / (atr_sma[i] + 1e-8),
                  ", Normalized:", atr_ratio);
        }
        
        model_input_data[idx + 2] = (float)atr_ratio;
        feature_values[2] = atr_ratio;
        
        // Feature 3: Volume change - exact Python implementation
        double volume_pct = 0.0;
        if(i > 0 && volume_values[i-1] > 0) {
            double current_vol = (double)volume_values[i];
            double prev_vol = (double)volume_values[i-1];
            volume_pct = (current_vol - prev_vol) / prev_vol;
            if(i == sequence_length - 1) {
                Print("DEBUG: Volume calc - current:", current_vol, 
                      ", previous:", prev_vol,
                      ", Change:", volume_pct);
            }
        }
        
        volume_pct = MathMin(MathMax(volume_pct, -1.0), 1.0);
        model_input_data[idx + 3] = (float)volume_pct;
        feature_values[3] = volume_pct;
        
        // Feature 4: Volatility breakout with exact Python ranges
        double band_range = upper_band_values[i] - lower_band_values[i];
        band_range = band_range < 1e-8 ? 1e-8 : band_range;  // Match Python's epsilon
        double position = close_prices[i] - lower_band_values[i];
        
        // Match Python's calculation: clip to [0,1] range
        double volatility_breakout = position / band_range;  // Calculate ratio
        volatility_breakout = MathMin(MathMax(volatility_breakout, 0.0), 1.0);  // Clip to [0,1] like Python
        
        // Debug the BB calculation for the latest bar
        if(i == sequence_length - 1) {
            Print("DEBUG: BB values - Upper:", upper_band_values[i], 
                  ", Lower:", lower_band_values[i],
                  ", Close:", close_prices[i],
                  ", Position from lower:", position,
                  ", Band Range:", band_range,
                  ", Breakout Value:", volatility_breakout);
        }
        model_input_data[idx + 4] = (float)volatility_breakout;
        feature_values[4] = volatility_breakout;
        
        // Feature 5: Trend strength - clip to Python's [-1,1] range
        double trend_strength = adx_values[i] / 25.0 - 1.0;
        trend_strength = MathMin(MathMax(trend_strength, -1.0), 1.0);  // Match Python's clipping
        model_input_data[idx + 5] = (float)trend_strength;
        feature_values[5] = trend_strength;
        
        // Feature 6: Candle pattern - match Python's exact formula
        double body = close_prices[i] - open_prices[i];
        double upper_wick = high_prices[i] - MathMax(close_prices[i], open_prices[i]);
        double lower_wick = MathMin(close_prices[i], open_prices[i]) - low_prices[i];
        double range = high_prices[i] - low_prices[i] + 1e-8;
        double candle_pattern = (body/range + (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2.0;  // Remove negative sign
        candle_pattern = MathMin(MathMax(candle_pattern, -1.0), 1.0);
        
        if(i == sequence_length - 1) {
            Print("DEBUG: Candle pattern - Body:", body,
                  ", Upper wick:", upper_wick,
                  ", Lower wick:", lower_wick,
                  ", Range:", range,
                  ", Pattern:", candle_pattern);
        }
        model_input_data[idx + 6] = (float)candle_pattern;
        feature_values[6] = candle_pattern;
        
        // Feature 7-8: Simple time encoding matching Python exactly
        MqlDateTime time_struct;
        TimeToStruct(time_values[i], time_struct);
        
        // Direct time calculation without any timezone shifts
        int minutes_in_day = 24 * 60;
        int minutes_since_midnight = time_struct.hour * 60 + time_struct.min;
        double angle = 2.0 * M_PI * minutes_since_midnight / minutes_in_day;
        double sin_time = MathSin(angle);
        double cos_time = MathCos(angle);
        
        // Debug time calculations for the latest bar
        if(i == sequence_length - 1) {
            Print("DEBUG: Time encoding - Time:", time_values[i],
                  ", Hour:", time_struct.hour,
                  ", Minutes since midnight:", minutes_since_midnight,
                  ", Sin:", sin_time,
                  ", Cos:", cos_time);
        }
        
        model_input_data[idx + 7] = (float)sin_time;
        model_input_data[idx + 8] = (float)cos_time;
        feature_values[7] = sin_time;
        feature_values[8] = cos_time;
        
        // Feature 9: Position type [-1,0,1] - match Python convention
        int position_type = CurrentPosition.direction;  // No inversion needed now
        model_input_data[idx + 9] = (float)position_type;
        feature_values[9] = position_type;
        
        // Feature 10: Normalized unrealized PnL [-1,1] - match Python's calculation
        double unrealized_pnl = 0.0;
        if(CurrentPosition.direction != 0) {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double entry = CurrentPosition.entryPrice;
            
            if(CurrentPosition.direction > 0) {
                // Long position: current bid - entry
                double pnl = (bid - entry) / entry;
                unrealized_pnl = MathMin(MathMax(pnl * 100.0, -1.0), 1.0);  // Scale by 100 to match Python
            } else {
                // Short position: entry - current ask
                double pnl = (entry - ask) / entry;
                unrealized_pnl = MathMin(MathMax(pnl * 100.0, -1.0), 1.0);  // Scale by 100 to match Python
            }
            
            // Debug info
            if(i == sequence_length - 1) {
                Print("PnL Debug: Entry=", entry,
                      ", Bid=", bid, 
                      ", Ask=", ask,
                      ", Direction=", CurrentPosition.direction,
                      ", PnL=", unrealized_pnl);
            }
        }
        model_input_data[idx + 10] = (float)unrealized_pnl;
        feature_values[10] = unrealized_pnl;

            // Only log features for the last (most recent) time step
            if(i == sequence_length - 1) {
                string feature_names[] = {"returns", "rsi", "atr", "volume_change", "volatility_breakout", 
                                     "trend_strength", "candle_pattern", "sin_time", "cos_time", 
                                     "position_type", "unrealized_pnl"};
                                     
                Print("", "Final Sequence Bar [", sequence_length-1, "] - Most Recent:");
                Print(StringFormat("  Time: %s", TimeToString(time_values[i])));
                Print(StringFormat("  OHLC: %.5f, %.5f, %.5f, %.5f", 
                    open_prices[i], high_prices[i], low_prices[i], close_prices[i]));
                Print(StringFormat("  Raw RSI: %.1f, ADX: %.1f, ATR: %.5f", 
                    rsi_values[i], adx_values[i], atr_values[i]));
                Print("  Array indices: [0] = oldest bar -> [", sequence_length-1, "] = newest bar");
                Print("");
                
                Print("", "Feature Values Comparison:");
                Print("MetaTrader 5 Values:");
                for(int f = 0; f < ArraySize(feature_names); f++) {
                    Print(StringFormat("  %s: %.6f", feature_names[f], feature_values[f]));
                }
                
                // Compare values with Python expected ranges
                Print("Feature Range Validation (Python alignment):");
                Print("  Returns [-0.1,0.1]:", feature_values[0]);
                Print("  RSI [-1,1]:", feature_values[1]);
                Print("  ATR ratio [-1,1]:", feature_values[2]);
                Print("  Volume change [-1,1]:", feature_values[3]);
                Print("  Volatility breakout [0,1]:", feature_values[4]);  // Corrected range to match Python's implementation
                Print("  Trend strength [-1,1]:", feature_values[5]);  // Corrected range to match Python's implementation
                Print("  Candle pattern [-1,1]:", feature_values[6]);
                Print("  Sin time [-1,1]:", feature_values[7]);
                Print("  Cos time [-1,1]:", feature_values[8]);
                Print("  Position type [-1,0,1]:", feature_values[9]);
                Print("  Unrealized PnL [-1,1]:", feature_values[10]);
                Print("");
                Print("All features normalized and aligned with Python implementation");
                Print("");
            }
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get prediction from the model                                     |
//+------------------------------------------------------------------+
bool GetPrediction(int &action, string &description) {
    // Check if ONNX is available
    if(!onnx_available) {
        action = 0; // Hold
        description = "Manual mode - ONNX runtime not available";
        return false;
    }
    
    // Prepare model input data
    if(!PrepareModelInput()) {
        Print("Failed to prepare model input data");
        return false;
    }
    
    // Run prediction
    double actionProbabilities[];
    
    if(!Model.Predict(model_input_data, action, description, actionProbabilities)) {
        Print("Model prediction failed: ", Model.LastError());
        return false;
    }
    
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    
    Print("", "Model Prediction Results:");
    Print(StringFormat("Selected Action: %s (%d)", action_names[action], action));
    Print("Action Probabilities:");
    
    for(int i = 0; i < ArraySize(actionProbabilities); i++) {
        Print(StringFormat("  %s: %.1f%%", action_names[i], actionProbabilities[i] * 100.0));
    }
    
    Print("Decision Context:");
    Print(StringFormat("  Position: %s", CurrentPosition.direction == 0 ? "None" : 
                                      (CurrentPosition.direction > 0 ? "LONG" : "SHORT")));
    if(CurrentPosition.direction != 0) {
        double current_price = CurrentPosition.direction > 0 ? 
            SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
            SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double points = (current_price - CurrentPosition.entryPrice) / SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 
            (CurrentPosition.direction > 0 ? 1 : -1);
        Print(StringFormat("  Current P/L: %.1f points", points));
        Print(StringFormat("  Bars held: %d", CurrentPosition.entryBar));
    }
    Print(StringFormat("  Model note: %s", description));
    Print("");
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trades based on model prediction                          |
//+------------------------------------------------------------------+
void ExecuteTrade(const int action, const string &description) {
    // Debug output
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    Print("");
    Print("Trade Execution Analysis:");
    Print(StringFormat("  Action: %s (%s)", action_names[action], description));
    
    // Account status
    Print("Account Status:");
    Print(StringFormat("  Balance: %.2f", AccountInfoDouble(ACCOUNT_BALANCE)));
    Print(StringFormat("  Equity: %.2f", AccountInfoDouble(ACCOUNT_EQUITY)));
    Print(StringFormat("  Free Margin: %.2f", AccountInfoDouble(ACCOUNT_MARGIN_FREE)));
    Print(StringFormat("  Margin Level: %.1f%%", AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)));
    
    double lotSize = CalculateLotSize();
    if(lotSize <= 0) {
        Print("ERROR: Invalid lot size calculated");
        return;
    }
    
    Print(StringFormat("Trade Size: %.2f lots", lotSize));
    
    switch(action) {
        case 1: // Buy
            if(CurrentPosition.direction == 0) {
                double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double stopLoss = CalculateStopLoss(askPrice, true);
                
                Print("", "Executing BUY order:");
                Print(StringFormat("  Lot Size: %.2f", lotSize));
                Print(StringFormat("  Entry: Market (Ask=%.5f)", askPrice));
                Print(StringFormat("  Stop Loss: %.5f (%.1f points)", stopLoss, StopLoss));
                
                if(Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0, Label)) {
                    CurrentPosition.direction = 1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                    CurrentPosition.entryBar = 0; // Current bar
                    Print("SUCCESS - BUY position opened:");
                    Print(StringFormat("  Entry Price: %.5f", CurrentPosition.entryPrice));
                    Print(StringFormat("  Stop Loss: %.5f", stopLoss));
                    Print(StringFormat("  Risk: %.1f points", MathAbs(CurrentPosition.entryPrice - stopLoss) / SymbolInfoDouble(_Symbol, SYMBOL_POINT)));
                }
                else {
                    Print("FAILED - BUY order error:");
                    Print(StringFormat("  Error Code: %d", Trade.ResultRetcode()));
                    Print(StringFormat("  Description: %s", Trade.ResultRetcodeDescription()));
                }
            }
            else {
                Print("Skipped BUY - Already have position");
            }
            break;
            
        case 2: // Sell
            if(CurrentPosition.direction == 0) {
                double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                double stopLoss = CalculateStopLoss(bidPrice, false);
                
                Print("", "Executing SELL order:");
                Print(StringFormat("  Lot Size: %.2f", lotSize));
                Print(StringFormat("  Entry: Market (Bid=%.5f)", bidPrice));
                Print(StringFormat("  Stop Loss: %.5f (%.1f points)", stopLoss, StopLoss));
                
                if(Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0, Label)) {
                    CurrentPosition.direction = -1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                    CurrentPosition.entryBar = 0; // Current bar
                    Print("SUCCESS - SELL position opened:");
                    Print(StringFormat("  Entry Price: %.5f", CurrentPosition.entryPrice));
                    Print(StringFormat("  Stop Loss: %.5f", stopLoss));
                    Print(StringFormat("  Risk: %.1f points", MathAbs(CurrentPosition.entryPrice - stopLoss) / SymbolInfoDouble(_Symbol, SYMBOL_POINT)));
                }
                else {
                    Print("FAILED - SELL order error:");
                    Print(StringFormat("  Error Code: %d", Trade.ResultRetcode()));
                    Print(StringFormat("  Description: %s", Trade.ResultRetcodeDescription()));
                }
            }
            else {
                Print("Skipped SELL - Already have position");
            }
            break;
            
        case 3: // Close
            if(CurrentPosition.direction != 0) {
                double closing_price = CurrentPosition.direction > 0 ? 
                    SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                    SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                
                Print("", "Executing CLOSE order:");
                Print(StringFormat("  Position: %s", CurrentPosition.direction > 0 ? "LONG" : "SHORT"));
                Print(StringFormat("  Entry Price: %.5f", CurrentPosition.entryPrice));
                Print(StringFormat("  Current Price: %.5f", closing_price));
                Print(StringFormat("  Lot Size: %.2f", CurrentPosition.lotSize));
                
                if(Trade.PositionClose(_Symbol)) {
                    double pnl_points = (closing_price - CurrentPosition.entryPrice) * 
                        (CurrentPosition.direction > 0 ? 1 : -1) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
                    
                    Print("SUCCESS - Position closed:");
                    Print(StringFormat("  Exit Price: %.5f", closing_price));
                    Print(StringFormat("  Points: %.1f", pnl_points));
                    
                    CurrentPosition.direction = 0;
                    CurrentPosition.entryPrice = 0;
                    CurrentPosition.lotSize = 0;
                    CurrentPosition.entryTime = 0;
                    CurrentPosition.entryBar = -1;
                }
                else {
                    Print("FAILED - Close order error:");
                    Print(StringFormat("  Error Code: %d", Trade.ResultRetcode()));
                    Print(StringFormat("  Description: %s", Trade.ResultRetcodeDescription()));
                }
            }
            else {
                Print("Skipped CLOSE - No position");
            }
            break;
            
        case 0: // Hold - do nothing
        default:
            // No action needed for hold
            break;
    }
}

//+------------------------------------------------------------------+
//| Verify position tracking is synchronized with actual positions    |
//+------------------------------------------------------------------+
void VerifyPositions() {
    bool has_mt5_position = false;
    
    // Debugging info - print all positions before verification
    Print("DEBUG: Position verification - Checking all positions");
    
    // Check all positions
    for(int i = 0; i < PositionsTotal(); i++) {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                // Debug output for all positions on this symbol
                Print("DEBUG: Found position - Symbol: ", PositionGetString(POSITION_SYMBOL),
                     ", Type: ", PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "BUY" : "SELL",
                     ", Comment: ", PositionGetString(POSITION_COMMENT),
                     ", Magic: ", PositionGetInteger(POSITION_MAGIC),
                     ", Label match: ", (PositionGetString(POSITION_COMMENT) == Label));
                
                // Check if this position belongs to our EA
                if(PositionGetString(POSITION_COMMENT) == Label) {
                    
                    has_mt5_position = true;
                    int mt5_direction = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
                    double mt5_lot_size = PositionGetDouble(POSITION_VOLUME);
                    double mt5_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
                    
                    // Case 1: We think we don't have a position but MT5 shows one
                    if(CurrentPosition.direction == 0) {
                        Print("Position tracking mismatch: Found MT5 position but no internal tracking. Updating internal tracking.");
                        CurrentPosition.direction = mt5_direction;
                        CurrentPosition.entryPrice = mt5_entry_price;
                        CurrentPosition.lotSize = mt5_lot_size;
                        CurrentPosition.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
                        CurrentPosition.entryBar = 0; // Approximate with current bar
                        Print("Updated position tracking to: ", 
                              CurrentPosition.direction > 0 ? "BUY" : "SELL",
                              ", Size: ", CurrentPosition.lotSize,
                              ", Entry: ", CurrentPosition.entryPrice);
                    }
                    // Case 2: Position details mismatch
                    else if(mt5_direction != CurrentPosition.direction || 
                           MathAbs(mt5_lot_size - CurrentPosition.lotSize) > 0.001) {
                        Print("Position details mismatch - MT5: ", mt5_direction > 0 ? "BUY" : "SELL", " ", 
                              mt5_lot_size, " lots @ ", mt5_entry_price,
                              ", Internal: ", CurrentPosition.direction > 0 ? "BUY" : "SELL", " ",
                              CurrentPosition.lotSize, " lots @ ", CurrentPosition.entryPrice);
                        
                        CurrentPosition.direction = mt5_direction;
                        CurrentPosition.entryPrice = mt5_entry_price;
                        CurrentPosition.lotSize = mt5_lot_size;
                        Print("Updated position tracking to match MT5 position");
                    } else {
                        Print("Position tracking is in sync, direction: ", 
                             CurrentPosition.direction > 0 ? "LONG(+1)" : "SHORT(-1)");
                    }
                    
                    break; // Only process first matching position
                }
            }
        }
    }
    
    // Case 3: We think we have a position but MT5 doesn't
    if(CurrentPosition.direction != 0 && !has_mt5_position) {
        Print("Position tracking mismatch: Internal position exists but no MT5 position found. Clearing internal tracking.");
        Print("Previous tracking had direction: ", CurrentPosition.direction > 0 ? "LONG(+1)" : "SHORT(-1)");
        CurrentPosition.direction = 0;
        CurrentPosition.entryPrice = 0;
        CurrentPosition.lotSize = 0;
        CurrentPosition.entryTime = 0;
        CurrentPosition.entryBar = -1;
    }
    
    Print("Current position direction after verification: ", CurrentPosition.direction);
}

//+------------------------------------------------------------------+
//| Initialize the RecurrentPPO model                                 |
//+------------------------------------------------------------------+
bool InitializeModel() {
    Print("", "Initializing RecurrentPPO Model");
    Print("Configuration:");
    Print(StringFormat("  Model path: %s", ModelPath));
    Print(StringFormat("  Sequence length: %d bars", SequenceLength));
    Print(StringFormat("  LSTM layers: %d", LSTMNumLayers));
    Print(StringFormat("  Hidden size: %d", LSTMHiddenSize));
    Print(StringFormat("  Features per bar: %d", NumFeatures));
    Print(StringFormat("  Possible actions: %d", NumActions));
    Print("");
    
    ModelSettings settings;
    settings.sequenceLength = SequenceLength;
    settings.numFeatures = NumFeatures;
    settings.lstmLayers = LSTMNumLayers;
    settings.lstmHiddenSize = LSTMHiddenSize;
    settings.numActions = NumActions;
    
    if(!Model.Initialize(ModelPath, settings)) {
        Print("Failed to initialize RecurrentPPO model: ", Model.LastError());
        Print("Model error details: ", Model.LastError());
        
        if(FallbackToManualMode) {
            Print("OPERATING IN MANUAL MODE: DRL model unavailable");
            Print("Manual mode allows direct trading with EA's magic number: ", MAGIC_NUMBER);
            return true; // Continue execution without the model
        }
        
        return false;
    }
    
    onnx_available = true;
    Print("Model successfully initialized");
    Print("Status: READY FOR AUTOMATIC TRADING");
    return true;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Initializing DRLTrader with ONNX model...");
    
    // Initialize indicators
    if(!InitializeIndicators()) {
        Print("Failed to initialize indicators");
        return INIT_FAILED;
    }
    
    // Initialize trade object
    Trade.SetExpertMagicNumber(MAGIC_NUMBER);
    Trade.SetMarginMode();
    Trade.SetTypeFillingBySymbol(_Symbol);
    
    // Initialize position tracking
    CurrentPosition.direction = 0;
    CurrentPosition.entryPrice = 0;
    CurrentPosition.lotSize = 0;
    CurrentPosition.entryTime = 0;
    CurrentPosition.entryBar = -1;
    
    // Check for existing positions
    if(PositionsTotal() > 0) {
        for(int i = 0; i < PositionsTotal(); i++) {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket)) {
                if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
                   PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER) {
                    CurrentPosition.direction = 
                        PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
                    CurrentPosition.entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                    CurrentPosition.lotSize = PositionGetDouble(POSITION_VOLUME);
                    CurrentPosition.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
                    CurrentPosition.entryBar = 0; // Approximate with current bar
                    Print("Recovered existing position: ", 
                          CurrentPosition.direction > 0 ? "LONG" : "SHORT", " ",
                          CurrentPosition.lotSize, " lots @ ", CurrentPosition.entryPrice);
                    break;
                }
            }
        }
    }
    
    // Initialize the model
    if(!InitializeModel()) {
        return INIT_FAILED;
    }
    
    if(onnx_available) {
        Print("DRLTrader initialized with ONNX model: ", ModelPath);
    } else if(FallbackToManualMode) {
        Print("DRLTrader initialized in MANUAL MODE (no automatic trading)");
        Print("To use manual mode, place buy/sell orders with the EA's magic number: ", MAGIC_NUMBER);
    }
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    string reasonText = "";
    switch(reason) {
        case REASON_PROGRAM:     reasonText = "Program"; break;
        case REASON_REMOVE:      reasonText = "Expert removed"; break;
        case REASON_RECOMPILE:   reasonText = "Expert recompiled"; break;
        case REASON_CHARTCHANGE: reasonText = "Symbol/timeframe changed"; break;
        case REASON_CHARTCLOSE:  reasonText = "Chart closed"; break;
        case REASON_PARAMETERS:  reasonText = "Parameters changed"; break;
        case REASON_ACCOUNT:     reasonText = "Account changed"; break;
        default:                 reasonText = "Other reason"; break;
    }
    
    Print("");
    Print("DRLTrader Shutdown - Reason: ", reasonText);
    
    if(CurrentPosition.direction != 0) {
        Print("WARNING: Deinitialized with active position:");
        Print(StringFormat("  Direction: %s", CurrentPosition.direction > 0 ? "LONG" : "SHORT"));
        Print(StringFormat("  Entry Price: %.5f", CurrentPosition.entryPrice));
        Print(StringFormat("  Lot Size: %.2f", CurrentPosition.lotSize));
    }
    
    Print("Cleaning up resources...");
    ReleaseIndicators();
    Model.Cleanup();
    Print("Cleanup completed");
    Print("EA deinitialized successfully");
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick() {
    // Skip if spread is too high
    if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpread) {
        Print("Skipping tick - spread too high: ", SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
        return;
    }
    
    // Skip if not a new bar
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, _Period, 0);
    if(current_bar_time == last_bar_time) return;
    last_bar_time = current_bar_time;
    
    // Get current bar info for comparison
    double current_open = iOpen(_Symbol, _Period, 0);
    double current_high = iHigh(_Symbol, _Period, 0);
    double current_low = iLow(_Symbol, _Period, 0);
    double current_close = iClose(_Symbol, _Period, 0);
    
    Print("", "Current incomplete bar:");
    Print(StringFormat("Bar[0] - Time: %s [%s], O: %.5f, H: %.5f, L: %.5f, C: %.5f", 
          TimeToString(current_bar_time),
          "incomplete",
          current_open, current_high, current_low, current_close));
    Print("");
    
    // Verify position tracking is synchronized with MT5 positions
    VerifyPositions();
    
    // Collect historical data - skipping current incomplete bar
    if(!CollectHistoricalData(MinDataBars)) {
        Print("Failed to collect historical data");
        return;
    }
    
    // In manual mode (ONNX unavailable), just track positions
    if(!onnx_available && FallbackToManualMode) {
        // Update entryBar if we have a position
        if(CurrentPosition.direction != 0) {
            CurrentPosition.entryBar++;
        }
        return;
    }
    
    // Get prediction
    int action = 0;
    string description = "";
    if(!GetPrediction(action, description)) {
        Print("Failed to get prediction: ", last_error);
        return;
    }
    
    // Execute trades based on prediction
    ExecuteTrade(action, description);
    
    // Update entryBar if we have a position
    if(CurrentPosition.direction != 0) {
        CurrentPosition.entryBar++;
    }
}
