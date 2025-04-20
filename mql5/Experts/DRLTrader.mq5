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
#define STOP_LOSS_PIPS 1500.0

// Input parameters
input string ModelGroup = ">>> Model Settings <<<";
input string ModelPath = "C:\\Dev\\drl\\mql5\\Models\\XAUUSDm.onnx";  // Path to ONNX model
input int SequenceLength = 500;                    // Number of bars for sequence input
input int LSTMHiddenSize = 256;                    // LSTM hidden state size, changed from 64 to 256
input int LSTMNumLayers = 2;                       // Number of LSTM layers, changed from 1 to 2
input int NumFeatures = 11;                        // Number of features per bar
input int NumActions = 4;                          // Number of possible actions

input string DataGroup = ">>> Data Settings <<<";
input int MinDataBars = 500;                       // Minimum data bars to collect

input string TradingGroup = ">>> Trading Settings <<<";
input int MaxSpread = 350;                         // Maximum allowed spread (points)
input double BalancePerLot = 2500.0;               // Amount required per 0.01 lot
input bool UseFixedLotSize = false;                // Use fixed lot size instead of calculating from balance
input double FixedLotSize = 0.01;                  // Fixed lot size if UseFixedLotSize is true

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
                    entryPrice - (STOP_LOSS_PIPS * pipValue) : 
                    entryPrice + (STOP_LOSS_PIPS * pipValue);
    
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
    // Resize arrays
    ArrayResize(open_prices, bars_to_collect);
    ArrayResize(high_prices, bars_to_collect);
    ArrayResize(low_prices, bars_to_collect);
    ArrayResize(close_prices, bars_to_collect);
    ArrayResize(spread_values, bars_to_collect);
    ArrayResize(volume_values, bars_to_collect);
    ArrayResize(time_values, bars_to_collect);
    
    // Set arrays as series
    ArraySetAsSeries(open_prices, true);
    ArraySetAsSeries(high_prices, true);
    ArraySetAsSeries(low_prices, true);
    ArraySetAsSeries(close_prices, true);
    ArraySetAsSeries(spread_values, true);
    ArraySetAsSeries(volume_values, true);
    ArraySetAsSeries(time_values, true);
    
    // Copy price data
    if (CopyOpen(_Symbol, _Period, 0, bars_to_collect, open_prices) != bars_to_collect) return false;
    if (CopyHigh(_Symbol, _Period, 0, bars_to_collect, high_prices) != bars_to_collect) return false;
    if (CopyLow(_Symbol, _Period, 0, bars_to_collect, low_prices) != bars_to_collect) return false;
    if (CopyClose(_Symbol, _Period, 0, bars_to_collect, close_prices) != bars_to_collect) return false;
    if (CopyTickVolume(_Symbol, _Period, 0, bars_to_collect, volume_values) != bars_to_collect) return false;
    if (CopyTime(_Symbol, _Period, 0, bars_to_collect, time_values) != bars_to_collect) return false;
    
    // Calculate spread values (as points)
    for (int i = 0; i < bars_to_collect; i++) {
        spread_values[i] = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    }
    
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
    
    // Copy indicator values
    if(CopyBuffer(rsi_handle, 0, 0, sequence_length, rsi_values) != sequence_length) return false;
    if(CopyBuffer(atr_handle, 0, 0, sequence_length, atr_values) != sequence_length) return false;
    if(CopyBuffer(bb_handle, 1, 0, sequence_length, upper_band_values) != sequence_length) return false;
    if(CopyBuffer(bb_handle, 2, 0, sequence_length, lower_band_values) != sequence_length) return false;
    if(CopyBuffer(adx_handle, 0, 0, sequence_length, adx_values) != sequence_length) return false;
    
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
    
    // Calculate ATR moving average (20-period) for ATR ratio feature
    double atr_sma[];
    ArrayResize(atr_sma, sequence_length);
    ArraySetAsSeries(atr_sma, true);
    
    for(int i = 0; i < sequence_length; i++) {
        double sum = 0.0;
        int count = 0;
        for(int j = i; j < i + 20 && j < sequence_length; j++) {
            sum += atr_values[j];
            count++;
        }
        atr_sma[i] = sum / count;
    }
    
    // Fill model input array with all 11 features
    for(int i = 0; i < sequence_length; i++) {
        int idx = i * NumFeatures;
        
        // Feature 0: Returns
        double returns = 0.0;
        if(i < sequence_length - 1) {
            returns = (close_prices[i] - close_prices[i+1]) / close_prices[i+1];
        }
        returns = MathMin(MathMax(returns, -0.1), 0.1);  // Clip between -0.1 and 0.1
        model_input_data[idx + 0] = (float)returns;
        
        // Feature 1: RSI normalized to [-1, 1]
        model_input_data[idx + 1] = (float)(rsi_values[i] / 50.0 - 1.0);
        
        // Feature 2: ATR ratio normalized to [-1, 1]
        double atr_ratio = atr_values[i] / (atr_sma[i] + 1e-8);
        double min_expected_ratio = 0.5;
        double max_expected_ratio = 2.0;
        double expected_range = max_expected_ratio - min_expected_ratio;
        double atr_norm = 2.0 * (atr_ratio - min_expected_ratio) / expected_range - 1.0;
        atr_norm = MathMin(MathMax(atr_norm, -1.0), 1.0);
        model_input_data[idx + 2] = (float)atr_norm;
        
        // Feature 3: Volume change
        double volume_pct = 0.0;
        if(i < sequence_length - 1 && volume_values[i+1] > 0) {
            volume_pct = ((double)volume_values[i] - volume_values[i+1]) / volume_values[i+1];
        }
        volume_pct = MathMin(MathMax(volume_pct, -1.0), 1.0);
        model_input_data[idx + 3] = (float)volume_pct;
        
        // Feature 4: Volatility breakout [0,1]
        double band_range = upper_band_values[i] - lower_band_values[i];
        band_range = band_range < 1e-8 ? 1e-8 : band_range;
        double position = close_prices[i] - lower_band_values[i];
        double volatility_breakout = position / band_range;
        volatility_breakout = MathMin(MathMax(volatility_breakout, 0.0), 1.0);
        model_input_data[idx + 4] = (float)volatility_breakout;
        
        // Feature 5: Trend strength [-1,1]
        double trend_strength = MathMin(MathMax(adx_values[i]/25.0 - 1.0, -1.0), 1.0);
        model_input_data[idx + 5] = (float)trend_strength;
        
        // Feature 6: Candle pattern [-1,1]
        double body = close_prices[i] - open_prices[i];
        double upper_wick = high_prices[i] - MathMax(close_prices[i], open_prices[i]);
        double lower_wick = MathMin(close_prices[i], open_prices[i]) - low_prices[i];
        double range = high_prices[i] - low_prices[i] + 1e-8;
        double candle_pattern = (body/range + (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2.0;
        candle_pattern = MathMin(MathMax(candle_pattern, -1.0), 1.0);
        model_input_data[idx + 6] = (float)candle_pattern;
        
        // Feature 7-8: Time encoding using sin/cos
        MqlDateTime time_struct;
        TimeToStruct(time_values[i], time_struct);
        int minutes_in_day = 24 * 60;
        int time_index = time_struct.hour * 60 + time_struct.min;
        double sin_time = MathSin(2.0 * M_PI * time_index / minutes_in_day);
        double cos_time = MathCos(2.0 * M_PI * time_index / minutes_in_day);
        model_input_data[idx + 7] = (float)sin_time;
        model_input_data[idx + 8] = (float)cos_time;
        
        // Feature 9: Position type [-1,0,1]
        int position_type = CurrentPosition.direction;
        model_input_data[idx + 9] = (float)position_type;
        
        // Feature 10: Normalized unrealized PnL [-1,1]
        double unrealized_pnl = 0.0;
        if(CurrentPosition.direction != 0) {
            double current_price = CurrentPosition.direction > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double point_value = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            double pip_size = StringFind(_Symbol, "XAU") >= 0 ? point_value * 10 : point_value;
            double price_diff = (current_price - CurrentPosition.entryPrice) * CurrentPosition.direction;
            unrealized_pnl = price_diff / pip_size * CurrentPosition.lotSize * 100.0; // Rough estimate of account percentage
            unrealized_pnl = MathMin(MathMax(unrealized_pnl / 5.0, -1.0), 1.0); // Normalize to [-1,1]
        }
        model_input_data[idx + 10] = (float)unrealized_pnl;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get prediction from the model                                     |
//+------------------------------------------------------------------+
bool GetPrediction(int &action, string &description) {
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
    
    // Debug output
    string probs = "";
    for(int i = 0; i < ArraySize(actionProbabilities); i++) {
        probs += StringFormat("%.2f", actionProbabilities[i] * 100.0) + "% ";
    }
    
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    Print("Model prediction: Action=", action_names[action], 
          ", Probabilities=[", probs, "], Description=", description);
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trades based on model prediction                          |
//+------------------------------------------------------------------+
void ExecuteTrade(const int action, const string &description) {
    double lotSize = CalculateLotSize();
    if(lotSize <= 0) return;
    
    // Map action string to action code
    // 0 = Hold, 1 = Buy, 2 = Sell, 3 = Close
    
    // Debug output
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    Print("Selected action: ", action_names[action], " with description: ", description);
    
    switch(action) {
        case 1: // Buy
            if(CurrentPosition.direction == 0) {
                double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double stopLoss = CalculateStopLoss(askPrice, true);
                
                if(Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0, "DRL_BUY: " + description)) {
                    CurrentPosition.direction = 1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                    CurrentPosition.entryBar = 0; // Current bar
                    Print("BUY position opened: ", lotSize, " lots @ ", CurrentPosition.entryPrice);
                }
                else {
                    Print("Failed to open BUY position: ", Trade.ResultRetcode(), ", ", Trade.ResultRetcodeDescription());
                }
            }
            break;
            
        case 2: // Sell
            if(CurrentPosition.direction == 0) {
                double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                double stopLoss = CalculateStopLoss(bidPrice, false);
                
                if(Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0, "DRL_SELL: " + description)) {
                    CurrentPosition.direction = -1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                    CurrentPosition.entryBar = 0; // Current bar
                    Print("SELL position opened: ", lotSize, " lots @ ", CurrentPosition.entryPrice);
                }
                else {
                    Print("Failed to open SELL position: ", Trade.ResultRetcode(), ", ", Trade.ResultRetcodeDescription());
                }
            }
            break;
            
        case 3: // Close
            if(CurrentPosition.direction != 0) {
                if(Trade.PositionClose(_Symbol)) {
                    Print("Position closed from ", CurrentPosition.direction > 0 ? "BUY" : "SELL", 
                          " @ ", CurrentPosition.entryPrice);
                    CurrentPosition.direction = 0;
                    CurrentPosition.entryPrice = 0;
                    CurrentPosition.lotSize = 0;
                    CurrentPosition.entryTime = 0;
                    CurrentPosition.entryBar = -1;
                }
                else {
                    Print("Failed to close position: ", Trade.ResultRetcode(), ", ", Trade.ResultRetcodeDescription());
                }
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
    
    // Check all positions
    for(int i = 0; i < PositionsTotal(); i++) {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER) {
                
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
                }
                
                break; // Only process first matching position
            }
        }
    }
    
    // Case 3: We think we have a position but MT5 doesn't
    if(CurrentPosition.direction != 0 && !has_mt5_position) {
        Print("Position tracking mismatch: Internal position exists but no MT5 position found. Clearing internal tracking.");
        CurrentPosition.direction = 0;
        CurrentPosition.entryPrice = 0;
        CurrentPosition.lotSize = 0;
        CurrentPosition.entryTime = 0;
        CurrentPosition.entryBar = -1;
    }
}

//+------------------------------------------------------------------+
//| Initialize the RecurrentPPO model                                 |
//+------------------------------------------------------------------+
bool InitializeModel() {
    ModelSettings settings;
    settings.sequenceLength = SequenceLength;
    settings.numFeatures = NumFeatures;
    settings.lstmLayers = LSTMNumLayers;
    settings.lstmHiddenSize = LSTMHiddenSize;
    settings.numActions = NumActions;
    
    if(!Model.Initialize(ModelPath, settings)) {
        Print("Failed to initialize RecurrentPPO model: ", Model.LastError());
        return false;
    }
    
    Print("RecurrentPPO model initialized successfully from: ", ModelPath);
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
    
    Print("DRLTrader initialized with ONNX model: ", ModelPath);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    ReleaseIndicators();
    Model.Cleanup();
    Print("DRLTrader deinitialized");
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
    
    // Verify position tracking is synchronized with MT5 positions
    VerifyPositions();
    
    // Collect historical data
    if(!CollectHistoricalData(MinDataBars)) {
        Print("Failed to collect historical data");
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
