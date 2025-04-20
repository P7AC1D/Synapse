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

// Import DLL functions
#import "DRLModel.dll"
   void* CreateModel(string model_path, string config_path);
   void DestroyModel(void* handle);
   bool Predict(void* handle, const double& features[], int feature_count,
                double& state[], int state_size, double& output[]);
   bool GetModelProperties(void* handle, int& feature_count,
                         int& hidden_size, int& action_count);
   const char* const* GetFeatureNames(void* handle, int& count);
   void FreeFeatureNames(const char* const* names);
#import

// Constants
#define MAGIC_NUMBER 20240417
#define STOP_LOSS_PIPS 1500.0

// Input parameters
input string ModelGroup = ">>> Model Settings <<<";
input string ModelPath = "C:\\MT5\\model.pt";  // Path to TorchScript model
input string ConfigPath = "C:\\MT5\\model_config.json";  // Path to model config

input string TradingGroup = ">>> Trading Settings <<<";
input int MaxSpread = 350;           // Maximum allowed spread (points)
input double BalancePerLot = 2500.0; // Amount required per 0.01 lot

// Indicators
int rsi_handle;
int atr_handle;
int bb_handle;
int adx_handle;

// Global variables
CTrade Trade;               // Trading object
void* ModelHandle = NULL;   // Model instance handle
double[] LSTMState;        // LSTM state vector
double[] Features;         // Feature vector
double[] ActionProbs;      // Action probabilities
string[] FeatureNames;     // Feature names from model

// Model properties
int FeatureCount = 0;
int HiddenSize = 0;
int ActionCount = 0;

// Position tracking
struct Position {
    int direction;      // 1 for long, -1 for short, 0 for none
    double entryPrice;  // Position entry price
    double lotSize;     // Position size in lots
    datetime entryTime; // Entry timestamp
};

Position CurrentPosition;

//+------------------------------------------------------------------+
//| Initialize indicators                                              |
//+------------------------------------------------------------------+
bool InitializeIndicators() {
    rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
    atr_handle = iATR(_Symbol, _Period, 14);
    bb_handle = iBands(_Symbol, _Period, 20, 0, 2, PRICE_CLOSE);
    adx_handle = iADX(_Symbol, _Period, 14);
    
    return rsi_handle != INVALID_HANDLE && 
           atr_handle != INVALID_HANDLE && 
           bb_handle != INVALID_HANDLE &&
           adx_handle != INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Release indicators                                                |
//+------------------------------------------------------------------+
void ReleaseIndicators() {
    IndicatorRelease(rsi_handle);
    IndicatorRelease(atr_handle);
    IndicatorRelease(bb_handle);
    IndicatorRelease(adx_handle);
}

//+------------------------------------------------------------------+
//| Calculate features                                                 |
//+------------------------------------------------------------------+
void ProcessFeatures() {
    ArrayResize(Features, 9); // Base features, position features added later
    
    // Price data
    double close[];
    double open[];
    ArraySetAsSeries(close, true);
    ArraySetAsSeries(open, true);
    CopyClose(_Symbol, _Period, 0, 2, close);
    CopyOpen(_Symbol, _Period, 0, 1, open);
    
    // Returns
    Features[0] = (close[0] - close[1]) / close[1];
    Features[0] = MathMax(MathMin(Features[0], 0.1), -0.1);
    
    // RSI
    double rsi[];
    ArraySetAsSeries(rsi, true);
    CopyBuffer(rsi_handle, 0, 0, 1, rsi);
    Features[1] = rsi[0] / 50.0 - 1.0;
    
    // ATR
    double atr[];
    ArraySetAsSeries(atr, true);
    CopyBuffer(atr_handle, 0, 0, 1, atr);
    Features[2] = atr[0] / close[0];
    
    // Volume Change
    long volume[];
    ArraySetAsSeries(volume, true);
    CopyTickVolume(_Symbol, _Period, 0, 2, volume);
    Features[3] = volume[1] > 0 ? 
                  ((double)volume[0] - volume[1]) / volume[1] : 0;
    Features[3] = MathMax(MathMin(Features[3], 1.0), -1.0);
    
    // Bollinger Bands
    double upper[], lower[];
    ArraySetAsSeries(upper, true);
    ArraySetAsSeries(lower, true);
    CopyBuffer(bb_handle, 1, 0, 1, upper);
    CopyBuffer(bb_handle, 2, 0, 1, lower);
    
    double band_range = upper[0] - lower[0];
    double position = close[0] - lower[0];
    Features[4] = position / (band_range + 1e-8);
    Features[4] = MathMax(MathMin(Features[4], 1.0), 0.0);
    
    // ADX (Trend Strength)
    double adx[];
    ArraySetAsSeries(adx, true);
    CopyBuffer(adx_handle, 0, 0, 1, adx);
    Features[5] = MathMax(MathMin(adx[0]/25.0 - 1.0, 1.0), -1.0);
    
    // Candle Pattern
    double high[], low[];
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    CopyHigh(_Symbol, _Period, 0, 1, high);
    CopyLow(_Symbol, _Period, 0, 1, low);
    
    double body = close[0] - open[0];
    double upper_wick = high[0] - MathMax(close[0], open[0]);
    double lower_wick = MathMin(close[0], open[0]) - low[0];
    double range = high[0] - low[0] + 1e-8;
    
    Features[6] = (body/range + 
                  (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2.0;
    Features[6] = MathMax(MathMin(Features[6], 1.0), -1.0);
    
    // Time Features
    MqlDateTime time;
    TimeToStruct(TimeCurrent(), time);
    int minutes = time.hour * 60 + time.min;
    Features[7] = MathSin(2.0 * M_PI * minutes / 1440);
    Features[8] = MathCos(2.0 * M_PI * minutes / 1440);
    
    // Add position features
    int current_size = ArraySize(Features);
    ArrayResize(Features, current_size + 2);
    
    // Position type
    Features[current_size] = (double)CurrentPosition.direction;
    
    // Unrealized P&L
    double unrealized_pnl = 0;
    if (CurrentPosition.direction != 0) {
        double current_price = CurrentPosition.direction == 1 ? 
                             SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                             SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        unrealized_pnl = CurrentPosition.direction *
                        (current_price - CurrentPosition.entryPrice) /
                        CurrentPosition.entryPrice;
    }
    Features[current_size + 1] = MathMax(MathMin(unrealized_pnl, 1.0), -1.0);
    
    // Debug: Log feature values
    for(int i = 0; i < ArraySize(Features); i++) {
        string feature_name = i < ArraySize(FeatureNames) ? FeatureNames[i] : StringFormat("Feature_%d", i);
        Print("Feature ", feature_name, ": ", Features[i]);
    }
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
//| Execute trades based on model output                               |
//+------------------------------------------------------------------+
void ExecuteTrade(const double& probs[]) {
    // Get action with highest probability
    int action = 0;
    double maxProb = probs[0];
    for(int i = 1; i < ActionCount; i++) {
        if(probs[i] > maxProb) {
            maxProb = probs[i];
            action = i;
        }
    }
    
    // Debug output
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    Print("Selected action: ", action_names[action], " (", action, ") with probability ", maxProb);
    
    double lotSize = CalculateLotSize();
    if(lotSize == 0) return;
    
    switch(action) {
        case 1: // Buy
            if(CurrentPosition.direction == 0) {
                double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double stopLoss = CalculateStopLoss(askPrice, true);
                
                if(Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0)) {
                    CurrentPosition.direction = 1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                }
            }
            break;
            
        case 2: // Sell
            if(CurrentPosition.direction == 0) {
                double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                double stopLoss = CalculateStopLoss(bidPrice, false);
                
                if(Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0)) {
                    CurrentPosition.direction = -1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
                }
            }
            break;
            
        case 3: // Close
            if(CurrentPosition.direction != 0) {
                if(Trade.PositionClose(_Symbol)) {
                    CurrentPosition.direction = 0;
                    CurrentPosition.entryPrice = 0;
                    CurrentPosition.lotSize = 0;
                    CurrentPosition.entryTime = 0;
                }
            }
            break;
    }
}

//+------------------------------------------------------------------+
//| Get feature names from model                                       |
//+------------------------------------------------------------------+
bool GetFeatureNames() {
    int count = 0;
    const char* const* names = GetFeatureNames(ModelHandle, count);
    
    if(names != NULL && count > 0) {
        ArrayResize(FeatureNames, count);
        for(int i = 0; i < count; i++) {
            FeatureNames[i] = names[i];
        }
        FreeFeatureNames(names);
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Initializing DRLTrader with LibTorch implementation...");
    
    // Initialize indicators
    if(!InitializeIndicators()) {
        Print("Failed to initialize indicators");
        return INIT_FAILED;
    }
    
    // Initialize model
    ModelHandle = CreateModel(ModelPath, ConfigPath);
    if(ModelHandle == NULL) {
        Print("Failed to load model");
        return INIT_FAILED;
    }
    
    // Get model properties
    if(!GetModelProperties(ModelHandle, FeatureCount, HiddenSize, ActionCount)) {
        Print("Failed to get model properties");
        return INIT_FAILED;
    }
    
    // Get feature names
    if(!GetFeatureNames()) {
        Print("Warning: Failed to get feature names");
    }
    
    Print("Model loaded successfully - Features: ", FeatureCount,
          ", Hidden Size: ", HiddenSize,
          ", Actions: ", ActionCount);
    
    // Initialize arrays
    ArrayResize(LSTMState, 2 * HiddenSize);
    ArrayInitialize(LSTMState, 0);
    
    ArrayResize(ActionProbs, ActionCount);
    ArrayInitialize(ActionProbs, 0);
    
    // Initialize trade object
    Trade.SetExpertMagicNumber(MAGIC_NUMBER);
    Trade.SetMarginMode();
    Trade.SetTypeFillingBySymbol(_Symbol);
    
    // Initialize position tracking
    CurrentPosition.direction = 0;
    CurrentPosition.entryPrice = 0;
    CurrentPosition.lotSize = 0;
    CurrentPosition.entryTime = 0;
    
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
                    break;
                }
            }
        }
    }
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(ModelHandle != NULL) {
        DestroyModel(ModelHandle);
        ModelHandle = NULL;
    }
    ReleaseIndicators();
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
    
    // Process features
    ProcessFeatures();
    
    // Get model prediction
    if(!Predict(ModelHandle, Features, ArraySize(Features),
                LSTMState, ArraySize(LSTMState), ActionProbs)) {
        Print("Prediction failed");
        return;
    }
    
    // Execute trades based on prediction
    ExecuteTrade(ActionProbs);
}
