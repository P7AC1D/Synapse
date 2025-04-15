//+------------------------------------------------------------------+
//|                                                    DRLTrader.mq5    |
//|                                   Copyright 2024, DRL Trading Bot   |
//|                                     https://github.com/your-repo    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"
#property version   "1.00"
#property strict

// Include required files
#include <Trade/Trade.mqh>
#include <DRL/features.mqh>
#include <DRL/model.mqh>
#include <DRL/matrix.mqh>
#include <DRL/weights.mqh>

// Constants
#define MAGIC_COMMENT "PPO_LSTM"
#define STOP_LOSS_PIPS 1500.0
#define BARS_TO_FETCH 500

// Input parameters
input int MaxSpread = 35;                                 // Maximum allowed spread (points)

sinput string PositionGroup = "Position Sizing";          // >>> Position Sizing <<<
input double BALANCE_PER_LOT = 2500.0;                   // Amount required per 0.01 lot

sinput string ModelGroup = "Model Settings";              // >>> Model Settings <<<
input bool ResetStatesOnGap = true;                      // Reset LSTM states on timeframe gap
input int TimeframeMinutes = 15;                        // Trading timeframe in minutes

// Global variables
CTrade Trade;                                            // Trading object
CFeatureProcessor *FeatureProcessor;    // Feature calculation class
double LSTMState[];                    // Current LSTM state
datetime LastBarTime;                  // Last processed bar time
int LastBarIndex;                      // Last processed bar index
bool FirstTick = true;                 // Flag for first tick
// Position tracking
struct Position {
    int direction;       // 1 for long, -1 for short, 0 for none
    double entryPrice;   // Position entry price
    double lotSize;      // Position size in lots
    int entryStep;       // Entry step relative to data window
    datetime entryTime;  // Entry timestamp
    bool pendingUpdate;  // Track if position update is pending
};

Position CurrentPosition;

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize trade object with comment as magic number
    Trade.SetExpertMagicNumber(StringGetTickCount());  // Unique identifier
    Trade.SetExpertComment(MAGIC_COMMENT);  // Use same comment as Python bot
    Trade.SetMarginMode();
    Trade.SetTypeFillingBySymbol(_Symbol);
    
    Print("Expert initialized with magic number: ", Trade.RequestMagic(),
          ", Comment: ", MAGIC_COMMENT);
    
    // Initialize feature processor
    FeatureProcessor = new CFeatureProcessor();
    FeatureProcessor.Init(_Symbol, PERIOD_CURRENT);
    
    // Initialize LSTM state array
    ArrayResize(LSTMState, LSTM_UNITS);
    ArrayInitialize(LSTMState, 0);
    
    // Initialize position tracking to match Python's None state
    CurrentPosition.direction = 0;
    CurrentPosition.entryPrice = 0.0;
    CurrentPosition.lotSize = 0.0;
    CurrentPosition.entryStep = 0;
    CurrentPosition.entryTime = 0;
    CurrentPosition.pendingUpdate = false;
    Print("Position initialized to None state");
    
    // Check for existing positions
    if(PositionsTotal() > 0) {
        for(int i = 0; i < PositionsTotal(); i++) {
            if(PositionSelectByTicket(PositionGetTicket(i))) {
                if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
                   PositionGetString(POSITION_COMMENT) == MAGIC_COMMENT) {
                    // Match Python's position recovery exactly
                    CurrentPosition.direction = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
                    CurrentPosition.entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                    CurrentPosition.lotSize = PositionGetDouble(POSITION_VOLUME);
                    CurrentPosition.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
                    CurrentPosition.entryStep = 0;  // Will be updated in first trading cycle
                    CurrentPosition.pendingUpdate = false;
                    Print("Recovered position: ", 
                          CurrentPosition.direction == 1 ? "LONG" : "SHORT",
                          " ", CurrentPosition.lotSize, " lots @ ",
                          CurrentPosition.entryPrice);
                    break;
                }
            }
        }
    }
    
    // Reset state tracking
    LastBarTime = 0;
    LastBarIndex = 0;
    FirstTick = true;
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Clean up resources
    if(FeatureProcessor != NULL) {
        FeatureProcessor.Deinit();
        delete FeatureProcessor;
    }
}

//+------------------------------------------------------------------+
//| Calculate stop loss price based on pips                            |
//+------------------------------------------------------------------+
double CalculateStopLoss(double entryPrice, bool isBuy) {
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    
    // For XAUUSD, 1 pip = 0.1 points (multiply by 10)
    double pipValue = StringFind(_Symbol, "XAU") >= 0 ? point * 10 : point;
    
    // Calculate stop loss price
    double slPrice = isBuy ? 
        entryPrice - (STOP_LOSS_PIPS * pipValue) :
        entryPrice + (STOP_LOSS_PIPS * pipValue);
    
    // Round to symbol digits
    return NormalizeDouble(slPrice, digits);
}

//+------------------------------------------------------------------+
//| Calculate lot size matching Python implementation                   |
//+------------------------------------------------------------------+
double CalculateLotSize() {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    // Match Python calculation exactly
    double lotSize = (balance / BALANCE_PER_LOT) * minLot;
    lotSize = MathRound(lotSize / lotStep) * lotStep;  // Round to nearest lot step
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
    
    return lotSize;
}

//+------------------------------------------------------------------+
//| Execute trade based on model prediction                            |
//+------------------------------------------------------------------+
void ExecuteTrade(int action, double features[]) {
    // Calculate lot size and current prices
    double lotSize = CalculateLotSize();
    if(lotSize == 0) return;
    
    double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    switch(action) {
        case 1:  // Buy
            if(CurrentPosition.direction == 0) {
                double stopLoss = CalculateStopLoss(askPrice, true);
                Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0, MAGIC_COMMENT);
                if(Trade.ResultRetcode() == TRADE_RETCODE_DONE) {
                    // Update position only after confirmed execution
                    CurrentPosition.direction = 1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryStep = BARS_TO_FETCH - 1;  // Last step in data window
                    CurrentPosition.entryTime = TimeCurrent();
                    Print("Buy executed: ", lotSize, " lots @ ", Trade.ResultPrice(), ", SL: ", stopLoss);
                }
            }
            break;
            
        case 2:  // Sell
            if(CurrentPosition.direction == 0) {
                double stopLoss = CalculateStopLoss(bidPrice, false);
                Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0, MAGIC_COMMENT);
                if(Trade.ResultRetcode() == TRADE_RETCODE_DONE) {
                    // Update position only after confirmed execution
                    CurrentPosition.direction = -1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryStep = BARS_TO_FETCH - 1;  // Last step in data window
                    CurrentPosition.entryTime = TimeCurrent();
                    Print("Sell executed: ", lotSize, " lots @ ", Trade.ResultPrice(), ", SL: ", stopLoss);
                }
            }
            break;
            
        case 3:  // Close
            if(CurrentPosition.direction != 0) {
                Trade.PositionClose(_Symbol);
                if(Trade.ResultRetcode() == TRADE_RETCODE_DONE) {
                    // Reset all position fields to match Python's None state
                    CurrentPosition.direction = 0;
                    CurrentPosition.entryPrice = 0.0;
                    CurrentPosition.lotSize = 0.0;
                    CurrentPosition.entryStep = 0;
                    CurrentPosition.entryTime = 0;
                    CurrentPosition.pendingUpdate = false;
                }
            }
            break;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick() {
    // Skip if spread is too high
    if(SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpread)
        return;
        
    // Check for new bar
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(currentBarTime == LastBarTime)
        return;
        
// Check for significant time gap using Python's logic
    if(ResetStatesOnGap && LastBarTime > 0) {
        datetime expectedTime = LastBarTime + TimeframeMinutes * 60;
        int timeDiff = (int)(currentBarTime - expectedTime);
        
        if(timeDiff > (TimeframeMinutes * 2 * 60)) {
            Print("Significant data gap detected (", timeDiff/60.0, " minutes), resetting LSTM states");
            ArrayInitialize(LSTMState, 0);
        }
    }
    
    // Calculate features
    double features[];
    FeatureProcessor.ProcessFeatures(features);
    
    // Add position features
    ArrayResize(features, ArraySize(features) + 2);
    features[ArraySize(features) - 2] = (double)CurrentPosition.direction;  // Position type
    
    // Calculate unrealized P&L
    double unrealizedPnl = 0;
    if(CurrentPosition.direction != 0) {
        double currentPrice = CurrentPosition.direction == 1 ? 
            SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
            SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        unrealizedPnl = CurrentPosition.direction * 
            (currentPrice - CurrentPosition.entryPrice) / 
            CurrentPosition.entryPrice;
    }
    features[ArraySize(features) - 1] = MathMax(MathMin(unrealizedPnl, 1.0), -1.0);
    
    // Run LSTM inference
    double lstm_output[];
    RunLSTMInference(features, LSTMState, lstm_output);
    
    // Get action with highest probability
    int action = 0;
    double maxProb = lstm_output[0];
    for(int i = 1; i < ACTION_COUNT; i++) {
        if(lstm_output[i] > maxProb) {
            maxProb = lstm_output[i];
            action = i;
        }
    }
    
    // Execute trade
    ExecuteTrade(action, features);
    
    // Update state tracking
    LastBarTime = currentBarTime;
    LastBarIndex++;
    FirstTick = false;
}

//+------------------------------------------------------------------+
//| Run LSTM inference                                                 |
//+------------------------------------------------------------------+
void RunLSTMInference(const double &features[], double &state[], double &output[]) {
    // Temporary arrays for LSTM gates
    double input_gate[];
    double forget_gate[];
    double cell_state[];
    double output_gate[];
    double hidden_state[];
    
    // Initialize arrays
    int hidden_size = LSTM_UNITS;
    ArrayResize(input_gate, hidden_size);
    ArrayResize(forget_gate, hidden_size);
    ArrayResize(cell_state, hidden_size);
    ArrayResize(output_gate, hidden_size);
    ArrayResize(hidden_state, hidden_size);
    ArrayResize(output, ACTION_COUNT);
    
    // Actor LSTM
    // Input transformation
    double actor_input[];
    MatrixMultiply(features, actor_input_weight, actor_input,
                   1, FEATURE_COUNT, FEATURE_COUNT, LSTM_UNITS);
    double actor_hidden_transform[];
    MatrixMultiply(state, actor_hidden_weight, actor_hidden_transform,
                   1, LSTM_UNITS, LSTM_UNITS, LSTM_UNITS * 4);
                   
    // Calculate gates
    for(int i = 0; i < LSTM_UNITS; i++) {
        int idx = i;
        forget_gate[i] = sigmoid(actor_input[idx] + 
                               actor_hidden_transform[idx] + 
                               actor_hidden_bias[idx]);
                               
        idx += LSTM_UNITS;
        input_gate[i] = sigmoid(actor_input[idx] + 
                              actor_hidden_transform[idx] + 
                              actor_input_bias[idx]);
                              
        idx += LSTM_UNITS;
        cell_state[i] = tanh(actor_input[idx] + 
                           actor_hidden_transform[idx] + 
                           actor_input_bias[idx]);
                           
        idx += LSTM_UNITS;
        output_gate[i] = sigmoid(actor_input[idx] + 
                               actor_hidden_transform[idx] + 
                               actor_hidden_bias[idx]);
    }
    
    // Update cell and hidden states
    for(int i = 0; i < LSTM_UNITS; i++) {
        cell_state[i] = forget_gate[i] * state[i] + 
                       input_gate[i] * cell_state[i];
        hidden_state[i] = output_gate[i] * tanh(cell_state[i]);
    }
    
    // Update LSTM state for next iteration
    ArrayCopy(state, hidden_state);
    
    // Calculate final output
    MatrixMultiply(hidden_state, actor_output_weight, output,
                   1, LSTM_UNITS, LSTM_UNITS, ACTION_COUNT);
                   
    // Add bias and apply softmax
    double sum = 0;
    for(int i = 0; i < ACTION_COUNT; i++) {
        output[i] = MathExp(output[i] + actor_output_bias[i]);
        sum += output[i];
    }
    
    // Normalize probabilities
    if(sum > 0) {
        for(int i = 0; i < ACTION_COUNT; i++) {
            output[i] /= sum;
        }
    }
}
