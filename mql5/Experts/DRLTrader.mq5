//+------------------------------------------------------------------+
//|                                                    DRLTrader.mq5    |
//|                                   Copyright 2024, DRL Trading Bot   |
//|                                     https://github.com/your-repo    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link "https://github.com/your-repo"
#property version "1.00"
#property strict

// Include required files
#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>
#include <Math/Stat/Math.mqh>
#include <Trade/SymbolInfo.mqh>
#include <DRL/features.mqh>
#include <DRL/model.mqh>
#include <DRL/matrix.mqh>
#include <DRL/weights.mqh>

// Constants
#define MAGIC_NUMBER 20240417
#define STOP_LOSS_PIPS 1500.0
#define BARS_TO_FETCH 500

// Input parameters
input int MaxSpread = 350; // Maximum allowed spread (points)

// Position sizing settings
input string PositionGroup = ">>> Position Sizing <<<"; // Position Sizing
input double BALANCE_PER_LOT = 2500.0;                 // Amount required per 0.01 lot

// Model settings
input string ModelGroup = ">>> Model Settings <<<"; // Model Settings
input bool ResetStatesOnGap = true;                 // Reset LSTM states on timeframe gap
input int TimeframeMinutes = 15;                    // Trading timeframe in minutes

// Global variables
CTrade Trade;                        // Trading object
CFeatureProcessor *FeatureProcessor; // Feature calculation class
double LSTMState[];                  // Current LSTM state
datetime LastBarTime;                // Last processed bar time
int LastBarIndex;                    // Last processed bar index
bool FirstTick = true;               // Flag for first tick

// Position tracking
struct Position
{
    int direction;      // 1 for long, -1 for short, 0 for none
    double entryPrice;  // Position entry price
    double lotSize;     // Position size in lots
    int entryStep;      // Entry step relative to data window
    datetime entryTime; // Entry timestamp
    bool pendingUpdate; // Track if position update is pending
};

Position CurrentPosition;

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("DEBUG: Starting initialization of DRLTrader");

    // Check account type and broker requirements
    Print("DEBUG: Account info - Leverage: 1:", AccountInfoInteger(ACCOUNT_LEVERAGE),
          ", Stop Out Level: ", AccountInfoInteger(ACCOUNT_MARGIN_SO_MODE),
          ", Allowed Trade Mode: ", AccountInfoInteger(ACCOUNT_TRADE_MODE));

    // Check symbol details
    Print("DEBUG: Symbol details - Min Lot: ", SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN),
          ", Max Lot: ", SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX),
          ", Lot Step: ", SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP),
          ", Trade Allowed: ", SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE));

    // Initialize trade object with magic number
    Trade.SetExpertMagicNumber(MAGIC_NUMBER);
    Trade.SetMarginMode();
    Trade.SetTypeFillingBySymbol(_Symbol);

    Print("DEBUG: Expert initialized with magic number: ", Trade.RequestMagic());

    // Initialize feature processor
    FeatureProcessor = new CFeatureProcessor();
    FeatureProcessor.Init(_Symbol, _Period);
    Print("DEBUG: Feature processor initialized");

    // Initialize LSTM state array
    ArrayResize(LSTMState, LSTM_UNITS);
    ArrayInitialize(LSTMState, 0);
    Print("DEBUG: LSTM state initialized with ", LSTM_UNITS, " units");

    // Initialize position tracking to match Python's None state
    CurrentPosition.direction = 0;
    CurrentPosition.entryPrice = 0.0;
    CurrentPosition.lotSize = 0.0;
    CurrentPosition.entryStep = 0;
    CurrentPosition.entryTime = 0;
    CurrentPosition.pendingUpdate = false;
    Print("DEBUG: Position initialized to None state");

    // Check for existing positions
    Print("DEBUG: Checking for existing positions, total positions: ", PositionsTotal());
    if (PositionsTotal() > 0)
    {
        for (int i = 0; i < PositionsTotal(); i++)
        {
            ulong ticket = PositionGetTicket(i);
            if (PositionSelectByTicket(ticket))
            {
                if (PositionGetString(POSITION_SYMBOL) == _Symbol &&
                    PositionGetInteger(POSITION_MAGIC) == MAGIC_NUMBER)
                {
                    // Match Python's position recovery exactly
                    CurrentPosition.direction = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
                    CurrentPosition.entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                    CurrentPosition.lotSize = PositionGetDouble(POSITION_VOLUME);
                    CurrentPosition.entryTime = (datetime)PositionGetInteger(POSITION_TIME);
                    CurrentPosition.entryStep = 0; // Will be updated in first trading cycle
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

    Print("DEBUG: Initialization completed successfully");
    return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Clean up resources
    if (CheckPointer(FeatureProcessor) == POINTER_DYNAMIC)
    {
        FeatureProcessor.Deinit();
        delete FeatureProcessor;
    }
}

//+------------------------------------------------------------------+
//| Calculate stop loss price based on pips                            |
//+------------------------------------------------------------------+
double CalculateStopLoss(const double entryPrice, const bool isBuy)
{
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // For XAUUSD, 1 pip = 0.1 points (multiply by 10)
    double pipValue = StringFind(_Symbol, "XAU") >= 0 ? point * 10 : point;

    // Calculate stop loss price
    double slPrice = isBuy ? entryPrice - (STOP_LOSS_PIPS * pipValue) : entryPrice + (STOP_LOSS_PIPS * pipValue);

    // Round to symbol digits
    return NormalizeDouble(slPrice, digits);
}

//+------------------------------------------------------------------+
//| Calculate lot size matching Python implementation                   |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    // Match Python calculation exactly
    double lotSize = (balance / BALANCE_PER_LOT) * minLot;
    lotSize = MathRound(lotSize / lotStep) * lotStep; // Round to nearest lot step
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

    return lotSize;
}

//+------------------------------------------------------------------+
//| Execute trade based on model prediction                            |
//+------------------------------------------------------------------+
void ExecuteTrade(const int action, const double &features[])
{
    // Calculate lot size and current prices
    double lotSize = CalculateLotSize();
    Print("DEBUG: CalculateLotSize() returned ", lotSize);

    if (lotSize == 0)
    {
        Print("DEBUG: Trade execution aborted - lotSize is zero");
        return;
    }

    double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    Print("DEBUG: Current prices - Ask: ", askPrice, ", Bid: ", bidPrice, ", Spread: ", SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));

    // Log account info
    Print("DEBUG: Account info - Balance: ", AccountInfoDouble(ACCOUNT_BALANCE),
          ", Equity: ", AccountInfoDouble(ACCOUNT_EQUITY),
          ", Margin level: ", AccountInfoDouble(ACCOUNT_MARGIN_LEVEL),
          ", Free margin: ", AccountInfoDouble(ACCOUNT_MARGIN_FREE));

    switch (action)
    {
    case 1: // Buy
        if (CurrentPosition.direction == 0)
        {
            double stopLoss = CalculateStopLoss(askPrice, true);
            Print("DEBUG: Attempting to BUY ", lotSize, " lots @ market, SL: ", stopLoss);
            if (Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0))
            {
                // Update position only after confirmed execution
                CurrentPosition.direction = 1;
                CurrentPosition.entryPrice = Trade.ResultPrice();
                CurrentPosition.lotSize = lotSize;
                CurrentPosition.entryStep = BARS_TO_FETCH - 1; // Last step in data window
                CurrentPosition.entryTime = TimeCurrent();
                Print("DEBUG: Buy executed successfully: ", lotSize, " lots @ ", Trade.ResultPrice(), ", SL: ", stopLoss);
            }
            else
            {
                int errorCode = GetLastError();
                Print("DEBUG: Buy execution FAILED - Error code: ", errorCode, ", Description: ", ErrorDescription(errorCode));
            }
        }
        else
        {
            Print("DEBUG: Buy action ignored - already have position: ",
                  CurrentPosition.direction == 1 ? "LONG" : "SHORT",
                  " with ", CurrentPosition.lotSize, " lots");
        }
        break;

    case 2: // Sell
        if (CurrentPosition.direction == 0)
        {
            double stopLoss = CalculateStopLoss(bidPrice, false);
            Print("DEBUG: Attempting to SELL ", lotSize, " lots @ market, SL: ", stopLoss);
            if (Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0))
            {
                // Update position only after confirmed execution
                CurrentPosition.direction = -1;
                CurrentPosition.entryPrice = Trade.ResultPrice();
                CurrentPosition.lotSize = lotSize;
                CurrentPosition.entryStep = BARS_TO_FETCH - 1; // Last step in data window
                CurrentPosition.entryTime = TimeCurrent();
                Print("DEBUG: Sell executed successfully: ", lotSize, " lots @ ", Trade.ResultPrice(), ", SL: ", stopLoss);
            }
            else
            {
                int errorCode = GetLastError();
                Print("DEBUG: Sell execution FAILED - Error code: ", errorCode, ", Description: ", ErrorDescription(errorCode));
            }
        }
        else
        {
            Print("DEBUG: Sell action ignored - already have position: ",
                  CurrentPosition.direction == 1 ? "LONG" : "SHORT",
                  " with ", CurrentPosition.lotSize, " lots");
        }
        break;

    case 3: // Close
        if (CurrentPosition.direction != 0)
        {
            Print("DEBUG: Attempting to close position");
            if (Trade.PositionClose(_Symbol))
            {
                Print("DEBUG: Position closed successfully");
                // Reset all position fields to match Python's None state
                CurrentPosition.direction = 0;
                CurrentPosition.entryPrice = 0.0;
                CurrentPosition.lotSize = 0.0;
                CurrentPosition.entryStep = 0;
                CurrentPosition.entryTime = 0;
                CurrentPosition.pendingUpdate = false;
            }
            else
            {
                int errorCode = GetLastError();
                Print("DEBUG: Position close FAILED - Error code: ", errorCode, ", Description: ", ErrorDescription(errorCode));
            }
        }
        else
        {
            Print("DEBUG: Close action ignored - no position to close");
        }
        break;

    case 0: // Hold
        Print("DEBUG: Hold action - no trades executed");
        break;

    default:
        Print("DEBUG: Unknown action value: ", action);
        break;
    }
}

//+------------------------------------------------------------------+
//| Get error description                                             |
//+------------------------------------------------------------------+
string ErrorDescription(int errorCode)
{
    switch (errorCode)
    {
    case 0: // ERR_NO_ERROR
        return "No error";
    case 4051: // ERR_INVALID_FUNCTION_PARAMETER_VALUE
        return "Invalid parameter value";
    case 4052: // ERR_INVALID_TRADE_PARAMETERS
        return "Invalid trade parameters";
    case 4022: // ERR_SYSTEM_BUSY
        return "System is busy";
    case 4008: // ERR_NO_RESULT
        return "No result";
    case 4055: // ERR_INVALID_PRICE
        return "Invalid price";
    case 4056: // ERR_INVALID_STOPS
        return "Invalid stops";
    case 4061: // ERR_INVALID_VOLUME
        return "Invalid volume";
    case 4109: // ERR_TRADE_DISABLED
        return "Trade is disabled";
    case 4060: // ERR_MARKET_CLOSED
        return "Market is closed";
    case 4062: // ERR_TRADE_TOO_MANY_ORDERS
        return "Too many orders";
    case 4059: // ERR_TRADE_CONTEXT_BUSY
        return "Trade context is busy";
    case 4113: // ERR_TRADE_EXPERT_DISABLED_BY_SERVER
        return "EA trading disabled by server";
    case 4057: // ERR_TRADE_EXPIRATION_DENIED
        return "Expiration is denied";
    case 4107: // ERR_TRADE_TOO_MANY_REQUESTS
        return "Too many requests";
    case 4110: // ERR_TRADE_HEDGE_PROHIBITED
        return "Hedge is prohibited";
    case 4111: // ERR_TRADE_PROHIBITED_BY_FIFO
        return "Prohibited by FIFO";
    case 4108: // ERR_TRADE_POSITION_NOT_FOUND
        return "Position not found";
    case 4114: // ERR_TRADE_IMPOSSIBLE_TO_CLOSE
        return "Impossible to close";
    case 4025: // ERR_TRADE_NOT_ALLOWED_IN_TESTING
        return "Not allowed in testing";
    default:
        return "Unknown error " + IntegerToString(errorCode);
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    // New debug section to verify model dimensions
    if (FirstTick) {
        Print("MODEL DEBUG: FEATURE_COUNT=", FEATURE_COUNT, 
              ", LSTM_UNITS=", LSTM_UNITS, 
              ", ACTION_COUNT=", ACTION_COUNT);
        
        // Check weight array dimensions
        Print("MODEL DEBUG: actor_input_weight dimensions: ", 
              ArrayRange(actor_input_weight, 0), "x", ArrayRange(actor_input_weight, 1));
        Print("MODEL DEBUG: actor_hidden_weight dimensions: ", 
              ArrayRange(actor_hidden_weight, 0), "x", ArrayRange(actor_hidden_weight, 1));
        Print("MODEL DEBUG: actor_output_weight dimensions: ", 
              ArrayRange(actor_output_weight, 0), "x", ArrayRange(actor_output_weight, 1));
        
        // Check bias array sizes
        Print("MODEL DEBUG: actor_input_bias size: ", ArraySize(actor_input_bias));
        Print("MODEL DEBUG: actor_hidden_bias size: ", ArraySize(actor_hidden_bias));
        Print("MODEL DEBUG: actor_output_bias size: ", ArraySize(actor_output_bias));
    }
    
    // Skip if spread is too high
    if (SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > MaxSpread)
    {
        Print("DEBUG: Skipping tick due to high spread: ", SymbolInfoInteger(_Symbol, SYMBOL_SPREAD), " > ", MaxSpread);
        return;
    }

    // Check for new bar
    datetime currentBarTime = iTime(_Symbol, _Period, 0);
    if (currentBarTime == LastBarTime)
    {
        // Uncomment if needed for very detailed logging
        // Print("DEBUG: Skipping tick - not a new bar");
        return;
    }

    Print("DEBUG: Processing new bar at ", TimeToString(currentBarTime), ", last bar was ",
          LastBarTime > 0 ? TimeToString(LastBarTime) : "none");

    // Check for significant time gap using Python's logic
    if (ResetStatesOnGap && LastBarTime > 0)
    {
        datetime expectedTime = LastBarTime + TimeframeMinutes * 60;
        int timeDiff = (int)(currentBarTime - expectedTime);

        if (timeDiff > (TimeframeMinutes * 2 * 60))
        {
            Print("DEBUG: Significant data gap detected (", timeDiff / 60.0, " minutes), resetting LSTM states");
            ArrayInitialize(LSTMState, 0);
        }
    }

    // Calculate features
    double features[];
    FeatureProcessor.ProcessFeatures(features);
    Print("DEBUG: Processed ", ArraySize(features), " features");

    // Add position features
    int baseFeatureCount = ArraySize(features);
    ArrayResize(features, baseFeatureCount + 2);
    features[baseFeatureCount] = (double)CurrentPosition.direction; // Position type

    // Calculate unrealized P&L
    double unrealizedPnl = 0;
    if (CurrentPosition.direction != 0)
    {
        double currentPrice = CurrentPosition.direction == 1 ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        unrealizedPnl = CurrentPosition.direction *
                        (currentPrice - CurrentPosition.entryPrice) /
                        CurrentPosition.entryPrice;
        Print("DEBUG: Current position: ",
              CurrentPosition.direction == 1 ? "LONG" : "SHORT",
              " entry price: ", CurrentPosition.entryPrice,
              " current price: ", currentPrice,
              " unrealized PnL: ", unrealizedPnl);
    }
    features[baseFeatureCount + 1] = MathMax(MathMin(unrealizedPnl, 1.0), -1.0);

    // Log selected key features for analysis
    Print("DEBUG: Key features - Position: ", features[baseFeatureCount],
          ", PnL: ", features[baseFeatureCount + 1]);

    // Run LSTM inference
    double lstm_output[];
    RunLSTMInference(features, LSTMState, lstm_output);

    // Get action with highest probability
    int action = 0;
    double maxProb = lstm_output[0];
    for (int i = 1; i < ACTION_COUNT; i++)
    {
        if (lstm_output[i] > maxProb)
        {
            maxProb = lstm_output[i];
            action = i;
        }
    }

    // Log model output and decision
    string actionDescription = "";
    switch (action)
    {
    case 0:
        actionDescription = "HOLD";
        break;
    case 1:
        actionDescription = "BUY";
        break;
    case 2:
        actionDescription = "SELL";
        break;
    case 3:
        actionDescription = "CLOSE";
        break;
    default:
        actionDescription = "UNKNOWN";
        break;
    }

    Print("DEBUG: Model output probabilities - Hold: ",
          DoubleToString(lstm_output[0], 4), ", Buy: ",
          DoubleToString(lstm_output[1], 4), ", Sell: ",
          DoubleToString(lstm_output[2], 4), ", Close: ",
          DoubleToString(lstm_output[3], 4));
    Print("DEBUG: Selected action: ", action, " (", actionDescription, ") with probability ", DoubleToString(maxProb, 4));

    // Execute trade
    ExecuteTrade(action, features);

    // Update state tracking
    LastBarTime = currentBarTime;
    LastBarIndex++;
    FirstTick = false;
}

// Helper function to convert 2D index to 1D array index
int GetArrayIndex(const int row, const int col, const int cols) {
    return row * cols + col;
}

// Convert a flattened 1D array index to a row
int GetRow(const int index, const int cols) {
    return index / cols;
}

// Convert a flattened 1D array index to a column
int GetCol(const int index, const int cols) {
    return index % cols;
}

// Create a new function for our custom flattening operation
void PrepareWeightsArray(const double &sourceArray[], double &targetArray[], const int inputDim, const int outputDim) {
    ArrayResize(targetArray, inputDim * outputDim);
    ArrayCopy(targetArray, sourceArray);
}

//+------------------------------------------------------------------+
//| Run LSTM inference                                                 |
//+------------------------------------------------------------------+
void RunLSTMInference(const double &features[], double &state[], double &output[])
{
    Print("DEBUG_LSTM: Starting RunLSTMInference with feature count: ", ArraySize(features), 
          ", state size: ", ArraySize(state));
          
    // Temporary arrays for LSTM gates
    double input_gate[];
    double forget_gate[];
    double cell_state[];
    double output_gate[];
    double hidden_state[];

    // Initialize arrays
    int hidden_size = LSTM_UNITS;
    Print("DEBUG_LSTM: Hidden size (LSTM_UNITS): ", hidden_size);
    
    ArrayResize(input_gate, hidden_size);
    ArrayResize(forget_gate, hidden_size);
    ArrayResize(cell_state, hidden_size);
    ArrayResize(output_gate, hidden_size);
    ArrayResize(hidden_state, hidden_size);
    ArrayResize(output, ACTION_COUNT);
    
    Print("DEBUG_LSTM: Arrays resized - input_gate: ", ArraySize(input_gate),
          ", forget_gate: ", ArraySize(forget_gate),
          ", output: ", ArraySize(output),
          ", ACTION_COUNT: ", ACTION_COUNT);
          
    // Create temporary arrays for matrix multiplication operations
    double temp_input_weights[];
    double temp_hidden_weights[];
    double temp_output_weights[];
    
    // Copy weights to temporary arrays to avoid parameter conversion issues
    ArrayResize(temp_input_weights, ArraySize(actor_input_weight));
    ArrayCopy(temp_input_weights, actor_input_weight);
    
    ArrayResize(temp_hidden_weights, ArraySize(actor_hidden_weight));
    ArrayCopy(temp_hidden_weights, actor_hidden_weight);
    
    ArrayResize(temp_output_weights, ArraySize(actor_output_weight));
    ArrayCopy(temp_output_weights, actor_output_weight);
    
    // Actor LSTM - Input transformation
    double actor_input[];
    if (FEATURE_COUNT > 0 && LSTM_UNITS > 0) {
        Print("DEBUG_LSTM: Feature count: ", ArraySize(features), 
              ", FEATURE_COUNT: ", FEATURE_COUNT);
              
        MatrixMultiply(features, temp_input_weights, actor_input,
                      1, FEATURE_COUNT, FEATURE_COUNT, LSTM_UNITS * 4);
                   
        Print("DEBUG_LSTM: actor_input size after multiply: ", ArraySize(actor_input));
    } else {
        Print("ERROR_LSTM: Invalid dimensions for feature processing");
        return;
    }
    
    // Actor LSTM - Hidden transformation
    double actor_hidden_transform[];
    if (LSTM_UNITS > 0) {
        Print("DEBUG_LSTM: State size: ", ArraySize(state),
              ", LSTM_UNITS: ", LSTM_UNITS);
        
        MatrixMultiply(state, temp_hidden_weights, actor_hidden_transform,
                      1, LSTM_UNITS, LSTM_UNITS, LSTM_UNITS * 4);
                   
        Print("DEBUG_LSTM: actor_hidden_transform size after multiply: ", ArraySize(actor_hidden_transform));
    } else {
        Print("ERROR_LSTM: Invalid dimensions for hidden state processing");
        return;
    }

    // Calculate gates
    Print("DEBUG_LSTM: Starting gate calculations for ", LSTM_UNITS, " units");
    for (int i = 0; i < LSTM_UNITS; i++)
    {
        if(i % 10 == 0) Print("DEBUG_LSTM: Processing gate calculations for unit ", i);
        
        // Debug bounds checks
        if(i >= ArraySize(actor_input) || i >= ArraySize(actor_hidden_transform) || 
           i >= ArraySize(actor_hidden_bias)) {
            Print("ERROR_LSTM: Index out of bounds at forget_gate calculation, i=", i, 
                  ", actor_input size: ", ArraySize(actor_input),
                  ", actor_hidden_transform size: ", ArraySize(actor_hidden_transform),
                  ", actor_hidden_bias size: ", ArraySize(actor_hidden_bias));
            return;
        }
        
        int idx = i;
        forget_gate[i] = sigmoid(actor_input[idx] +
                                 actor_hidden_transform[idx] +
                                 actor_hidden_bias[idx]);

        // Debug bounds checks for input_gate
        idx += LSTM_UNITS;
        if(idx >= ArraySize(actor_input) || idx >= ArraySize(actor_hidden_transform) || 
           i >= ArraySize(actor_input_bias)) {
            Print("ERROR_LSTM: Index out of bounds at input_gate calculation, idx=", idx, ", i=", i,
                  ", actor_input size: ", ArraySize(actor_input),
                  ", actor_hidden_transform size: ", ArraySize(actor_hidden_transform),
                  ", actor_input_bias size: ", ArraySize(actor_input_bias));
            return;
        }
        
        input_gate[i] = sigmoid(actor_input[idx] +
                                actor_hidden_transform[idx] +
                                actor_input_bias[i]);

        // Debug bounds checks for cell_state
        idx += LSTM_UNITS;
        if(idx >= ArraySize(actor_input) || idx >= ArraySize(actor_hidden_transform) || 
           i >= ArraySize(actor_input_bias)) {
            Print("ERROR_LSTM: Index out of bounds at cell_state calculation, idx=", idx, ", i=", i,
                  ", actor_input size: ", ArraySize(actor_input),
                  ", actor_hidden_transform size: ", ArraySize(actor_hidden_transform),
                  ", actor_input_bias size: ", ArraySize(actor_input_bias));
            return;
        }
        
        cell_state[i] = custom_tanh(actor_input[idx] +
                                    actor_hidden_transform[idx] +
                                    actor_input_bias[i]);

        // Debug bounds checks for output_gate
        idx += LSTM_UNITS;
        if(idx >= ArraySize(actor_input) || idx >= ArraySize(actor_hidden_transform) || 
           i >= ArraySize(actor_hidden_bias)) {
            Print("ERROR_LSTM: Index out of bounds at output_gate calculation, idx=", idx, ", i=", i,
                  ", actor_input size: ", ArraySize(actor_input),
                  ", actor_hidden_transform size: ", ArraySize(actor_hidden_transform),
                  ", actor_hidden_bias size: ", ArraySize(actor_hidden_bias));
            return;
        }
        
        output_gate[i] = sigmoid(actor_input[idx] +
                                 actor_hidden_transform[idx] +
                                 actor_hidden_bias[i]);
    }

    // Update cell and hidden states
    Print("DEBUG_LSTM: Updating cell and hidden states");
    for (int i = 0; i < LSTM_UNITS; i++)
    {
        if(i % 10 == 0) Print("DEBUG_LSTM: Updating state for unit ", i);
        
        if(i >= ArraySize(forget_gate) || i >= ArraySize(state) || 
           i >= ArraySize(input_gate) || i >= ArraySize(cell_state)) {
            Print("ERROR_LSTM: Index out of bounds at state update, i=", i,
                  ", forget_gate size: ", ArraySize(forget_gate),
                  ", state size: ", ArraySize(state),
                  ", input_gate size: ", ArraySize(input_gate),
                  ", cell_state size: ", ArraySize(cell_state));
            return;
        }
        
        cell_state[i] = forget_gate[i] * state[i] +
                        input_gate[i] * cell_state[i];
        hidden_state[i] = output_gate[i] * custom_tanh(cell_state[i]);
    }

    // Update LSTM state for next iteration
    Print("DEBUG_LSTM: Copying hidden state to state array");
    ArrayCopy(state, hidden_state);

    // Calculate final output with temporary weights array
    Print("DEBUG_LSTM: Starting final output calculation");
          
    MatrixMultiply(hidden_state, temp_output_weights, output,
                   1, LSTM_UNITS, LSTM_UNITS, ACTION_COUNT);
                   
    Print("DEBUG_LSTM: Output size after matrix multiply: ", ArraySize(output));

    // Add bias and apply softmax
    Print("DEBUG_LSTM: Applying softmax with bias");
    if(ArraySize(output) != ACTION_COUNT || ArraySize(actor_output_bias) < ACTION_COUNT) {
        Print("ERROR_LSTM: Array size mismatch before softmax - output size: ", ArraySize(output),
              ", actor_output_bias size: ", ArraySize(actor_output_bias),
              ", ACTION_COUNT: ", ACTION_COUNT);
        return;
    }
    
    double sum = 0;
    for (int i = 0; i < ACTION_COUNT; i++)
    {
        output[i] = MathExp(output[i] + actor_output_bias[i]);
        sum += output[i];
    }

    // Normalize probabilities
    Print("DEBUG_LSTM: Normalizing probabilities, sum=", sum);
    if (sum > 0)
    {
        for (int i = 0; i < ACTION_COUNT; i++)
        {
            output[i] /= sum;
        }
    }
    
    Print("DEBUG_LSTM: LSTM inference completed successfully");
}
