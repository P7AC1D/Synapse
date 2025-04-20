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

// Constants
#define MAGIC_NUMBER 20240417
#define STOP_LOSS_PIPS 1500.0
#define ERR_FUNCTION_NOT_ALLOWED_IN_TESTING_MODE 4806

// Input parameters
input string ApiGroup = ">>> API Settings <<<";
input string ApiUrl = "http://localhost:8000";  // API base URL
input int MinDataBars = 500;                    // Minimum data bars to collect

input string TradingGroup = ">>> Trading Settings <<<";
input int MaxSpread = 350;                      // Maximum allowed spread (points)
input double BalancePerLot = 2500.0;            // Amount required per 0.01 lot

// Indicators
int rsi_handle;
int atr_handle;
int bb_handle;
int adx_handle;

// Indicator parameters
int atr_period = 14;
int rsi_period = 14;
int boll_period = 20;
int adx_period = 14;

// Global variables
CTrade Trade;                   // Trading object
string last_error = "";         // Last error message

// HTTP request related
int http_timeout = 5000;        // Timeout in milliseconds
string http_headers;            // HTTP headers

// Position tracking
struct Position {
    int direction;      // 1 for long, -1 for short, 0 for none
    double entryPrice;  // Position entry price
    double lotSize;     // Position size in lots
    datetime entryTime; // Entry timestamp
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

//+------------------------------------------------------------------+
//| Initialize indicators                                              |
//+------------------------------------------------------------------+
bool InitializeIndicators() {
    // Initialize standard MT5 indicators
    rsi_handle = iRSI(_Symbol, _Period, rsi_period, PRICE_CLOSE);
    atr_handle = iATR(_Symbol, _Period, atr_period);
    bb_handle = iBands(_Symbol, _Period, boll_period, 0, 2, PRICE_CLOSE);
    adx_handle = iADX(_Symbol, _Period, adx_period);
    
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
//| Build JSON request for API                                        |
//+------------------------------------------------------------------+
string BuildApiRequest() {
    string json = "{";
    
    // Arrays for data
    json += "\"timestamp\":[";
    json += BuildJsonArray(time_values, true);
    json += "],";
    
    json += "\"open\":[";
    json += BuildJsonArray(open_prices);
    json += "],";
    
    json += "\"high\":[";
    json += BuildJsonArray(high_prices);
    json += "],";
    
    json += "\"low\":[";
    json += BuildJsonArray(low_prices);
    json += "],";
    
    json += "\"close\":[";
    json += BuildJsonArray(close_prices);
    json += "],";
    
    json += "\"volume\":[";
    json += BuildJsonArray(volume_values);
    json += "],";
    
    // Symbol
    json += "\"symbol\":\"" + _Symbol + "\"";
    
    json += "}";
    return json;
}

//+------------------------------------------------------------------+
//| Helper function to build JSON arrays                              |
//+------------------------------------------------------------------+
string BuildJsonArray(const double &arr[], bool isTimestamp=false) {
    string result = "";
    int size = ArraySize(arr);
    
    for(int i = 0; i < size; i++) {
        if(isTimestamp) {
            result += IntegerToString((long)arr[i]);
        } else {
            result += DoubleToString(arr[i], 8);
        }
        if(i < size - 1) result += ",";
    }
    
    return result;
}

string BuildJsonArray(const datetime &arr[], bool isTimestamp=true) {
    string result = "";
    int size = ArraySize(arr);
    
    for(int i = 0; i < size; i++) {
        if(isTimestamp) {
            result += IntegerToString((long)arr[i]);
        } else {
            result += "\"" + TimeToString(arr[i]) + "\"";
        }
        if(i < size - 1) result += ",";
    }
    
    return result;
}

string BuildJsonArray(const long &arr[]) {
    string result = "";
    int size = ArraySize(arr);
    
    for(int i = 0; i < size; i++) {
        result += IntegerToString(arr[i]);
        if(i < size - 1) result += ",";
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Make prediction API call                                          |
//+------------------------------------------------------------------+
bool GetPrediction(string &action, string &description) {
    // Build API request
    string request_body = BuildApiRequest();
    char request[];
    char result[];
    string result_headers;
    
    // Convert request string to char array
    StringToCharArray(request_body, request, 0, StringLen(request_body));
    
    // Setup headers
    string headers = "Content-Type: application/json\r\n";
    
    // Make POST request to API
    ResetLastError();
    int res = WebRequest(
        "POST",                // method
        ApiUrl + "/predict",   // URL
        headers,              // headers
        5000,                // timeout
        request,             // request data
        result,              // response data
        result_headers       // response headers
    );
    
    if(res == -1) {
        int error_code = GetLastError();
        string error_desc = "HTTP request failed with error " + IntegerToString(error_code);
        
        switch(error_code) {
            case ERR_FUNCTION_NOT_ALLOWED:
                error_desc += "\nMake sure URL is added to 'Tools' -> 'Options' -> 'Expert Advisors' -> 'Allow WebRequest'";
                break;
        }
        
        last_error = error_desc;
        Print(last_error);
        return false;
    }
    
    // Parse response
    string response = CharArrayToString(result);
    
    // Parse JSON response manually since MQL5 doesn't have a built-in JSON parser
    if(StringFind(response, "\"action\"") >= 0 && StringFind(response, "\"description\"") >= 0) {
        // Extract action
        int action_start = StringFind(response, "\"action\":\"") + 9;
        int action_end = StringFind(response, "\"", action_start);
        action = StringSubstr(response, action_start, action_end - action_start);
        
        // Extract description
        int desc_start = StringFind(response, "\"description\":\"") + 14;
        int desc_end = StringFind(response, "\"", desc_start);
        description = StringSubstr(response, desc_start, desc_end - desc_start);
        
        Print("API Prediction: Action=", action, ", Description=", description);
        return true;
    }
    
    last_error = "Failed to parse JSON response: " + response;
    Print(last_error);
    return false;
}

//+------------------------------------------------------------------+
//| Execute trades based on API prediction                            |
//+------------------------------------------------------------------+
void ExecuteTrade(const string &action, const string &description) {
    double lotSize = CalculateLotSize();
    if(lotSize <= 0) return;
    
    // Map action string to action code
    int action_code = 0; // Hold by default
    if (action == "buy") action_code = 1;
    else if (action == "sell") action_code = 2;
    else if (action == "close") action_code = 3;
    
    // Debug output
    string action_names[] = {"Hold", "Buy", "Sell", "Close"};
    Print("Selected action: ", action_names[action_code], " (", action_code, ") with description: ", description);
    
    switch(action_code) {
        case 1: // Buy
            if(CurrentPosition.direction == 0) {
                double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double stopLoss = CalculateStopLoss(askPrice, true);
                
                if(Trade.Buy(lotSize, _Symbol, 0, stopLoss, 0, "API_BUY: " + description)) {
                    CurrentPosition.direction = 1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
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
                
                if(Trade.Sell(lotSize, _Symbol, 0, stopLoss, 0, "API_SELL: " + description)) {
                    CurrentPosition.direction = -1;
                    CurrentPosition.entryPrice = Trade.ResultPrice();
                    CurrentPosition.lotSize = lotSize;
                    CurrentPosition.entryTime = TimeCurrent();
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
    }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Initializing DRLTrader with API connection...");
    
    // Initialize indicators
    if(!InitializeIndicators()) {
        Print("Failed to initialize indicators");
        return INIT_FAILED;
    }
    
    // Check if URL is allowed for web requests
    string terminal_path = TerminalInfoString(TERMINAL_PATH);
    string config_path = terminal_path + "\\config\\webserver.ini";
    
    if(FileIsExist(config_path)) {
        int file_handle = FileOpen(config_path, FILE_READ|FILE_TXT);
        if(file_handle != INVALID_HANDLE) {
            bool url_found = false;
            while(!FileIsEnding(file_handle)) {
                string line = FileReadString(file_handle);
                if(StringFind(line, ApiUrl) >= 0) {
                    url_found = true;
                    break;
                }
            }
            FileClose(file_handle);
            
            if(!url_found) {
                Print("Web requests not allowed. Please enable in Tools > Options > Expert Advisors > Allow WebRequest for URL:");
                Print(ApiUrl);
                return INIT_FAILED;
            }
        }
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
                    Print("Recovered existing position: ", 
                          CurrentPosition.direction > 0 ? "LONG" : "SHORT", " ",
                          CurrentPosition.lotSize, " lots @ ", CurrentPosition.entryPrice);
                    break;
                }
            }
        }
    }
    
    Print("DRLTrader initialized with API URL: ", ApiUrl);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    ReleaseIndicators();
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
    
    // Get prediction from API
    string action = "hold";
    string description = "";
    if(!GetPrediction(action, description)) {
        Print("Failed to get prediction from API: ", last_error);
        return;
    }
    
    // Execute trades based on prediction
    ExecuteTrade(action, description);
}
