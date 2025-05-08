using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using System.IO;
using cAlgo.API;
using cAlgo.API.Indicators;
using DRLTrader.Models;
using DRLTrader.Services;
using System.Reflection;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class DRLTrader : Robot
    {
        [Parameter("ONNX Model Path", DefaultValue = @"Models\synapse.onnx", Group = "Model")]
        public string OnnxModelPath { get; set; }
        
        [Parameter("Is Recurrent Model", DefaultValue = false, Group = "Model")]
        public bool IsRecurrentModel { get; set; }
        
        [Parameter("Hidden Size", DefaultValue = 64, Group = "Model")]
        public int HiddenSize { get; set; }
        
        [Parameter("LSTM Layers", DefaultValue = 1, Group = "Model")]
        public int LstmLayers { get; set; }

        [Parameter("Balance Per Lot", DefaultValue = 1000.0, Group = "Risk")]
        public double BalancePerLot { get; set; }

        [Parameter("Stop Loss Pips", DefaultValue = 2500.0, Group = "Risk")]
        public double StopLossPips { get; set; }

        [Parameter("Minimum Bars", DefaultValue = 100, Group = "Data")]
        public int MinimumBars { get; set; }

        private OnnxModelPredictor _onnxPredictor;
        private RiskManager _riskManager;
        private readonly Queue<Bar> _bars;
        private bool _isInitialized;

        private const string POSITION_LABEL = "DRLTrader";

        public DRLTrader()
        {
            _bars = new Queue<Bar>();
        }

        protected override void OnStart()
        {
            try
            {
                Print("==== DRLTrader Startup - Detailed Diagnostic Log ====");
                Print($"Current time: {DateTime.Now}");
                Print($"Robot version: {Assembly.GetExecutingAssembly().GetName().Version}");
                Print($"Symbol: {Symbol.Name}, TimeFrame: {TimeFrame}");
                Print($"Account: {Account.Number}, Balance: {Account.Balance}");
                Print($"Robot parameters:");
                Print($"  - ONNX Model Path: {OnnxModelPath}");
                Print($"  - Is Recurrent Model: {IsRecurrentModel}");
                Print($"  - Hidden Size: {HiddenSize}");
                Print($"  - LSTM Layers: {LstmLayers}");
                Print($"  - Balance Per Lot: {BalancePerLot}");
                Print($"  - Stop Loss Pips: {StopLossPips}");
                Print($"  - Minimum Bars: {MinimumBars}");
                
                // Get execution directory with multiple fallbacks
                string executionDir = null;
                
                // Try approach 1: Assembly location
                try {
                    executionDir = Path.GetDirectoryName(GetType().Assembly.Location);
                    Print($"Attempted to get directory from Assembly.Location: {executionDir ?? "null"}");
                } catch (Exception ex) {
                    Print($"Error getting Assembly.Location: {ex.Message}");
                }
                
                // Try approach 2: Base directory if first approach failed
                if (string.IsNullOrEmpty(executionDir)) {
                    try {
                        executionDir = AppDomain.CurrentDomain.BaseDirectory;
                        Print($"Using AppDomain.BaseDirectory: {executionDir ?? "null"}");
                    } catch (Exception ex) {
                        Print($"Error getting AppDomain.BaseDirectory: {ex.Message}");
                    }
                }
                
                // Try approach 3: Current directory as last resort
                if (string.IsNullOrEmpty(executionDir)) {
                    try {
                        executionDir = Directory.GetCurrentDirectory();
                        Print($"Using current directory: {executionDir ?? "null"}");
                    } catch (Exception ex) {
                        Print($"Error getting current directory: {ex.Message}");
                    }
                }
                
                // Try approach 4: Hard-coded path based on cTrader's typical structure
                if (string.IsNullOrEmpty(executionDir)) {
                    string cAlgoDocsPath = Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                        "cAlgo");
                    
                    string[] possiblePaths = new[] {
                        Path.Combine(cAlgoDocsPath, "Sources", "Robots", "DRLTrader"),
                        Path.Combine(cAlgoDocsPath, "cBots", "DRLTrader")
                    };
                    
                    foreach (string path in possiblePaths) {
                        if (Directory.Exists(path)) {
                            executionDir = path;
                            Print($"Using hard-coded path: {executionDir}");
                            break;
                        }
                    }
                }
                
                Print($"Final execution directory: {executionDir ?? "null"}");
                
                // Check if execution directory exists
                if (string.IsNullOrEmpty(executionDir) || !Directory.Exists(executionDir))
                {
                    Print($"ERROR: Could not find a valid execution directory. Trying to continue anyway.");
                    
                    // As a last resort, try to find the model directly
                    string directModelPath = OnnxModelPath;
                    if (File.Exists(directModelPath)) {
                        Print($"Found model directly at specified path: {directModelPath}");
                        
                        try
                        {
                            _onnxPredictor = new OnnxModelPredictor(
                                directModelPath,
                                IsRecurrentModel, 
                                HiddenSize,
                                LstmLayers);
                            
                            Print("ONNX predictor initialized successfully with direct path");
                            
                            // Continue with initialization
                            InitializeRemainingComponents();
                            return;
                        }
                        catch (Exception ex)
                        {
                            Print($"ERROR initializing ONNX predictor with direct path: {ex.Message}");
                            Print($"Stack trace: {ex.StackTrace}");
                        }
                    }
                    
                    Print("ERROR: All attempts to find the model failed. Cannot continue.");
                    return;
                }
                
                // Resolve the absolute model path relative to the bot folder
                string absModelPath = Path.Combine(executionDir, OnnxModelPath);
                Print($"Full model path resolved to: {absModelPath}");
                
                // Check if model file exists
                if (!File.Exists(absModelPath))
                {
                    Print($"ERROR: ONNX model file not found at: {absModelPath}");
                    Print("Checking for model file in alternative locations...");
                    
                    // Try to find in Models subfolder directly
                    string altPath = Path.Combine(executionDir, "Models", Path.GetFileName(OnnxModelPath));
                    Print($"Checking alternative path: {altPath}");
                    
                    if (File.Exists(altPath))
                    {
                        Print($"Found model at alternative path: {altPath}");
                        absModelPath = altPath;
                    }
                    else
                    {
                        // Try various locations where the model might be
                        string[] possibleLocations = {
                            Path.Combine(executionDir, "bin", "Debug", "net6.0", "Models", Path.GetFileName(OnnxModelPath)),
                            Path.Combine(executionDir, "bin", "Release", "net6.0", "Models", Path.GetFileName(OnnxModelPath)),
                            Path.Combine(executionDir, "bin", "Debug", "net6.0", OnnxModelPath),
                            Path.Combine(executionDir, "bin", "Release", "net6.0", OnnxModelPath),
                            // Try just the filename in executionDir
                            Path.Combine(executionDir, Path.GetFileName(OnnxModelPath))
                        };
                        
                        foreach (string path in possibleLocations) {
                            if (File.Exists(path)) {
                                Print($"Found model at location: {path}");
                                absModelPath = path;
                                break;
                            }
                        }
                        
                        if (!File.Exists(absModelPath)) {
                            // List files in the execution directory to help diagnose
                            Print("Listing files in execution directory:");
                            try
                            {
                                foreach (string file in Directory.GetFiles(executionDir))
                                {
                                    Print($"  - {file}");
                                }
                                
                                string modelsDir = Path.Combine(executionDir, "Models");
                                if (Directory.Exists(modelsDir))
                                {
                                    Print($"Listing files in Models directory:");
                                    foreach (string file in Directory.GetFiles(modelsDir))
                                    {
                                        Print($"  - {file}");
                                    }
                                }
                                else
                                {
                                    Print($"Models directory not found at: {modelsDir}");
                                }
                            }
                            catch (Exception ex)
                            {
                                Print($"Error listing files: {ex.Message}");
                            }
                            
                            Print("ERROR: Model file not found in any location. Cannot continue.");
                            return;
                        }
                    }
                }

                Print($"Initializing ONNX predictor with model: {absModelPath}");
                Print($"Model config - IsRecurrent: {IsRecurrentModel}, HiddenSize: {HiddenSize}, Layers: {LstmLayers}");
                
                try
                {
                    _onnxPredictor = new OnnxModelPredictor(
                        absModelPath,
                        IsRecurrentModel, 
                        HiddenSize,
                        LstmLayers);
                    
                    // Set the logger to use ctrader's Print method to ensure logs appear in ctrader
                    _onnxPredictor.SetLogger(Print);
                    
                    Print("ONNX predictor initialized successfully");
                }
                catch (Exception ex)
                {
                    Print($"ERROR initializing ONNX predictor: {ex.Message}");
                    Print($"Stack trace: {ex.StackTrace}");
                    return;
                }
                
                // Initialize the remaining components
                InitializeRemainingComponents();
            }
            catch (Exception ex)
            {
                Print("==== FATAL ERROR during DRLTrader initialization ====");
                Print($"Error message: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                
                // Check for inner exception
                if (ex.InnerException != null)
                {
                    Print($"Inner exception: {ex.InnerException.Message}");
                    Print($"Inner exception stack trace: {ex.InnerException.StackTrace}");
                }
                
                Print("==== End of error report ====");
            }
        }
        
        private void InitializeRemainingComponents()
        {
            try
            {
                Print($"Initializing Risk Manager with BalancePerLot: {BalancePerLot}, StopLossPips: {StopLossPips}");
                _riskManager = new RiskManager(BalancePerLot, StopLossPips, Symbol);
                Print("Risk Manager initialized successfully");
            }
            catch (Exception ex)
            {
                Print($"ERROR initializing Risk Manager: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                return;
            }

            try
            {
                // Load historical data
                Print($"Loading initial {MinimumBars} historical bars...");
                LoadHistoricalData();
                Print($"Historical data loaded successfully: {_bars.Count} bars");
            }
            catch (Exception ex)
            {
                Print($"ERROR loading historical data: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                return;
            }

            _isInitialized = true;
            Print("==== DRLTrader initialization completed successfully ====");
        }

        protected override void OnBar()
        {
            if (!_isInitialized)
            {
                Print("OnBar skipped - Robot not initialized");
                return;
            }

            try
            {
                Print($"==== OnBar processing at {Time} ====");
                
                // Add new bar to queue
                _bars.Enqueue(Bars.Last());
                if (_bars.Count > MinimumBars)
                    _bars.Dequeue();

                // Only proceed if we have enough data
                if (_bars.Count < MinimumBars)
                {
                    Print($"Waiting for more data: {_bars.Count}/{MinimumBars} bars");
                    return;
                }

                Print("Getting current position info...");
                // Get current position directly from the TradeManager
                var position = Positions.FirstOrDefault(p => p.SymbolName == Symbol.Name && p.Label == POSITION_LABEL);
                
                // Calculate position direction and PnL
                int positionDirection = 0;
                double positionPnl = 0.0;
                
                if (position != null) 
                {
                    // Set direction based on the current managed position
                    positionDirection = position.TradeType == TradeType.Buy ? 1 : -1;
                    
                    // Get PnL from the current position and normalize to [-1, 1] range
                    // Use account balance for normalization instead of fixed value
                    positionPnl = Math.Clamp(position.NetProfit / Account.Balance, -1.0, 1.0);
                    
                    Print($"Current position: Direction={positionDirection}, PnL={positionPnl}, ID={position.Id}, Raw PnL={position.NetProfit}, Balance={Account.Balance}");
                }
                else
                {
                    Print($"No open position found: Direction={positionDirection}, PnL={positionPnl}");
                }

                // Create market data
                Print("Preparing market data...");
                var marketData = new MarketData
                {
                    Symbol = Symbol.Name,
                    PositionDirection = positionDirection,
                    PositionPnl = positionPnl
                };

                // Add historical data with optional small perturbation
                Random rnd = new Random();
                bool useRandomPerturbation = true; // Set to true to apply random noise to make predictions more diverse
                double perturbationScale = 0.0001; // Small scale to avoid changing actual market structure
                
                foreach (var bar in _bars)
                {
                    // Apply small random perturbation to price data to get varied model responses
                    double perturbation = useRandomPerturbation ? ((rnd.NextDouble() * 2 - 1) * perturbationScale) : 0;
                    
                    marketData.Timestamp.Add(new DateTimeOffset(bar.OpenTime).ToUnixTimeSeconds());
                    marketData.Open.Add(bar.Open * (1 + perturbation));
                    marketData.High.Add(bar.High * (1 + perturbation));
                    marketData.Low.Add(bar.Low * (1 + perturbation));
                    marketData.Close.Add(bar.Close * (1 + perturbation));
                    marketData.Volume.Add(bar.TickVolume);
                }
                Print($"Market data prepared with {_bars.Count} bars");
                
                // Reset the LSTM state occasionally to avoid getting stuck in one state
                if (IsRecurrentModel && rnd.NextDouble() < 0.05) // 5% chance to reset
                {
                    Print("Randomly resetting LSTM state to prevent getting stuck in predictions");
                    _onnxPredictor.ResetLstmState();
                }
                
                PredictionResponse prediction;
                try
                {
                    Print("Getting prediction from ONNX model...");
                    prediction = _onnxPredictor.GetPrediction(marketData);
                    Print($"Received prediction: {prediction.Action} ({prediction.Description})");
                }
                catch (Exception ex)
                {
                    Print($"ERROR getting prediction from ONNX model: {ex.Message}");
                    Print($"Stack trace: {ex.StackTrace}");
                    return;
                }

                // Execute trade based on prediction
                Print("Executing trade based on prediction...");
                try
                {
                    bool success = ExecuteTradeAsync(prediction, position).Result;
                    if (!success)
                        Print("Trade execution failed");
                    else
                        Print("Trade execution successful");
                }
                catch (Exception ex)
                {
                    Print($"ERROR executing trade: {ex.Message}");
                    Print($"Stack trace: {ex.StackTrace}");
                }
                
                Print($"==== OnBar processing completed at {Time} ====");
            }
            catch (Exception ex)
            {
                Print($"==== ERROR in OnBar at {Time} ====");
                Print($"Error message: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                
                // Check for inner exception
                if (ex.InnerException != null)
                {
                    Print($"Inner exception: {ex.InnerException.Message}");
                    Print($"Inner exception stack trace: {ex.InnerException.StackTrace}");
                }
                
                Print("==== End of error report ====");
            }
        }

        /// Execute trade based on prediction
        /// </summary>
        public async Task<bool> ExecuteTradeAsync(PredictionResponse prediction, Position position)
        {
            try
            {
                // Parse the action
                if (!Enum.TryParse(prediction.Action, true, out TradingAction action))
                {
                    Print($"Invalid action received: {prediction.Action}");
                    return false;
                }

                // Check for current position
                bool hasPosition = position != null;
                
                if (hasPosition)
                {
                    Print($"Current position: {position.Id}, {position.TradeType}, {position.VolumeInUnits} units, PnL: {position.NetProfit}");
                }

                switch (action)
                {
                    case TradingAction.Hold:
                        Print("Hold signal - no trade execution");
                        return true;

                    case TradingAction.Close:
                        if (hasPosition)
                        {
                            return await ClosePositionAsync(position);
                        }
                        else
                        {
                            Print("No position to close");
                            return true;
                        }

                    case TradingAction.Buy:
                        if (hasPosition)
                        {
                            // Ignore Buy signal when any position is open
                            Print("Buy signal received but a position is already open - ignoring");
                            return true;
                        }
                        else
                        {
                            return await OpenPositionAsync(action);
                        }
                        
                    case TradingAction.Sell:
                        if (hasPosition)
                        {
                            // Ignore Sell signal when any position is open
                            Print("Sell signal received but a position is already open - ignoring");
                            return true;
                        }
                        else
                        {
                            return await OpenPositionAsync(action);
                        }

                    default:
                        Print($"Unsupported action: {action}");
                        return false;
                }
            }
            catch (Exception ex)
            {
                Print($"Error executing trade: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                return false;
            }
        }

        /// <summary>
        /// Open a new position based on the trading action
        /// </summary>
        private Task<bool> OpenPositionAsync(TradingAction action)
        {
            try
            {
                // Calculate position size
                double volume = _riskManager.CalculatePositionSize(Account.Balance);
                
                // Get current market price
                double entryPrice = action == TradingAction.Buy ? Symbol.Ask : Symbol.Bid;
                
                // Calculate stop loss
                TradeType tradeType = action == TradingAction.Buy ? TradeType.Buy : TradeType.Sell;
                double stopLoss = _riskManager.CalculateStopLoss(entryPrice, tradeType);
                
                // Open position
                var result = ExecuteMarketOrder(
                    tradeType,
                    Symbol.Name,
                    volume,
                    POSITION_LABEL,
                    stopLoss,
                    null  // No take profit
                );
                
                if (result.IsSuccessful)
                {
                    Print($"Position opened: {action} {volume:F2} lots @ {entryPrice:F5} (SL: {stopLoss:F5})");
                    return Task.FromResult(true);
                }
                else
                {
                    Print($"Failed to open position: {result.Error}");
                    return Task.FromResult(false);
                }
            }
            catch (Exception ex)
            {
                Print($"Error opening position: {ex.Message}");
                return Task.FromResult(false);
            }
        }        /// <summary>
        /// Close current position if it exists
        /// </summary>
        private Task<bool> ClosePositionAsync(Position position)
        {
            try
            {
                
                if (position == null)
                {
                    Print("No position to close");
                    return Task.FromResult(true);
                }

                Print($"Attempting to close position {position.Id}...");
                var result = ClosePosition(position);
                if (result.IsSuccessful)
                {
                    Print($"Position {position.Id} closed successfully");
                    return Task.FromResult(true);
                }
                else
                {
                    Print($"Failed to close position: {result.Error}");
                    return Task.FromResult(false);
                }
            }
            catch (Exception ex)
            {
                Print($"Error closing position: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
                return Task.FromResult(false);
            }
        }


        protected override void OnStop()
        {
            Print("==== DRLTrader stopping ====");
            try
            {
                _onnxPredictor?.Dispose();
                Print("ONNX predictor disposed successfully");
            }
            catch (Exception ex)
            {
                Print($"Error disposing ONNX predictor: {ex.Message}");
            }
            Print("==== DRLTrader stopped ====");
        }

        private void LoadHistoricalData()
        {
            // Load initial historical bars
            var historicalBars = Bars.Take(MinimumBars).ToList();
            foreach (var bar in historicalBars)
            {
                _bars.Enqueue(bar);
            }
            Print($"Loaded {_bars.Count} historical bars");
        }
    }
}
