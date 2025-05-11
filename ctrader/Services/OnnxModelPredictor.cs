using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DRLTrader.Models;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DRLTrader.Services
{
    /// <summary>
    /// Provides local inference using ONNX models via ML.NET
    /// </summary>
    public class OnnxModelPredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly bool _isRecurrentModel;
        private float[] _lstmHidden;
        private float[] _lstmCell;
        private readonly int _hiddenSize;
        private readonly int _numLayers;
        private bool _disposed;
        private Action<string> _logger;

        /// <summary>
        /// Initialize a new ONNX model predictor
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file</param>
        /// <param name="isRecurrent">Whether the model is recurrent (LSTM-based)</param>
        /// <param name="hiddenSize">For recurrent models, the hidden state size</param>
        /// <param name="numLayers">For recurrent models, the number of LSTM layers</param>
        public OnnxModelPredictor(string modelPath, bool isRecurrent = false, int hiddenSize = 64, int numLayers = 1)
        {
            // Use Console.WriteLine as default logger since cAlgo Print method is not accessible here
            _logger = Console.WriteLine;
            
            _logger($"==== Initializing ONNX Model Predictor ====");
            _logger($"Model path: {modelPath}");
            _logger($"Is recurrent model: {isRecurrent}");
            _logger($"Hidden size: {hiddenSize}");
            _logger($"Number of LSTM layers: {numLayers}");
            
            // Check file existence with more details
            try
            {
                _logger($"Checking if model file exists...");
                if (!File.Exists(modelPath))
                {
                    _logger($"ERROR: ONNX model not found at: {modelPath}");
                    _logger($"Current directory: {Directory.GetCurrentDirectory()}");
                    _logger($"File path is absolute: {Path.IsPathRooted(modelPath)}");
                    
                    // Try to list files in model directory
                    try
                    {
                        string modelDirectory = Path.GetDirectoryName(modelPath);
                        if (Directory.Exists(modelDirectory))
                        {
                            _logger($"Model directory exists: {modelDirectory}");
                            _logger($"Files in model directory:");
                            foreach (var file in Directory.GetFiles(modelDirectory))
                            {
                                _logger($"  - {file}");
                            }
                        }
                        else
                        {
                            _logger($"Model directory does not exist: {modelDirectory}");
                        }
                    }
                    catch (Exception dirEx)
                    {
                        _logger($"Error listing files in model directory: {dirEx.Message}");
                    }
                    
                    throw new FileNotFoundException($"ONNX model not found at: {modelPath}");
                }
                _logger($"Model file exists: {modelPath}");
                _logger($"Model file size: {new FileInfo(modelPath).Length} bytes");
            }
            catch (Exception ex)
            {
                _logger($"ERROR checking model file: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                throw;
            }

            _isRecurrentModel = isRecurrent;
            _hiddenSize = hiddenSize;
            _numLayers = numLayers;

            // Initialize LSTM states if using a recurrent model
            if (isRecurrent)
            {
                _logger($"Initializing LSTM states for recurrent model");
                _lstmHidden = new float[numLayers * 1 * hiddenSize]; // [num_layers, batch_size, hidden_size]
                _lstmCell = new float[numLayers * 1 * hiddenSize];   // [num_layers, batch_size, hidden_size]
                _logger($"LSTM states initialized with shapes: [{numLayers}, 1, {hiddenSize}]");
            }

            // Create inference session with the ONNX model
            try
            {
                _logger($"Creating ONNX inference session...");
                var sessionOptions = new SessionOptions();
                _logger($"Setting session options...");
                sessionOptions.EnableMemoryPattern = true;
                sessionOptions.EnableCpuMemArena = true;
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                _logger($"Session options set: MemPattern={sessionOptions.EnableMemoryPattern}, CpuArena={sessionOptions.EnableCpuMemArena}, OptLevel={sessionOptions.GraphOptimizationLevel}");

                _logger($"Loading model into inference session...");
                byte[] modelBytes;
                try
                {
                    modelBytes = File.ReadAllBytes(modelPath);
                    _logger($"Model file loaded into memory: {modelBytes.Length} bytes");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR reading model file: {ex.Message}");
                    _logger($"Stack trace: {ex.StackTrace}");
                    throw;
                }

                try
                {
                    _session = new InferenceSession(modelBytes, sessionOptions);
                    _logger($"Inference session created successfully");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR creating inference session: {ex.Message}");
                    _logger($"Stack trace: {ex.StackTrace}");
                    
                    // Check for specific known errors
                    if (ex.Message.Contains("DLL") || ex.Message.Contains("dependency"))
                    {
                        _logger("This may be a missing dependency issue. Check that Microsoft Visual C++ Redistributable is installed.");
                    }
                    
                    throw;
                }

                // Log model metadata
                _logger($"ONNX model loaded: {modelPath}");
                _logger($"Model inputs ({_session.InputMetadata.Count}):");
                foreach (var input in _session.InputMetadata)
                {
                    _logger($"  - {input.Key}: {string.Join("x", input.Value.Dimensions)}");
                }
                
                _logger($"Model outputs ({_session.OutputMetadata.Count}):");
                foreach (var output in _session.OutputMetadata)
                {
                    _logger($"  - {output.Key}: {string.Join("x", output.Value.Dimensions)}");
                }
                
                _logger($"==== ONNX Model Predictor initialized successfully ====");
            }
            catch (Exception ex)
            {
                _logger($"==== FAILED to initialize ONNX Model Predictor ====");
                _logger($"Error: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                
                // Check for inner exception
                if (ex.InnerException != null)
                {
                    _logger($"Inner exception: {ex.InnerException.Message}");
                    _logger($"Inner exception stack trace: {ex.InnerException.StackTrace}");
                }
                throw;
            }
        }

        /// <summary>
        /// Set the logger to use for all messages
        /// </summary>
        /// <param name="logger">The action to call to log messages</param>
        public void SetLogger(Action<string> logger)
        {
            if (logger != null)
            {
                _logger = logger;
                _logger("Logger set to custom implementation");
            }
        }

        /// <summary>
        /// Get prediction from the ONNX model
        /// </summary>
        public PredictionResponse GetPrediction(MarketData data)
        {
            try
            {
                _logger($"Getting prediction for symbol: {data.Symbol}, Bars: {data.Close.Count}");
                
                // Validate input data
                if (data == null)
                {
                    _logger("ERROR: Null MarketData passed to GetPrediction");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR: Null input data, defaulting to Hold"
                    };
                }

                if (data.Close == null || data.Close.Count == 0 || 
                    data.Open == null || data.Open.Count == 0 ||
                    data.High == null || data.High.Count == 0 ||
                    data.Low == null || data.Low.Count == 0 ||
                    data.Volume == null || data.Volume.Count == 0 ||
                    data.Timestamp == null || data.Timestamp.Count == 0)
                {
                    _logger("ERROR: MarketData contains null or empty collections");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR: Invalid market data, defaulting to Hold"
                    };
                }
                
                // Call the appropriate prediction method
                if (_isRecurrentModel)
                {
                    _logger("Using recurrent model for prediction");
                    return GetRecurrentPrediction(data);
                }
                else
                {
                    _logger("Using standard model for prediction");
                    return GetStandardPrediction(data);
                }
            }
            catch (Exception ex)
            {
                _logger($"ERROR in GetPrediction: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                
                // Return safe default rather than crashing
                return new PredictionResponse
                {
                    Action = "Hold",
                    Confidence = 1.0f,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    Description = $"ERROR: {ex.Message}, defaulting to Hold"
                };
            }
        }

        /// <summary>
        /// Get prediction using a standard (non-recurrent) model
        /// </summary>
        private PredictionResponse GetStandardPrediction(MarketData data)
        {
            try
            {
                _logger("Preprocessing features for standard model...");
                // Create input tensor with shape [1, features]
                var features = PreprocessFeatures(data);
                _logger($"Preprocessed features count: {features.Length}");
                
                var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });
                _logger($"Created input tensor with shape: [1, {features.Length}]");

                // Create input dictionary for ONNX inference
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("observation", inputTensor)
                };
                _logger("Created input dictionary for inference");
                
                // Log inputs for diagnostic purposes
                _logger("Input names for model:");
                foreach (var input in _session.InputMetadata)
                {
                    _logger($"  - {input.Key}: {string.Join("x", input.Value.Dimensions)}");
                }
                _logger("Expected output names:");
                foreach (var output in _session.OutputMetadata)
                {
                    _logger($"  - {output.Key}: {string.Join("x", output.Value.Dimensions)}");
                }

                // Run inference
                _logger("Running inference...");
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                try
                {
                    results = _session.Run(inputs);
                    _logger("Inference completed successfully");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR during model inference: {ex.Message}");
                    _logger($"Stack trace: {ex.StackTrace}");
                    
                    // Return safe default since inference failed
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = $"ERROR during model inference: {ex.Message}, defaulting to Hold"
                    };
                }
                
                // Get the output tensor (action probabilities)
                _logger("Processing output...");
                
                // Check if we have any results
                if (results == null || !results.Any())
                {
                    _logger("ERROR: Model returned no results");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR: Model returned no results, defaulting to Hold"
                    };
                }
                
                // Log all available output names
                _logger("Available output names in results:");
                foreach (var r in results)
                {
                    _logger($"  - {r.Name}");
                }
                
                // Use First() safely
                DisposableNamedOnnxValue firstOutput;
                try
                {
                    firstOutput = results.First();
                    _logger($"Retrieved first output with name: {firstOutput.Name}");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR retrieving first output: {ex.Message}");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR accessing model output, defaulting to Hold"
                    };
                }
                
                // Convert to tensor safely
                Tensor<float> outputTensor;
                try
                {
                    outputTensor = firstOutput.AsTensor<float>();
                    _logger($"Output tensor retrieved successfully");
                    _logger($"Output tensor shape: [{string.Join(", ", outputTensor.Dimensions.ToArray())}]");
                    _logger($"Output tensor length: {outputTensor.Length}");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR converting output to tensor: {ex.Message}");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR converting model output, defaulting to Hold"
                    };
                }
                
                // Check if tensor has elements before accessing them
                if (outputTensor.Length == 0)
                {
                    _logger("ERROR: Output tensor is empty");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR: Empty output tensor, defaulting to Hold"
                    };
                }
                
                // Log all action probabilities for debugging
                _logger("Raw action probabilities:");
                try {
                    bool is2DTensor = outputTensor.Dimensions.Length == 2 && outputTensor.Dimensions[0] == 1;
                    
                    for (int i = 0; i < Math.Min(outputTensor.Length, 4); i++) // Safety: limit to 4 elements max (our action space)
                    {
                        TradingAction actionType = MapIndexToAction(i);
                        float value = is2DTensor ? outputTensor[0, i] : outputTensor[i];
                        _logger($"  - {actionType}: {value:P2}");
                    }
                }
                catch (Exception ex) {
                    _logger($"ERROR logging action probabilities: {ex.Message}");
                }

                // Get the action with highest probability (same as argmax in Python)
                int actionIndex = GetMaxProbabilityIndex(outputTensor);
                TradingAction selectedAction = MapIndexToAction(actionIndex);
                
                // Safely get confidence
                float confidence;
                try {
                    // Check if tensor is 2D [1,4] or 1D [4]
                    bool is2DTensor = outputTensor.Dimensions.Length == 2 && outputTensor.Dimensions[0] == 1;
                    confidence = is2DTensor ? outputTensor[0, actionIndex] : outputTensor[actionIndex];
                    _logger($"Confidence value retrieved successfully: {confidence:F6} ({confidence:P2})");
                }
                catch (Exception ex) {
                    _logger($"ERROR retrieving confidence value: {ex.Message}");
                    confidence = 1.0f;
                }
                
                // Safely get timestamp
                long timestamp;
                try {
                    timestamp = data.Timestamp.Last();
                }
                catch (Exception) {
                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                    _logger("WARNING: Could not get timestamp from data, using current time");
                }
                
                _logger($"Final prediction: Action={selectedAction}, Confidence={confidence:P2}, Timestamp={timestamp}");
                
                return new PredictionResponse
                {
                    Action = selectedAction.ToString(),
                    Confidence = confidence,
                    Timestamp = timestamp,
                    Description = $"Local ONNX prediction: {selectedAction} ({confidence:P2})"
                };
            }
            catch (Exception ex)
            {
                _logger($"ERROR in GetStandardPrediction: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                return new PredictionResponse
                {
                    Action = "Hold",
                    Confidence = 1.0f,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    Description = $"ERROR in standard prediction: {ex.Message}, defaulting to Hold"
                };
            }
        }

        /// <summary>
        /// Get prediction using a recurrent (LSTM) model
        /// </summary>
        private PredictionResponse GetRecurrentPrediction(MarketData data)
        {
            try 
            {
                _logger("Preprocessing sequence features for recurrent model...");
                // Create sequence input tensor with shape [1, sequence_length, features]
                var featuresSequence = PreprocessSequenceFeatures(data);
                
                var batchSize = 1;
                var seqLength = data.Close.Count;
                var featuresPerBar = featuresSequence.Length / seqLength;
                
                _logger($"Sequence shape parameters: batch_size={batchSize}, seq_length={seqLength}, features_per_bar={featuresPerBar}");
                _logger($"Total features in sequence: {featuresSequence.Length}");
                
                var inputTensor = new DenseTensor<float>(
                    featuresSequence, 
                    new[] { batchSize, seqLength, featuresPerBar });
                _logger($"Created input tensor with shape: [1, {seqLength}, {featuresPerBar}]");

                // Create tensors for LSTM state
                _logger("Creating LSTM state tensors...");
                var hiddenTensor = new DenseTensor<float>(_lstmHidden, new[] { _numLayers, batchSize, _hiddenSize });
                var cellTensor = new DenseTensor<float>(_lstmCell, new[] { _numLayers, batchSize, _hiddenSize });
                _logger($"Created LSTM state tensors with shapes: [{_numLayers}, {batchSize}, {_hiddenSize}]");

                // Create input dictionary
                _logger("Creating input dictionary...");
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("observation", inputTensor),
                    NamedOnnxValue.CreateFromTensor("lstm_h", hiddenTensor),
                    NamedOnnxValue.CreateFromTensor("lstm_c", cellTensor)
                };
                
                // Log inputs for diagnostic purposes
                _logger("Input names for model:");
                foreach (var input in _session.InputMetadata)
                {
                    _logger($"  - {input.Key}: {string.Join("x", input.Value.Dimensions)}");
                }
                _logger("Expected output names:");
                foreach (var output in _session.OutputMetadata)
                {
                    _logger($"  - {output.Key}: {string.Join("x", output.Value.Dimensions)}");
                }

                // Run inference with error handling
                _logger("Running inference...");
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                try
                {
                    results = _session.Run(inputs);
                    _logger("Inference completed successfully");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR during model inference: {ex.Message}");
                    _logger($"Stack trace: {ex.StackTrace}");
                    
                    // Return safe default since inference failed
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = $"ERROR during recurrent inference: {ex.Message}, defaulting to Hold"
                    };
                }
                
                // Check if we have results
                if (results == null || !results.Any())
                {
                    _logger("ERROR: Recurrent model returned no results");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = "ERROR: Recurrent model returned no results, defaulting to Hold"
                    };
                }
                
                // Log all available output names
                _logger("Available output names in results:");
                foreach (var r in results)
                {
                    _logger($"  - {r.Name}");
                }
                
                // Get action probabilities with error handling
                _logger("Processing outputs...");
                Tensor<float> outputTensor;
                try
                {
                    var actionProbs = results.FirstOrDefault(x => x.Name == "action_probs");
                    if (actionProbs == null)
                    {
                        _logger("ERROR: 'action_probs' output not found in model results");
                        // Try to use first output as fallback
                        actionProbs = results.First();
                        _logger($"Using fallback output: {actionProbs.Name}");
                    }
                    
                    outputTensor = actionProbs.AsTensor<float>();
                    _logger($"Action probs tensor retrieved successfully");
                    _logger($"Action probs tensor shape: [{string.Join(", ", outputTensor.Dimensions.ToArray())}]");
                    _logger($"Action probs tensor length: {outputTensor.Length}");
                    
                    // Check if tensor has elements before accessing them
                    if (outputTensor.Length == 0)
                    {
                        _logger("ERROR: Output tensor is empty");
                        return new PredictionResponse
                        {
                            Action = "Hold",
                            Confidence = 1.0f,
                            Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                            Description = "ERROR: Empty output tensor, defaulting to Hold"
                        };
                    }
                }
                catch (Exception ex)
                {
                    _logger($"ERROR getting action probabilities: {ex.Message}");
                    return new PredictionResponse
                    {
                        Action = "Hold",
                        Confidence = 1.0f,
                        Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        Description = $"ERROR getting action probabilities: {ex.Message}, defaulting to Hold"
                    };
                }
                
                // Log all probabilities for debugging
                _logger("Raw action probabilities:");
                try
                {
                    bool is2DTensor = outputTensor.Dimensions.Length == 2 && outputTensor.Dimensions[0] == 1;
                    
                    for (int i = 0; i < Math.Min(outputTensor.Length, 4); i++) // Safety: limit to 4 elements max
                    {
                        TradingAction actionType = MapIndexToAction(i);
                        float value = is2DTensor ? outputTensor[0, i] : outputTensor[i];
                        _logger($"  - {actionType}: {value:P2}");
                    }
                }
                catch (Exception ex)
                {
                    _logger($"ERROR logging action probabilities: {ex.Message}");
                }
                
                // Get updated LSTM states with error handling
                try
                {
                    var newHiddenState = results.FirstOrDefault(x => x.Name == "new_lstm_h");
                    var newCellState = results.FirstOrDefault(x => x.Name == "new_lstm_c");
                    
                    if (newHiddenState != null && newCellState != null)
                    {
                        var newHiddenTensor = newHiddenState.AsTensor<float>();
                        var newCellTensor = newCellState.AsTensor<float>();
                        _logger("Retrieved updated LSTM states");
                        
                        // Update internal states
                        _lstmHidden = newHiddenTensor.ToArray();
                        _lstmCell = newCellTensor.ToArray();
                        _logger("Updated internal LSTM states");
                    }
                    else
                    {
                        _logger("WARNING: Could not find LSTM state outputs, keeping current state");
                    }
                }
                catch (Exception ex)
                {
                    _logger($"ERROR updating LSTM states: {ex.Message}");
                    // Continue processing without updating states
                }
                
                // Process the output - simply take the highest probability action
                int actionIndex = GetMaxProbabilityIndex(outputTensor);
                TradingAction selectedAction = MapIndexToAction(actionIndex);
                
                // Safely get confidence
                float confidence;
                try
                {
                    bool is2DTensor = outputTensor.Dimensions.Length == 2 && outputTensor.Dimensions[0] == 1;
                    confidence = is2DTensor ? outputTensor[0, actionIndex] : outputTensor[actionIndex];
                    _logger($"Confidence value retrieved successfully: {confidence:F6} ({confidence:P2})");
                }
                catch (Exception ex)
                {
                    _logger($"ERROR retrieving confidence value: {ex.Message}");
                    confidence = 1.0f;
                }
                
                // Safely get timestamp
                long timestamp;
                try
                {
                    timestamp = data.Timestamp.Last();
                }
                catch (Exception)
                {
                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                    _logger("WARNING: Could not get timestamp from data, using current time");
                }
                
                _logger($"Final prediction: Action={selectedAction}, Confidence={confidence:P2}, Timestamp={timestamp}");
                
                return new PredictionResponse
                {
                    Action = selectedAction.ToString(),
                    Confidence = confidence,
                    Timestamp = timestamp,
                    Description = $"Local ONNX prediction: {selectedAction} ({confidence:P2})"
                };
            }
            catch (Exception ex)
            {
                _logger($"ERROR in GetRecurrentPrediction: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                return new PredictionResponse
                {
                    Action = "Hold",
                    Confidence = 1.0f,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    Description = $"ERROR in recurrent prediction: {ex.Message}, defaulting to Hold"
                };
            }
        }

        /// <summary>
        /// Reset LSTM states to zero (useful between episodes or when market conditions change drastically)
        /// </summary>
        public void ResetLstmState()
        {
            if (_isRecurrentModel)
            {
                _logger("Resetting LSTM states to zero");
                Array.Clear(_lstmHidden, 0, _lstmHidden.Length);
                Array.Clear(_lstmCell, 0, _lstmCell.Length);
                _logger("LSTM states reset completed");
            }
        }

        /// <summary>
        /// Preprocess features for a single timestep
        /// </summary>
        private float[] PreprocessFeatures(MarketData data)
        {
            try
            {
                _logger("Preprocessing features for single timestep...");
                
                // Validate data
                if (data == null || data.Close == null || data.Close.Count == 0)
                {
                    _logger("ERROR: Invalid MarketData");
                    throw new ArgumentException("Invalid market data");
                }
                
                // Make sure all data arrays have the same length
                if (data.Open.Count != data.Close.Count || 
                    data.High.Count != data.Close.Count || 
                    data.Low.Count != data.Close.Count ||
                    data.Volume.Count != data.Close.Count ||
                    data.Timestamp.Count != data.Close.Count)
                {
                    _logger("ERROR: Inconsistent data lengths in MarketData");
                    throw new InvalidOperationException("Inconsistent data lengths in market data");
                }
                
                _logger($"Creating features following exact Python order for {data.Symbol}");
                  // Convert to arrays for calculation
                double[] close = data.Close.Select(d => (double)d).ToArray();
                double[] open = data.Open.Select(d => (double)d).ToArray();
                double[] high = data.High.Select(d => (double)d).ToArray();
                double[] low = data.Low.Select(d => (double)d).ToArray();
                double[] volume = data.Volume.Select(d => (double)d).ToArray();

                int lastIdx = 0;
                
                // Create features in EXACT order from Python's get_feature_names():
                // [returns, rsi, atr, volume_change, volatility_breakout, trend_strength, 
                //  candle_pattern, sin_time, cos_time, position_type, unrealized_pnl]
                List<float> features = new List<float>();

                  // 1. returns: [-0.1, 0.1] Price momentum
                float returns = 0f;
                if (data.Close.Count > 1 && close[1] != 0)
                {
                    // Calculate returns using most recent close (index 0) divided by previous close (index 1)
                    returns = (float)((close[0] / close[1]) - 1.0);                    
                    returns = Math.Clamp(returns, -0.1f, 0.1f);
                }
                else
                {
                    _logger("Not enough bars to calculate returns, using default value 0");
                }
                features.Add(returns);

                  // 2. rsi: [-1, 1] Momentum oscillator
                // Use RSI value directly from MarketData instead of manual calculation
                float rsi = 50f; // Default to neutral
                if (data.RSI != null && data.RSI.Count > lastIdx)
                {
                    rsi = (float)data.RSI[lastIdx];
                }
                else
                {
                    _logger("Using default neutral RSI (50) as no valid RSI data available");
                }

                // Normalize RSI exactly like Python: (rsi/50)-1
                float rsiNorm = rsi / 50f - 1f;
                features.Add(rsiNorm);

                  // 3. atr: [-1, 1] Volatility indicator
                // Use ATR value directly from MarketData instead of manual calculation
                float atr = 0f;
                float atrNorm = 0f;
                
                if (data.ATR != null && data.ATR.Count > lastIdx)
                {
                    atr = (float)data.ATR[lastIdx];
                    
                    // Apply ATR normalization to match Python's implementation
                    float[] atrWindow = data.ATR.Take(Math.Min(20, data.ATR.Count))
                                          .Select(x => (float)x)
                                          .ToArray();
                    
                    // Calculate average ATR like Python's bfill().fillna(atr.mean())
                    float atrMean = atrWindow.Average();
                    atr = float.IsNaN(atr) ? atrMean : atr;
                    
                    // Calculate SMA like Python's rolling(window_size).mean()
                    float atrSma = atrWindow.Average();
                    float atrRatio = atrSma != 0 ? atr / atrSma : 1f;
                    float minExpectedRatio = 0.5f;
                    float maxExpectedRatio = 2.0f;
                    float expectedRange = maxExpectedRatio - minExpectedRatio;
                    atrNorm = 2f * (atrRatio - minExpectedRatio) / expectedRange - 1f;
                    atrNorm = Math.Clamp(atrNorm, -1f, 1f);
                }
                else
                {
                    _logger("Using default ATR normalization (0) as no valid ATR data available");
                }
                
                features.Add(atrNorm);

                  // 4. volume_change: [-1, 1] Volume percentage change
                float volumeChange = 0f;
                if (data.Volume.Count > 1 && volume[1] != 0)
                {
                    // Calculate volume change using rate of change
                    volumeChange = (float)((volume[0] - volume[1]) / volume[1]);  // Current volume [0] minus previous volume [1]
                    volumeChange = Math.Clamp(volumeChange, -1f, 1f);
                }
                features.Add(volumeChange);                // 5. volatility_breakout: [0, 1] Trend with volatility context
                float volatilityBreakout = 0.5f; // Default to middle
                // These variables are declared outside the conditional for logging
                double upperBand = 0;
                double lowerBand = 0;
                double sma = 0;

                // Use Bollinger Bands directly from MarketData if available
                if (data.BollingerUpper != null && data.BollingerUpper.Count > lastIdx &&
                    data.BollingerLower != null && data.BollingerLower.Count > lastIdx &&
                    data.BollingerMiddle != null && data.BollingerMiddle.Count > lastIdx)
                {
                    upperBand = data.BollingerUpper[lastIdx];
                    lowerBand = data.BollingerLower[lastIdx];
                    sma = data.BollingerMiddle[lastIdx];

                    // Calculate position in band - EXACTLY as in Python
                    double bandRange = upperBand - lowerBand;
                    if (bandRange > 1e-8) // Avoid division by very small numbers
                    {
                        double position = close[lastIdx] - lowerBand;
                        volatilityBreakout = (float)(position / bandRange);
                        volatilityBreakout = Math.Clamp(volatilityBreakout, 0f, 1f);
                    }
                }
                else
                {
                    _logger("MarketData Bollinger Bands not available, falling back to manual calculation");
                    
                    if (lastIdx >= 20) // Need at least 20 bars for Bollinger Bands
                    {
                        // Calculate 20-period SMA
                        double sum = 0;
                        for (int i = 0; i < 20; i++)
                        {
                            sum += close[lastIdx - i];
                        }
                        sma = sum / 20;
                        
                        // Calculate standard deviation
                        double sumSquares = 0;
                        for (int i = 0; i < 20; i++)
                        {
                            sumSquares += Math.Pow(close[lastIdx - i] - sma, 2);
                        }
                        double stdDev = Math.Sqrt(sumSquares / 20);
                        
                        upperBand = sma + (2 * stdDev);
                        lowerBand = sma - (2 * stdDev);
                        
                        // Calculate position in band - EXACTLY as in Python
                        double bandRange = upperBand - lowerBand;
                        if (bandRange > 1e-8) // Avoid division by very small numbers
                        {
                            double position = close[lastIdx] - lowerBand;
                            volatilityBreakout = (float)(position / bandRange);
                            volatilityBreakout = Math.Clamp(volatilityBreakout, 0f, 1f);
                            _logger($"Volatility breakout calculated manually: {volatilityBreakout:F6}");
                        }
                    }
                    else
                    {
                        _logger("Not enough bars for Bollinger Bands calculation, using default value 0.5");
                    }
                }
                features.Add(volatilityBreakout);

                  // 6. trend_strength: [-1, 1] ADX-based trend quality
                float trendStrength = 0f; // Default to no trend
                
                // Use ADX value directly from MarketData if available
                if (data.ADX != null && data.ADX.Count > lastIdx)
                {
                    float adx = (float)data.ADX[lastIdx];
                    _logger($"Using ADX from MarketData: {adx:F2}");
                    
                    // Normalize exactly like Python code: (adx/25)-1
                    trendStrength = (adx / 25.0f) - 1.0f;
                    trendStrength = Math.Clamp(trendStrength, -1f, 1f);
                }
                else
                {
                    _logger("MarketData ADX not available, falling back to manual calculation");
                    
                    if (lastIdx >= 15) // Need at least 15 bars for meaningful ADX calculation
                    {
                        double posDMSum = 0;
                        double negDMSum = 0;
                        double trSum = 0;
                        
                        for (int i = 1; i <= 14; i++)
                        {
                            int idx = lastIdx - 14 + i;
                            int prevIdx = idx - 1;
                            
                            if (idx > 0 && prevIdx >= 0)
                            {
                                // Directional Movement
                                double upMove = high[idx] - high[prevIdx];
                                double downMove = low[prevIdx] - low[idx];
                                
                                // Positive DM
                                if (upMove > 0 && upMove > downMove)
                                    posDMSum += upMove;
                                
                                // Negative DM
                                if (downMove > 0 && downMove > upMove)
                                    negDMSum += downMove;
                                
                                // True Range for same period
                                double tr1 = high[idx] - low[idx];
                                double tr2 = Math.Abs(high[idx] - close[prevIdx]);
                                double tr3 = Math.Abs(low[idx] - close[prevIdx]);
                                trSum += Math.Max(Math.Max(tr1, tr2), tr3);
                            }
                        }
                        
                        // Calculate DI values
                        if (trSum > 0)
                        {
                            double posDI = (posDMSum / trSum) * 100;
                            double negDI = (negDMSum / trSum) * 100;
                            
                            if (posDI + negDI > 0)
                            {
                                // Calculate DX
                                double dx = Math.Abs(posDI - negDI) / (posDI + negDI) * 100;
                                
                                // Normalize exactly like Python code: (dx/25)-1
                                trendStrength = (float)(dx / 25.0 - 1.0);
                                trendStrength = Math.Clamp(trendStrength, -1f, 1f);
                                _logger($"ADX (trend strength) calculated manually: {trendStrength:F6}");
                            }
                        }
                    }
                    else
                    {
                        _logger("Not enough bars for ADX calculation, using default value 0");
                    }
                }
                features.Add(trendStrength);

                  // 7. candle_pattern: [-1, 1] Combined price action signal
                float body = (float)(close[lastIdx] - open[lastIdx]);
                float upperWick = (float)(high[lastIdx] - Math.Max(close[lastIdx], open[lastIdx]));
                float lowerWick = (float)(Math.Min(close[lastIdx], open[lastIdx]) - low[lastIdx]);
                float range = (float)(high[lastIdx] - low[lastIdx] + 1e-8); // Avoid div by zero
                
                float candlePattern;
                if (upperWick + lowerWick > 1e-8)
                {
                    // Exactly match Python: (body/range + (upper_wick-lower_wick)/(upper_wick+lower_wick))/2
                    candlePattern = (body / range + (upperWick - lowerWick) / (upperWick + lowerWick)) / 2f;
                }
                else
                {
                    candlePattern = body / range / 2f;
                }
                candlePattern = Math.Clamp(candlePattern, -1f, 1f);
                features.Add(candlePattern);
                
                // 8. sin_time: [-1, 1] Sine encoding of time
                // 9. cos_time: [-1, 1] Cosine encoding of time
                DateTimeOffset barTimeOffset;
                try
                {
                    barTimeOffset = DateTimeOffset.FromUnixTimeSeconds(data.Timestamp[lastIdx]);
                }
                catch
                {
                    barTimeOffset = DateTimeOffset.UtcNow;
                    _logger("WARNING: Using current time because timestamp conversion failed");
                }
                var barTime = barTimeOffset.DateTime;
                
                // Use exact same time encoding as Python: minutes in day
                double minutesInDay = 24.0 * 60.0; 
                double timeIndex = barTime.Hour * 60.0 + barTime.Minute;
                double timeNormalized = 2.0 * Math.PI * timeIndex / minutesInDay;

                // Debug output for verification
                _logger($"Time: {barTime}, Hours: {barTime.Hour}, Minutes: {barTime.Minute}");
                _logger($"Time index: {timeIndex} minutes, Angle: {timeNormalized} radians");

                // Make sure these match the Python implementation order
                float sinTime = (float)Math.Sin(timeNormalized);
                float cosTime = (float)Math.Cos(timeNormalized);

                // Debug the actual values
                _logger($"sin_time: {sinTime}, cos_time: {cosTime}");

                features.Add(cosTime);  // Add cos_time first to match Python
                features.Add(sinTime);  // Add sin_time second to match Python
                
                // 10. position_type: [-1, 0, 1] Current position
                features.Add((float)data.PositionDirection);
                
                // 11. unrealized_pnl: [-1, 1] Current position P&L
                features.Add(float.IsNaN((float)data.PositionPnl) ? 0f : (float)data.PositionPnl);
                  // Log normalized features similar to Python bot                _logger($"Created {features.Count} features in exact Python order");
                _logger("Normalized features for prediction:");
                _logger($"  returns: {features[0]:F6}");
                _logger($"  rsi: {features[1]:F6}");
                _logger($"  atr: {features[2]:F6}");
                _logger($"  volume_change: {features[3]:F6}");
                _logger($"  volatility_breakout: {features[4]:F6}");
                _logger($"  trend_strength: {features[5]:F6}");
                _logger($"  candle_pattern: {features[6]:F6}");
                _logger($"  cos_time: {features[7]:F6}");
                _logger($"  sin_time: {features[8]:F6}");
                _logger($"  position_type: {features[9]:F6}");
                _logger($"  unrealized_pnl: {features[10]:F6}");                  
                  // Log OHLCV data for the most recent bar (index 0)
                _logger("Most recent bar OHLCV data (index 0):");
                _logger($"Symbol: {data.Symbol}");
                _logger($"Open: {open[0]:F6}");
                _logger($"High: {high[0]:F6}");
                _logger($"Close: {close[0]:F6}");
                _logger($"Low: {low[0]:F6}");
                _logger($"Volume: {volume[0]:F6}");
                
                // Also log previous bar if available (for comparison)
                if (data.Close.Count > 1)
                {
                    _logger("Previous bar OHLCV data (index 1):");
                    _logger($"Open: {open[1]:F6}");
                    _logger($"High: {high[1]:F6}");
                    _logger($"Close: {close[1]:F6}");
                    _logger($"Low: {low[1]:F6}");
                    _logger($"Volume: {volume[1]:F6}");
                    
                    // Format timestamp for previous bar
                    string prevDateTimeStr = "Unknown";
                    try
                    {
                        DateTimeOffset prevDto = DateTimeOffset.FromUnixTimeSeconds(data.Timestamp[1]);
                        prevDateTimeStr = prevDto.ToString("yyyy-MM-dd HH:mm:ss");
                        _logger($"Previous Timestamp: {data.Timestamp[1]} ({prevDateTimeStr})");
                    }
                    catch
                    {
                        _logger("WARNING: Could not format timestamp for previous bar");
                    }
                }
                
                // Format timestamp as a readable date/time for current bar
                string dateTimeStr = "Unknown";
                try
                {
                    DateTimeOffset dto = DateTimeOffset.FromUnixTimeSeconds(data.Timestamp[lastIdx]);
                    dateTimeStr = dto.ToString("yyyy-MM-dd HH:mm:ss");
                }
                catch
                {
                    _logger("WARNING: Could not format timestamp");
                }
                _logger($"Current Timestamp: {data.Timestamp[lastIdx]} ({dateTimeStr})");
                
                // Log raw indicator values for the last bar
                _logger("Raw feature values (last bar):");
                _logger($"ATR: {atr:F6}");
                _logger($"RSI: {rsi:F6}");
                
                // Always log BB values 
                _logger($"BB Upper: {upperBand:F6}");
                _logger($"BB Middle: {sma:F6}");
                _logger($"BB Lower: {lowerBand:F6}");
                
                // Get ADX raw value for logging
                float rawAdx = 0f;
                if (data.ADX != null && data.ADX.Count > lastIdx)
                {
                    rawAdx = (float)data.ADX[lastIdx];
                }
                _logger($"ADX: {rawAdx:F6}");
                _logger($"Trend Strength: {trendStrength:F6}");
                
                // Final validation
                for (int i = 0; i < features.Count; i++)
                {
                    if (float.IsNaN(features[i]) || float.IsInfinity(features[i]))
                    {
                        _logger($"WARNING: Feature {i} is NaN or Infinity, replacing with 0");
                        features[i] = 0f;
                    }
                }
                
                return features.ToArray();
            }
            catch (Exception ex)
            {
                _logger($"ERROR in PreprocessFeatures: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                return new float[11] { 0f, 0f, 0f, 0f, 0.5f, 0f, 0f, 0f, 0f, 0f, 0f };
            }
        }

        /// <summary>
        /// Preprocess sequence features for the whole time series
        /// </summary>
        private float[] PreprocessSequenceFeatures(MarketData data)
        {
            try
            {
                _logger($"Preprocessing sequence features for {data.Close.Count} bars...");
                
                // Validate data
                if (data == null)
                {
                    _logger("ERROR: Received null MarketData");
                    throw new ArgumentNullException(nameof(data));
                }
                
                if (data.Close == null || data.Close.Count == 0)
                {
                    _logger("ERROR: MarketData contains no price data");
                    throw new InvalidOperationException("Market data contains no price data");
                }
                
                // Make sure all data arrays have the same length
                if (data.Open.Count != data.Close.Count || 
                    data.High.Count != data.Close.Count || 
                    data.Low.Count != data.Close.Count ||
                    data.Volume.Count != data.Close.Count ||
                    data.Timestamp.Count != data.Close.Count)
                {
                    _logger("ERROR: Inconsistent data lengths in MarketData");
                    throw new InvalidOperationException("Inconsistent data lengths in market data");
                }
                
                // Convert to arrays for easier calculation
                double[] close = data.Close.Select(d => (double)d).ToArray();
                double[] open = data.Open.Select(d => (double)d).ToArray();
                double[] high = data.High.Select(d => (double)d).ToArray();
                double[] low = data.Low.Select(d => (double)d).ToArray();
                double[] volume = data.Volume.Select(d => (double)d).ToArray();
                
                var features = new List<float>();
                int featuresPerBar = 11; // Should match the number of features in the Python implementation
                
                // Calculate technical indicators for the entire series
                // These arrays will hold indicator values for each bar
                float[] atrValues = new float[data.Close.Count];
                float[] rsiValues = new float[data.Close.Count];
                float[] upperBand = new float[data.Close.Count];
                float[] lowerBand = new float[data.Close.Count];
                float[] adxValues = new float[data.Close.Count];
                
                // Calculate ATR (Average True Range)
                int atrPeriod = 14;
                for (int i = 0; i < data.Close.Count; i++)
                {
                    if (i >= 1)
                    {
                        double tr1 = high[i] - low[i];
                        double tr2 = Math.Abs(high[i] - close[i - 1]);
                        double tr3 = Math.Abs(low[i] - close[i - 1]);
                        double tr = Math.Max(Math.Max(tr1, tr2), tr3);
                        atrValues[i] = (float)tr;
                    }
                    else
                    {
                        atrValues[i] = (float)(high[i] - low[i]); // Fallback for first bar
                    }
                }
                
                // Apply SMA to smooth ATR values
                for (int i = 0; i < data.Close.Count; i++)
                {
                    if (i >= atrPeriod)
                    {
                        float sum = 0;
                        for (int j = 0; j < atrPeriod; j++)
                        {
                            sum += atrValues[i - j];
                        }
                        atrValues[i] = sum / atrPeriod;
                    }
                    else if (i > 0)
                    {
                        // For bars < period, use available data
                        float sum = 0;
                        for (int j = 0; j <= i; j++)
                        {
                            sum += atrValues[j];
                        }
                        atrValues[i] = sum / (i + 1);
                    }
                }
                
                // Calculate RSI
                int rsiPeriod = 14;
                double[] gains = new double[data.Close.Count];
                double[] losses = new double[data.Close.Count];
                
                // First calculate gains and losses
                for (int i = 1; i < data.Close.Count; i++)
                {
                    double change = close[i] - close[i - 1];
                    if (change > 0)
                    {
                        gains[i] = change;
                        losses[i] = 0;
                    }
                    else
                    {
                        gains[i] = 0;
                        losses[i] = -change;
                    }
                }
                
                // Calculate average gains and losses
                double[] avgGains = new double[data.Close.Count];
                double[] avgLosses = new double[data.Close.Count];
                
                for (int i = rsiPeriod; i < data.Close.Count; i++)
                {
                    if (i == rsiPeriod)
                    {
                        // First RSI calculation uses simple average
                        double gainSum = 0;
                        double lossSum = 0;
                        
                        for (int j = 1; j <= rsiPeriod; j++)
                        {
                            gainSum += gains[j];
                            lossSum += losses[j];
                        }
                        
                        avgGains[i] = gainSum / rsiPeriod;
                        avgLosses[i] = lossSum / rsiPeriod;
                    }
                    else
                    {
                        // Subsequent RSI calculations use smoothed average
                        avgGains[i] = (avgGains[i - 1] * (rsiPeriod - 1) + gains[i]) / rsiPeriod;
                        avgLosses[i] = (avgLosses[i - 1] * (rsiPeriod - 1) + losses[i]) / rsiPeriod;
                    }
                    
                    // Calculate RSI
                    if (avgLosses[i] == 0)
                    {
                        rsiValues[i] = 100f;
                    }
                    else
                    {
                        double rs = avgGains[i] / avgLosses[i];
                        rsiValues[i] = (float)(100 - (100 / (1 + rs)));
                    }
                }
                
                // Backfill RSI values for early bars
                for (int i = 0; i < rsiPeriod; i++)
                {
                    rsiValues[i] = 50f; // Neutral RSI
                }
                
                // Calculate Bollinger Bands
                int bollPeriod = 20;
                for (int i = bollPeriod - 1; i < data.Close.Count; i++)
                {
                    // Calculate SMA for the period
                    double sum = 0;
                    for (int j = 0; j < bollPeriod; j++)
                    {
                        sum += close[i - j];
                    }
                    double sma = sum / bollPeriod;
                    
                    // Calculate standard deviation
                    double squareSum = 0;
                    for (int j = 0; j < bollPeriod; j++)
                    {
                        squareSum += Math.Pow(close[i - j] - sma, 2);
                    }
                    double stdDev = Math.Sqrt(squareSum / bollPeriod);
                    
                    // Calculate bands
                    upperBand[i] = (float)(sma + (2 * stdDev));
                    lowerBand[i] = (float)(sma - (2 * stdDev));
                }
                
                // Backfill Bollinger values
                for (int i = 0; i < bollPeriod - 1; i++)
                {
                    upperBand[i] = (float)close[i];
                    lowerBand[i] = (float)close[i];
                }
                
                // Process each bar in the sequence to create feature vectors
                for (int i = 0; i < data.Close.Count; i++)
                {
                    try
                    {
                        // 1. Calculate returns (clipped to [-0.1, 0.1])
                        float returns;
                        if (i > 0 && close[i - 1] != 0)
                        {
                            returns = (float)((close[i] / close[i - 1]) - 1.0);
                            returns = Math.Clamp(returns, -0.1f, 0.1f);
                        }
                        else
                        {
                            returns = 0f;
                        }
                        features.Add(returns);
                        
                        // 2. RSI (normalized to [-1, 1])
                        float rsiNorm = rsiValues[i] / 50f - 1f;
                        features.Add(rsiNorm);
                        
                        // 3. ATR (normalized)
                        // Calculate ATR ratio to its SMA
                        int windowSize = Math.Min(20, i + 1);
                        float atrSma = 0f;
                        
                        if (i >= windowSize)
                        {
                            float sum = 0;
                            for (int j = 0; j < windowSize; j++)
                            {
                                sum += atrValues[i - j];
                            }
                            atrSma = sum / windowSize;
                        }
                        else if (i > 0)
                        {
                            float sum = 0;
                            for (int j = 0; j <= i; j++)
                            {
                                sum += atrValues[j];
                            }
                            atrSma = sum / (i + 1);
                        }
                        else
                        {
                            atrSma = atrValues[i];
                        }
                        
                        float atrRatio = atrSma != 0 ? atrValues[i] / atrSma : 1f;
                        float minExpectedRatio = 0.5f;
                        float maxExpectedRatio = 2.0f;
                        float expectedRange = maxExpectedRatio - minExpectedRatio;
                        float atrNorm = 2f * (atrRatio - minExpectedRatio) / expectedRange - 1f;
                        atrNorm = Math.Clamp(atrNorm, -1f, 1f);
                        features.Add(atrNorm);
                        
                        // 4. Volume change
                        float volumeChange = 0f;
                        if (i > 0 && volume[i - 1] != 0)
                        {
                            volumeChange = (float)((volume[i] - volume[i - 1]) / volume[i - 1]);
                            volumeChange = Math.Clamp(volumeChange, -1f, 1f);
                        }
                        features.Add(volumeChange);
                          // 5. Volatility breakout (Bollinger Band position)
                        float volatilityBreakout = 0.5f; // Default to middle
                        if (i >= bollPeriod - 1)
                        {
                            float bandRange = upperBand[i] - lowerBand[i];
                            // Use exact same 1e-8 threshold as Python implementation
                            if (bandRange > 1e-8)
                            {
                                double position = close[i] - lowerBand[i];
                                volatilityBreakout = (float)(position / bandRange);
                                volatilityBreakout = Math.Clamp(volatilityBreakout, 0f, 1f);
                            }
                        }
                        features.Add(volatilityBreakout);
                        
                        // 6. Trend strength (ADX-like)
                        float trendStrength = 0f; // Default to no trend
                        if (i >= 14) // Use 14 period for ADX approximation
                        {
                            // Simplified ADX calculation using DM from last 14 bars
                            double posDMSum = 0;
                            double negDMSum = 0;
                            double trSum = 0;
                            
                            for (int j = Math.Max(1, i - 13); j <= i; j++)
                            {
                                double posDM = high[j] - high[j - 1];
                                double negDM = low[j - 1] - low[j];
                                
                                if (posDM > 0 && posDM > negDM)
                                    posDMSum += posDM;
                                
                                if (negDM > 0 && negDM > posDM)
                                    negDMSum += negDM;
                                
                                trSum += atrValues[j]; // Use precalculated TR values
                            }
                            
                            if (trSum > 0)
                            {
                                double posDI = (posDMSum / trSum) * 100;
                                double negDI = (negDMSum / trSum) * 100;
                                
                                if (posDI + negDI > 0)
                                {
                                    double dx = Math.Abs(posDI - negDI) / (posDI + negDI) * 100;
                                    // Normalize like Python code (ADX/25 - 1)
                                    trendStrength = (float)(dx / 25.0 - 1.0);
                                    trendStrength = Math.Clamp(trendStrength, -1f, 1f);
                                }
                            }
                        }
                        features.Add(trendStrength);
                          // 7. Calculate candle pattern
                        float body = (float)(close[i] - open[i]);
                        float upperWick = (float)(high[i] - Math.Max(close[i], open[i]));
                        float lowerWick = (float)(Math.Min(close[i], open[i]) - low[i]);
                        float range = (float)(high[i] - low[i] + 1e-8);
                        
                        float candlePattern;
                        if (upperWick + lowerWick > 1e-8) // Match Python's 1e-8 threshold
                        {
                            // Exactly match Python formula
                            candlePattern = (body / range + (upperWick - lowerWick) / (upperWick + lowerWick)) / 2f;
                        }
                        else
                        {
                            candlePattern = body / range / 2f;
                        }
                        candlePattern = Math.Clamp(candlePattern, -1f, 1f);
                        features.Add(candlePattern);
                        
                        // 8-9. Time features using sin/cos encoding
                        DateTimeOffset barTimeOffset;
                        try
                        {
                            barTimeOffset = DateTimeOffset.FromUnixTimeSeconds(data.Timestamp[i]);
                        }
                        catch
                        {
                            barTimeOffset = DateTimeOffset.UtcNow;
                        }
                        var barTime = barTimeOffset.DateTime;
                        
                        double minutesInDay = 24.0 * 60.0;
                        double timeIndex = barTime.Hour * 60.0 + barTime.Minute;
                        double timeNormalized = 2.0 * Math.PI * timeIndex / minutesInDay;
                        
                        features.Add((float)Math.Cos(timeNormalized)); // Cos time first to match Python
                        features.Add((float)Math.Sin(timeNormalized)); // Sin time second to match Python
                        
                        // 10-11. Position direction and PnL
                        features.Add((float)data.PositionDirection);
                        features.Add(float.IsNaN((float)data.PositionPnl) ? 0f : (float)data.PositionPnl);
                    }
                    catch (Exception ex)
                    {
                        _logger($"Error processing bar {i}: {ex.Message}");
                        
                        // Add default features for this bar to maintain consistency
                        int remainingFeatures = featuresPerBar - (features.Count % featuresPerBar);
                        if (remainingFeatures < featuresPerBar)
                        {
                            for (int j = 0; j < remainingFeatures; j++)
                            {
                                features.Add(0f);
                            }
                        }
                    }
                }
                  _logger($"Preprocessed total of {features.Count} features for sequence");
                
                // Log the features from the last bar in the sequence (similar to Python bot)
                if (features.Count >= featuresPerBar)
                {
                    // Get the last bar's features
                    int lastBarStartIndex = features.Count - featuresPerBar;
                    
                    // Log normalized features
                    _logger("Normalized features for last timestep prediction:");
                    _logger($"  returns: {features[lastBarStartIndex]:F6}");
                    _logger($"  rsi: {features[lastBarStartIndex + 1]:F6}");
                    _logger($"  atr: {features[lastBarStartIndex + 2]:F6}");
                    _logger($"  volume_change: {features[lastBarStartIndex + 3]:F6}");
                    _logger($"  volatility_breakout: {features[lastBarStartIndex + 4]:F6}");
                    _logger($"  trend_strength: {features[lastBarStartIndex + 5]:F6}");
                    _logger($"  candle_pattern: {features[lastBarStartIndex + 6]:F6}");
                    _logger($"  sin_time: {features[lastBarStartIndex + 7]:F6}");
                    _logger($"  cos_time: {features[lastBarStartIndex + 8]:F6}");
                    _logger($"  position_type: {features[lastBarStartIndex + 9]:F6}");
                    _logger($"  unrealized_pnl: {features[lastBarStartIndex + 10]:F6}");
                      // Log raw indicator values for the last bar
                    int lastBarIndex = data.Close.Count - 1;
                    _logger("Raw feature values (last bar):");
                    _logger($"ATR: {atrValues[lastBarIndex]:F6}");
                    _logger($"RSI: {rsiValues[lastBarIndex]:F6}");
                    _logger($"BB Upper: {upperBand[lastBarIndex]:F6}");
                    _logger($"BB Lower: {lowerBand[lastBarIndex]:F6}");
                    // For trend strength, adxValues array may not be populated, so use the feature value
                    float trendStrength = features[lastBarStartIndex + 5];
                    _logger($"Trend Strength: {trendStrength:F6}");
                }
                
                // Final validation - ensure we have the correct number of features
                int expectedFeatureCount = data.Close.Count * featuresPerBar;
                if (features.Count != expectedFeatureCount)
                {
                    _logger($"WARNING: Feature count mismatch. Expected {expectedFeatureCount}, got {features.Count}");
                    
                    // Pad or truncate to correct size
                    if (features.Count < expectedFeatureCount)
                    {
                        _logger("Padding feature array with zeros");
                        features.AddRange(Enumerable.Repeat(0f, expectedFeatureCount - features.Count));
                    }
                    else if (features.Count > expectedFeatureCount)
                    {
                        _logger("Truncating feature array to expected size");
                        features = features.Take(expectedFeatureCount).ToList();
                    }
                }
                
                // Replace any NaN or Infinity values
                for (int i = 0; i < features.Count; i++)
                {
                    if (float.IsNaN(features[i]) || float.IsInfinity(features[i]))
                    {
                        features[i] = 0f;
                    }
                }
                
                return features.ToArray();
            }
            catch (Exception ex)
            {
                _logger($"CRITICAL ERROR in PreprocessSequenceFeatures: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                
                // Return a safe default feature vector instead of throwing
                _logger("Returning fallback feature vector to avoid crashing");
                int minFeaturesNeeded = data?.Close?.Count > 0 ? data.Close.Count * 11 : 11;
                return Enumerable.Repeat(0f, minFeaturesNeeded).ToArray();
            }
        }

        /// <summary>
        /// Get the index with the highest probability
        /// </summary>
        private int GetMaxProbabilityIndex(Tensor<float> tensor)
        {
            try
            {
                if (tensor == null)
                {
                    _logger("WARNING: Received null tensor in GetMaxProbabilityIndex");
                    return 0; // Default to Hold action
                }
                
                if (tensor.Length == 0)
                {
                    _logger("WARNING: Received empty tensor in GetMaxProbabilityIndex");
                    return 0; // Default to Hold action
                }
                
                _logger($"Finding max probability among {tensor.Length} values with tensor shape [{string.Join(", ", tensor.Dimensions.ToArray())}]");
                
                // Handle tensor shape properly. If it's a 2D tensor with shape [1, N], we need to access elements differently
                bool is2DTensor = tensor.Dimensions.Length == 2 && tensor.Dimensions[0] == 1;
                
                // Detailed logging of all probabilities
                _logger("DETAILED ACTION PROBABILITIES:");
                for (int i = 0; i < Math.Min(tensor.Length, 4); i++) // Limit to 4 actions max
                {
                    TradingAction actionType = MapIndexToAction(i);
                    float value = is2DTensor ? tensor[0, i] : tensor[i]; // Access correctly based on shape
                    _logger($"  - Index {i} ({actionType}): {value:F6} ({value:P2})");
                }
                
                // Find the actual max value
                int maxIndex = 0;
                float maxValue = is2DTensor ? tensor[0, 0] : tensor[0];
                
                for (int i = 1; i < Math.Min(tensor.Length, 4); i++) // Limit to 4 actions max
                {
                    float value = is2DTensor ? tensor[0, i] : tensor[i];
                    if (value > maxValue)
                    {
                        maxValue = value;
                        maxIndex = i;
                    }
                }
                
                TradingAction selectedAction = MapIndexToAction(maxIndex);
                _logger($"Max probability found: index={maxIndex}, action={selectedAction}, value={maxValue:F6} ({maxValue:P2})");
                
                return maxIndex;
            }
            catch (Exception ex)
            {
                _logger($"ERROR in GetMaxProbabilityIndex: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                return 0; // Default to Hold action in case of error
            }
        }

        /// <summary>
        /// Map the model's output index to a trading action
        /// </summary>
        private TradingAction MapIndexToAction(int index)
        {
            _logger($"Mapping action index {index} to trading action");
            
            // Check for invalid indices
            if (index < 0)
            {
                _logger($"WARNING: Negative action index {index}, defaulting to Hold");
                return TradingAction.Hold;
            }
            
            // This mapping should match your model's training setup
            switch (index)
            {
                case 0:
                    _logger("Mapped index 0 to Hold");
                    return TradingAction.Hold;
                case 1:
                    _logger("Mapped index 1 to Buy");
                    return TradingAction.Buy;
                case 2:
                    _logger("Mapped index 2 to Sell");
                    return TradingAction.Sell;
                case 3:
                    _logger("Mapped index 3 to Close");
                    return TradingAction.Close;
                default:
                    _logger($"WARNING: Unknown action index {index}, defaulting to Hold");
                    return TradingAction.Hold;
            }
        }

        /// <summary>
        /// Log feature values for debugging and transparency
        /// </summary>
        private void LogFeatureValues(float[] features, Dictionary<string, float> rawIndicators)
        {
            if (features == null || features.Length == 0)
            {
                _logger("Cannot log features: Empty feature array");
                return;
            }

            // Log normalized features
            _logger("Normalized features for prediction:");
            
            // Define feature names
            string[] featureNames = { 
                "returns", "rsi", "atr", "volume_change", "volatility_breakout", 
                "trend_strength", "candle_pattern", "sin_time", "cos_time", 
                "position_type", "unrealized_pnl" 
            };
            
            // Log normalized values by name
            for (int i = 0; i < Math.Min(features.Length, featureNames.Length); i++)
            {
                _logger($"  {featureNames[i]}: {features[i]:F6}");
            }
            
            // Log raw indicator values
            _logger("Raw feature values (last bar):");
            if (rawIndicators != null)
            {
                if (rawIndicators.ContainsKey("ATR"))
                    _logger($"ATR: {rawIndicators["ATR"]:F6}");
                if (rawIndicators.ContainsKey("RSI"))
                    _logger($"RSI: {rawIndicators["RSI"]:F6}");
                if (rawIndicators.ContainsKey("BB_Upper"))
                    _logger($"BB Upper: {rawIndicators["BB_Upper"]:F6}");
                if (rawIndicators.ContainsKey("BB_Lower"))
                    _logger($"BB Lower: {rawIndicators["BB_Lower"]:F6}");
                if (rawIndicators.ContainsKey("Trend_Strength"))
                    _logger($"Trend Strength: {rawIndicators["Trend_Strength"]:F6}");
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _logger?.Invoke("Disposing ONNX Model Predictor resources");
                    _session?.Dispose();
                    _logger?.Invoke("ONNX session disposed");
                }
                _disposed = true;
            }
        }

        ~OnnxModelPredictor()
        {
            Dispose(false);
        }
    }
}
