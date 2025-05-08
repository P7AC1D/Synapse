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

                // Check that all collections have the same length
                int referenceLength = data.Close.Count;
                if (data.Open.Count != referenceLength || 
                    data.High.Count != referenceLength || 
                    data.Low.Count != referenceLength ||
                    data.Volume.Count != referenceLength ||
                    data.Timestamp.Count != referenceLength)
                {
                    _logger($"WARNING: Inconsistent data lengths - Close:{data.Close.Count}, Open:{data.Open.Count}, " +
                           $"High:{data.High.Count}, Low:{data.Low.Count}, Volume:{data.Volume.Count}, Timestamp:{data.Timestamp.Count}");
                    
                    // Try to adjust the data to make it consistent
                    int minLength = new[] { 
                        data.Close.Count, data.Open.Count, data.High.Count, 
                        data.Low.Count, data.Volume.Count, data.Timestamp.Count 
                    }.Min();
                    
                    _logger($"Adjusting all data collections to minimum length: {minLength}");
                    
                    if (minLength == 0)
                    {
                        _logger("ERROR: Cannot process zero-length data");
                        return new PredictionResponse
                        {
                            Action = "Hold",
                            Confidence = 1.0f,
                            Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                            Description = "ERROR: Zero-length data, defaulting to Hold"
                        };
                    }

                    // Truncate all collections to the minimum length
                    while (data.Close.Count > minLength) data.Close.RemoveAt(data.Close.Count - 1);
                    while (data.Open.Count > minLength) data.Open.RemoveAt(data.Open.Count - 1);
                    while (data.High.Count > minLength) data.High.RemoveAt(data.High.Count - 1);
                    while (data.Low.Count > minLength) data.Low.RemoveAt(data.Low.Count - 1);
                    while (data.Volume.Count > minLength) data.Volume.RemoveAt(data.Volume.Count - 1);
                    while (data.Timestamp.Count > minLength) data.Timestamp.RemoveAt(data.Timestamp.Count - 1);
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
                    for (int i = 0; i < Math.Min(outputTensor.Length, 4); i++) // Safety: limit to 4 elements max (our action space)
                    {
                        TradingAction actionType = MapIndexToAction(i);
                        _logger($"  - {actionType}: {outputTensor[i]:P2}");
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
                    confidence = outputTensor[actionIndex];
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
                    for (int i = 0; i < Math.Min(outputTensor.Length, 4); i++) // Safety: limit to 4 elements max
                    {
                        TradingAction actionType = MapIndexToAction(i);
                        _logger($"  - {actionType}: {outputTensor[i]:P2}");
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
                    confidence = outputTensor[actionIndex];
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
                
                _logger($"MarketData validation: Close.Count={data.Close.Count}, Open.Count={data.Open.Count}, " +
                        $"High.Count={data.High.Count}, Low.Count={data.Low.Count}, Volume.Count={data.Volume.Count}, " +
                        $"Timestamp.Count={data.Timestamp.Count}");
                
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
                
                // Extract data and create features
                var features = new List<float>();
                int lastIdx = data.Close.Count - 1;
                _logger($"Processing last bar at index {lastIdx}");
                
                try
                {
                    // Convert to arrays for calculation
                    double[] close = data.Close.Select(d => (double)d).ToArray();
                    double[] open = data.Open.Select(d => (double)d).ToArray();
                    double[] high = data.High.Select(d => (double)d).ToArray();
                    double[] low = data.Low.Select(d => (double)d).ToArray();
                    double[] volume = data.Volume.Select(d => (double)d).ToArray();
                    
                    // 1. Calculate returns (same as Python implementation)
                    float returns;
                    if (lastIdx > 0 && !double.IsNaN(close[lastIdx]) && !double.IsNaN(close[lastIdx - 1]) && 
                        close[lastIdx - 1] != 0)
                    {
                        returns = (float)((close[lastIdx] / close[lastIdx - 1]) - 1.0);
                    }
                    else
                    {
                        returns = 0f;
                    }
                    // Clip returns like Python code does
                    returns = Math.Clamp(returns, -0.1f, 0.1f);
                    features.Add(returns);
                    
                    // 2. Calculate RSI (14-period) - simplified implementation
                    int rsiPeriod = 14;
                    float rsi = 50f; // Default to neutral
                    if (lastIdx >= rsiPeriod)
                    {
                        double[] gains = new double[rsiPeriod];
                        double[] losses = new double[rsiPeriod];
                        
                        for (int i = 0; i < rsiPeriod; i++)
                        {
                            double change = close[lastIdx - i] - close[lastIdx - i - 1];
                            if (change > 0)
                                gains[i] = change;
                            else
                                losses[i] = -change;
                        }
                        
                        double avgGain = gains.Average();
                        double avgLoss = losses.Average();
                        
                        if (avgLoss != 0)
                        {
                            double rs = avgGain / avgLoss;
                            rsi = (float)(100.0 - (100.0 / (1.0 + rs)));
                        }
                        else if (avgGain != 0)
                        {
                            rsi = 100f;
                        }
                    }
                    // Normalize RSI to [-1, 1] like Python does
                    features.Add(rsi / 50f - 1f);
                    
                    // 3. Calculate ATR (14-period)
                    int atrPeriod = 14;
                    float atr = 0f;
                    if (lastIdx >= atrPeriod)
                    {
                        double[] trValues = new double[atrPeriod];
                        
                        for (int i = 0; i < atrPeriod; i++)
                        {
                            int idx = lastIdx - i;
                            double tr1 = high[idx] - low[idx];
                            double tr2 = Math.Abs(high[idx] - close[Math.Max(0, idx - 1)]);
                            double tr3 = Math.Abs(low[idx] - close[Math.Max(0, idx - 1)]);
                            trValues[i] = Math.Max(Math.Max(tr1, tr2), tr3);
                        }
                        
                        atr = (float)trValues.Average();
                    }
                    else if (lastIdx >= 1)
                    {
                        // Simplified ATR if we don't have enough data
                        atr = (float)(high[lastIdx] - low[lastIdx]);
                    }
                    
                    // Normalize ATR like Python code - ATR ratio to its SMA
                    int windowSize = Math.Min(20, lastIdx + 1);
                    double atrSma = 0;
                    if (lastIdx >= windowSize)
                    {
                        double[] atrValues = new double[windowSize];
                        for (int i = 0; i < windowSize; i++)
                        {
                            int idx = lastIdx - i;
                            double tr1 = high[idx] - low[idx];
                            double tr2 = idx > 0 ? Math.Abs(high[idx] - close[idx - 1]) : 0;
                            double tr3 = idx > 0 ? Math.Abs(low[idx] - close[idx - 1]) : 0;
                            atrValues[i] = Math.Max(Math.Max(tr1, tr2), tr3);
                        }
                        atrSma = atrValues.Average();
                    }
                    else
                    {
                        atrSma = atr;
                    }
                    
                    float atrRatio = atrSma != 0 ? atr / (float)atrSma : 1f;
                    float minExpectedRatio = 0.5f;
                    float maxExpectedRatio = 2.0f;
                    float expectedRange = maxExpectedRatio - minExpectedRatio;
                    float atrNorm = 2f * (atrRatio - minExpectedRatio) / expectedRange - 1f;
                    atrNorm = Math.Clamp(atrNorm, -1f, 1f);
                    features.Add(atrNorm);
                    
                    // 4. Calculate volume change
                    float volumeChange = 0f;
                    if (lastIdx > 0 && volume[lastIdx - 1] != 0)
                    {
                        volumeChange = (float)((volume[lastIdx] - volume[lastIdx - 1]) / volume[lastIdx - 1]);
                        volumeChange = Math.Clamp(volumeChange, -1f, 1f);
                    }
                    features.Add(volumeChange);
                    
                    // 5. Calculate volatility breakout (Bollinger Band position)
                    float volatilityBreakout = 0.5f; // Default to middle
                    int bollPeriod = 20;
                    if (lastIdx >= bollPeriod)
                    {
                        double[] closePrices = new double[bollPeriod];
                        for (int i = 0; i < bollPeriod; i++)
                        {
                            closePrices[i] = close[lastIdx - i];
                        }
                        
                        double sma = closePrices.Average();
                        double sum = closePrices.Sum(p => Math.Pow(p - sma, 2));
                        double stdDev = Math.Sqrt(sum / bollPeriod);
                        
                        double upper = sma + (2 * stdDev);
                        double lower = sma - (2 * stdDev);
                        double bandRange = upper - lower;
                        
                        if (bandRange > 0)
                        {
                            volatilityBreakout = (float)((close[lastIdx] - lower) / bandRange);
                            volatilityBreakout = Math.Clamp(volatilityBreakout, 0f, 1f);
                        }
                    }
                    features.Add(volatilityBreakout);
                    
                    // 6. Calculate trend strength (ADX-like)
                    float trendStrength = 0f; // Default to no trend
                    int adxPeriod = 14;
                    if (lastIdx >= adxPeriod + 1)
                    {
                        // Simplified ADX calculation - we use directional movement instead
                        double posDMSum = 0;
                        double negDMSum = 0;
                        double trSum = 0;
                        
                        for (int i = 1; i <= adxPeriod; i++)
                        {
                            int idx = lastIdx - adxPeriod + i;
                            if (idx > 0)
                            {
                                double posDM = high[idx] - high[idx - 1];
                                double negDM = low[idx - 1] - low[idx];
                                
                                if (posDM > 0 && posDM > negDM)
                                    posDMSum += posDM;
                                
                                if (negDM > 0 && negDM > posDM)
                                    negDMSum += negDM;
                                
                                double tr1 = high[idx] - low[idx];
                                double tr2 = Math.Abs(high[idx] - close[idx - 1]);
                                double tr3 = Math.Abs(low[idx] - close[idx - 1]);
                                trSum += Math.Max(Math.Max(tr1, tr2), tr3);
                            }
                        }
                        
                        if (trSum > 0)
                        {
                            double posDI = (posDMSum / trSum) * 100;
                            double negDI = (negDMSum / trSum) * 100;
                            double dx = Math.Abs(posDI - negDI) / (posDI + negDI) * 100;
                            
                            // Normalize like Python code (ADX/25 - 1)
                            trendStrength = (float)(dx / 25.0 - 1.0);
                            trendStrength = Math.Clamp(trendStrength, -1f, 1f);
                        }
                    }
                    features.Add(trendStrength);
                    
                    // 7. Calculate candle pattern
                    float body = (float)(close[lastIdx] - open[lastIdx]);
                    float upperWick = (float)(high[lastIdx] - Math.Max(close[lastIdx], open[lastIdx]));
                    float lowerWick = (float)(Math.Min(close[lastIdx], open[lastIdx]) - low[lastIdx]);
                    float range = (float)(high[lastIdx] - low[lastIdx] + 1e-8);
                    
                    float candlePattern;
                    if (upperWick + lowerWick > 0)
                    {
                        candlePattern = (body / range + (upperWick - lowerWick) / (upperWick + lowerWick)) / 2f;
                    }
                    else
                    {
                        candlePattern = body / range / 2f;
                    }
                    candlePattern = Math.Clamp(candlePattern, -1f, 1f);
                    features.Add(candlePattern);
                    
                    // 8-9. Time features (same as before)
                    DateTimeOffset barTimeOffset;
                    try
                    {
                        barTimeOffset = DateTimeOffset.FromUnixTimeSeconds(data.Timestamp[lastIdx]);
                    }
                    catch
                    {
                        barTimeOffset = DateTimeOffset.UtcNow;
                    }
                    var barTime = barTimeOffset.DateTime;
                    
                    double minutesInDay = 24.0 * 60.0;
                    double timeIndex = barTime.Hour * 60.0 + barTime.Minute;
                    double timeNormalized = 2.0 * Math.PI * timeIndex / minutesInDay;
                    
                    features.Add((float)Math.Sin(timeNormalized));
                    features.Add((float)Math.Cos(timeNormalized));
                    
                    // 10-11. Position direction and PnL (unchanged)
                    features.Add((float)data.PositionDirection);
                    features.Add(float.IsNaN((float)data.PositionPnl) ? 0f : (float)data.PositionPnl);
                    
                    _logger($"Preprocessed {features.Count} features for single timestep");
                    
                    // Final validation
                    for (int i = 0; i < features.Count; i++)
                    {
                        if (float.IsNaN(features[i]) || float.IsInfinity(features[i]))
                        {
                            _logger($"WARNING: Invalid value in feature {i}, replacing with 0");
                            features[i] = 0f;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger($"ERROR during feature calculation: {ex.Message}");
                    throw;
                }
                
                return features.ToArray();
            }
            catch (Exception ex)
            {
                _logger($"CRITICAL ERROR in PreprocessFeatures: {ex.Message}");
                _logger($"Stack trace: {ex.StackTrace}");
                
                // Return a safe default feature vector instead of throwing
                _logger("Returning fallback feature vector to avoid crashing");
                return new float[11] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
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
                            if (bandRange > 0)
                            {
                                volatilityBreakout = (float)((close[i] - lowerBand[i]) / bandRange);
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
                        if (upperWick + lowerWick > 0)
                        {
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
                        
                        features.Add((float)Math.Sin(timeNormalized)); // Sin time
                        features.Add((float)Math.Cos(timeNormalized)); // Cos time
                        
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
                
                _logger($"Finding max probability among {tensor.Length} values");
                
                int maxIndex = 0;
                float maxValue = tensor[0];
                
                for (int i = 1; i < tensor.Length; i++)
                {
                    if (tensor[i] > maxValue)
                    {
                        maxValue = tensor[i];
                        maxIndex = i;
                    }
                }
                
                _logger($"Max probability found: index={maxIndex}, value={maxValue}");
                return maxIndex;
            }
            catch (Exception ex)
            {
                _logger($"ERROR in GetMaxProbabilityIndex: {ex.Message}");
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