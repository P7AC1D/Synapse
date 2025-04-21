//+------------------------------------------------------------------+
//|                                              RecurrentPPOModel.mqh  |
//|                                   Copyright 2024, DRL Trading Bot   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"

//+------------------------------------------------------------------+
//| Model settings struct                                             |
//+------------------------------------------------------------------+
struct ModelSettings {
    int sequenceLength;     // Number of bars to feed into the model
    int numFeatures;        // Number of features per bar (e.g., OHLCV)
    int lstmLayers;         // Number of LSTM layers
    int lstmHiddenSize;     // Size of LSTM hidden state
    int numActions;         // Number of possible actions (usually 4: hold, buy, sell, close)
};

//+------------------------------------------------------------------+
//| RecurrentPPO model wrapper class for MQL5                        |
//+------------------------------------------------------------------+
class RecurrentPPOModel : public CObject {
private:
    long m_modelHandle;        // Handle to the ONNX model
    string m_modelPath;
    bool m_initialized;
    string m_lastError;
    
    // Model settings
    ModelSettings m_settings;
    
    // LSTM state
    matrixf m_lstmHidden;      // Using matrixf for LSTM hidden state
    matrixf m_lstmCell;        // Using matrixf for LSTM cell state
    bool m_hasState;

public:
    RecurrentPPOModel();
    ~RecurrentPPOModel();
    
    // Initialization
    bool Initialize(const string modelPath, const ModelSettings &settings);
    void Cleanup();
    void ResetLSTMState();
    
    // Prediction
    bool Predict(
        float &observationData[],
        int &action,
        string &description,
        double &actionProbabilities[]
    );
    
    // Getters
    string LastError() const { return m_lastError; }
    bool IsInitialized() const { return m_initialized; }
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
RecurrentPPOModel::RecurrentPPOModel() {
    m_initialized = false;
    m_modelHandle = INVALID_HANDLE;
    m_modelPath = "";
    m_lastError = "";
    m_hasState = false;
    
    // Set default model settings
    m_settings.sequenceLength = 500;
    m_settings.numFeatures = 5; // OHLCV
    m_settings.lstmLayers = 1;
    m_settings.lstmHiddenSize = 64;
    m_settings.numActions = 4;
}

//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
RecurrentPPOModel::~RecurrentPPOModel() {
    Cleanup();
}

//+------------------------------------------------------------------+
//| Initialize the model                                             |
//+------------------------------------------------------------------+
bool RecurrentPPOModel::Initialize(const string modelPath, const ModelSettings &settings) {
    if(m_initialized) Cleanup();
    
    m_modelPath = modelPath;
    m_settings = settings;
    
    // Check if file exists first
    if(!FileIsExist(modelPath, FILE_COMMON)) {
        m_lastError = StringFormat("Model file not found: %s - Please check if the file exists in the MQL5 Data Folder", modelPath);
        Print(m_lastError);
        return false;
    }
    
    Print("Trying to load ONNX model from: ", modelPath);
    
    // Create the ONNX model using the official MQL5 function
    m_modelHandle = OnnxCreate(modelPath, ONNX_DEBUG_LOGS | ONNX_COMMON_FOLDER);
    
    if(m_modelHandle == INVALID_HANDLE) {
        m_lastError = "Failed to initialize ONNX model: " + (string)GetLastError();
        Print(m_lastError);
        
        // Add more diagnostic information
        Print("ONNX model initialization failed. This could be due to:");
        Print("1. Invalid ONNX model format");
        Print("2. Incompatible ONNX Runtime version");
        Print("3. Incorrect model path: ", modelPath);
        
        // Suggest potential solutions
        Print("Possible solutions:");
        Print("- Make sure the model is in a valid ONNX format");
        Print("- Place the model file in the Terminal/MQL5/Files/ directory");
        Print("- Check the MetaTrader 5 log for detailed error messages");
        
        return false;
    }
    
    // Define input shape - shape is [batch_size, seq_len, features]
    long inputShape[] = {1, m_settings.sequenceLength, m_settings.numFeatures};
    if(!OnnxSetInputShape(m_modelHandle, 0, inputShape)) {
        m_lastError = "Failed to set input shape: " + (string)GetLastError();
        Print(m_lastError);
        Cleanup();
        return false;
    }
    
    // Define output shape for actions - shape is [batch_size, num_actions]
    long outputShape[] = {1, m_settings.numActions};
    if(!OnnxSetOutputShape(m_modelHandle, 0, outputShape)) {
        m_lastError = "Failed to set output shape: " + (string)GetLastError();
        Print(m_lastError);
        Cleanup();
        return false;
    }
    
    // If using LSTM, your model will need additional input/output shapes for the LSTM states
    // For LSTM hidden state - shape is [num_layers, batch_size, hidden_size]
    long lstmHiddenShape[] = {m_settings.lstmLayers, 1, m_settings.lstmHiddenSize};
    
    // Set additional input shapes for LSTM state if available in your model
    // Input index 1 for hidden state
    if(!OnnxSetInputShape(m_modelHandle, 1, lstmHiddenShape)) {
        m_lastError = "Failed to set LSTM hidden state input shape: " + (string)GetLastError();
        Print(m_lastError);
        // This might not be a critical error if your model doesn't use LSTM state as input
    }
    
    // Input index 2 for cell state
    if(!OnnxSetInputShape(m_modelHandle, 2, lstmHiddenShape)) {
        m_lastError = "Failed to set LSTM cell state input shape: " + (string)GetLastError();
        Print(m_lastError);
        // This might not be a critical error if your model doesn't use LSTM state as input
    }
    
    // Set additional output shapes for the new LSTM state if available in your model
    // Output index 1 for new hidden state
    if(!OnnxSetOutputShape(m_modelHandle, 1, lstmHiddenShape)) {
        m_lastError = "Failed to set LSTM hidden state output shape: " + (string)GetLastError();
        Print(m_lastError);
        // This might not be a critical error if your model doesn't output LSTM state
    }
    
    // Output index 2 for new cell state
    if(!OnnxSetOutputShape(m_modelHandle, 2, lstmHiddenShape)) {
        m_lastError = "Failed to set LSTM cell state output shape: " + (string)GetLastError();
        Print(m_lastError);
        // This might not be a critical error if your model doesn't output LSTM state
    }
    
    // Initialize LSTM states
    ResetLSTMState();
    
    m_initialized = true;
    Print("ONNX model initialized successfully with handle: ", m_modelHandle);
    return true;
}

//+------------------------------------------------------------------+
//| Clean up resources                                               |
//+------------------------------------------------------------------+
void RecurrentPPOModel::Cleanup() {
    if(m_modelHandle != INVALID_HANDLE) {
        OnnxRelease(m_modelHandle);
        m_modelHandle = INVALID_HANDLE;
    }
    m_initialized = false;
    ResetLSTMState();
}

//+------------------------------------------------------------------+
//| Reset LSTM state                                                 |
//+------------------------------------------------------------------+
void RecurrentPPOModel::ResetLSTMState() {
    // Create matrices for the LSTM state
    m_lstmHidden.Resize(m_settings.lstmLayers, m_settings.lstmHiddenSize);
    m_lstmCell.Resize(m_settings.lstmLayers, m_settings.lstmHiddenSize);
    
    // Initialize to zeros
    m_lstmHidden.Fill(0.0);
    m_lstmCell.Fill(0.0);
    
    m_hasState = false;
}

//+------------------------------------------------------------------+
//| Run prediction on the model                                      |
//+------------------------------------------------------------------+
bool RecurrentPPOModel::Predict(
    float &observationData[],
    int &action,
    string &description,
    double &actionProbabilities[]
) {
    if(!m_initialized) {
        m_lastError = "Model not initialized";
        return false;
    }
    
    // Check array dimensions
    int obsSize = ArraySize(observationData);
    int expectedSize = m_settings.sequenceLength * m_settings.numFeatures;
    if(obsSize != expectedSize) {
        m_lastError = StringFormat(
            "Invalid observation size: got %d, expected %d",
            obsSize, expectedSize);
        return false;
    }
    
    // Create input matrix from observation data
    matrixf inputMatrix;
    inputMatrix.Resize(1, m_settings.sequenceLength * m_settings.numFeatures); // Flatten input for now
    
    // Copy data to the input matrix
    for(int i = 0; i < obsSize; i++) {
        inputMatrix[0][i] = observationData[i];
    }
    
    // Reshape the matrix to the expected dimensions [1, seq_len, features]
    // Fix: Reshape only takes 2 parameters (rows, cols), not 3
    // We need to keep it as a 2D matrix with rows=1, cols=seq_len*features
    // The ONNX runtime will interpret the shape correctly based on the OnnxSetInputShape we did earlier
    
    // Create output matrices
    matrixf outputProbs;      // For action probabilities
    matrixf newLstmHidden;    // For the new LSTM hidden state
    matrixf newLstmCell;      // For the new LSTM cell state
    
    // Create input array for OnnxRun - we need to pass the model all required inputs
    matrixf inputs[];
    ArrayResize(inputs, 3); // Observation + hidden state + cell state
    
    // Set the input matrices
    inputs[0] = inputMatrix;  // Observation data
    inputs[1] = m_lstmHidden; // LSTM hidden state
    inputs[2] = m_lstmCell;   // LSTM cell state
    
    // Create output array for OnnxRun
    matrixf outputs[];
    ArrayResize(outputs, 3); // Probabilities + new hidden state + new cell state
    
    // Run inference using official OnnxRun function
    // Fix: Replace ONNX_VERBOSE_LOGS with ONNX_DEBUG_LOGS
    if(!OnnxRun(m_modelHandle, ONNX_DEBUG_LOGS, inputs, outputs)) {
        m_lastError = "Inference failed: " + (string)GetLastError();
        return false;
    }
    
    // Get action probabilities from outputs[0]
    outputProbs = outputs[0];
    
    // Update LSTM states with new states from model output
    if(ArraySize(outputs) > 1) {
        m_lstmHidden = outputs[1];
        m_lstmCell = outputs[2];
        m_hasState = true;
    }
    
    // Resize and fill action probabilities
    ArrayResize(actionProbabilities, m_settings.numActions);
    
    // Find the action with highest probability
    float maxProb = -1;
    action = 0;
    
    for(int i = 0; i < m_settings.numActions; i++) {
        // Get the probability from the output matrix (shape is [1, num_actions])
        float prob = outputProbs[0][i];
        actionProbabilities[i] = prob;
        
        if(prob > maxProb) {
            maxProb = prob;
            action = i;
        }
    }
    
    // Create a description based on the action
    switch(action) {
        case 0: // Hold
            description = "Hold position";
            break;
        case 1: // Buy
            description = "Open long position";
            break;
        case 2: // Sell
            description = "Open short position";
            break;
        case 3: // Close
            description = "Close current position";
            break;
        default:
            description = "Unknown action";
            break;
    }
    
    return true;
}