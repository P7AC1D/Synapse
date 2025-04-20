//+------------------------------------------------------------------+
//|                                                RecurrentPPOModel.mqh |
//|                                   Copyright 2024, DRL Trading Bot |
//|                                     https://github.com/your-repo |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"

#include <DRL/ONNXRuntime.mqh>

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
    ONNXRuntime m_runtime;
    string m_modelPath;
    bool m_initialized;
    string m_lastError;
    
    // Model settings
    ModelSettings m_settings;
    
    // LSTM state
    float m_lstmHidden[];
    float m_lstmCell[];
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
    
    // Initialize ONNXRuntime
    if(!m_runtime.Initialize(modelPath)) {
        m_lastError = "Failed to initialize ONNXRuntime: " + m_runtime.LastError();
        Print(m_lastError);
        
        // Add more diagnostic information
        Print("ONNX Runtime initialization failed. This could be due to:");
        Print("1. Missing or incompatible ONNX Runtime DLL");
        Print("2. API version mismatch - your MT5 is using a newer ONNX Runtime than expected");
        Print("3. Incorrect model path: ", modelPath);
        
        // Suggest potential solutions
        Print("Possible solutions:");
        Print("- Place the correct version of onnxruntime.dll in your MT5 terminal 'terminal_dir/MQL5/Libraries/' folder");
        Print("- Use the CPU provider version of ONNX Runtime");
        Print("- Verify that your model is compatible with the ONNX Runtime version you're using");
        
        return false;
    }
    
    // Initialize LSTM states
    ResetLSTMState();
    
    m_initialized = true;
    return true;
}

//+------------------------------------------------------------------+
//| Clean up resources                                               |
//+------------------------------------------------------------------+
void RecurrentPPOModel::Cleanup() {
    m_runtime.Cleanup();
    m_initialized = false;
    ResetLSTMState();
}

//+------------------------------------------------------------------+
//| Reset LSTM state                                                 |
//+------------------------------------------------------------------+
void RecurrentPPOModel::ResetLSTMState() {
    // Resize and initialize LSTM hidden and cell states to zeros
    ArrayResize(m_lstmHidden, 
                m_settings.lstmLayers * 1 * m_settings.lstmHiddenSize);
    ArrayResize(m_lstmCell, 
                m_settings.lstmLayers * 1 * m_settings.lstmHiddenSize);
                
    ArrayInitialize(m_lstmHidden, 0.0);
    ArrayInitialize(m_lstmCell, 0.0);
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
    
    // Prepare dimensions of input/output tensors
    int inputDims[3] = {1, m_settings.sequenceLength, m_settings.numFeatures};
    int lstmDims[4] = {1, m_settings.lstmLayers, 1, m_settings.lstmHiddenSize};
    int outputDims[2] = {1, m_settings.numActions};
    
    // Create output arrays
    float outputProbs[];
    float newLstmH[];
    float newLstmC[];
    
    ArrayResize(outputProbs, m_settings.numActions);
    ArrayResize(newLstmH, m_settings.lstmLayers * 1 * m_settings.lstmHiddenSize);
    ArrayResize(newLstmC, m_settings.lstmLayers * 1 * m_settings.lstmHiddenSize);
    
    // Run inference with LSTM state
    if(!m_runtime.RunInferenceWithLSTM(
        observationData, m_lstmHidden, m_lstmCell,
        outputProbs, newLstmH, newLstmC,
        inputDims, lstmDims, outputDims
    )) {
        m_lastError = "Inference failed: " + m_runtime.LastError();
        return false;
    }
    
    // Update LSTM states
    ArrayCopy(m_lstmHidden, newLstmH);
    ArrayCopy(m_lstmCell, newLstmC);
    m_hasState = true;
    
    // Find the action with highest probability
    float maxProb = -1;
    action = 0;
    
    // Resize and fill action probabilities
    ArrayResize(actionProbabilities, m_settings.numActions);
    
    for(int i = 0; i < m_settings.numActions; i++) {
        actionProbabilities[i] = outputProbs[i];
        
        if(outputProbs[i] > maxProb) {
            maxProb = outputProbs[i];
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