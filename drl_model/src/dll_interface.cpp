#include "model_wrapper.h"
#include <memory>
#include <stdexcept>
#include <iostream>

namespace {
    // Helper function to log errors
    void log_error(const std::string& message) {
        std::cerr << "DLL Error: " << message << std::endl;
    }
}

extern "C" {

__declspec(dllexport) void* CreateModel(const char* model_path, const char* config_path) {
    if (!model_path || !config_path) {
        log_error("Invalid model or config path");
        return nullptr;
    }

    try {
        auto* model = new ModelWrapper(model_path, config_path);
        return static_cast<void*>(model);
    }
    catch (const std::exception& e) {
        log_error(std::string("Model creation failed: ") + e.what());
        return nullptr;
    }
}

__declspec(dllexport) void DestroyModel(void* handle) {
    if (handle) {
        try {
            auto* model = static_cast<ModelWrapper*>(handle);
            delete model;
        }
        catch (const std::exception& e) {
            log_error(std::string("Model destruction failed: ") + e.what());
        }
    }
}

__declspec(dllexport) bool Predict(
    void* handle,
    const double* features,
    int feature_count,
    double* lstm_state,
    int state_size,
    double* output_probs
) {
    if (!handle || !features || !lstm_state || !output_probs) {
        log_error("Invalid parameters in Predict");
        return false;
    }
    
    try {
        auto* model = static_cast<ModelWrapper*>(handle);
        
        // Validate input dimensions
        if (feature_count != model->get_feature_count() || 
            state_size != 2 * model->get_hidden_size()) {
            log_error("Invalid dimensions in Predict");
            return false;
        }
        
        // Convert features to vector
        std::vector<double> feature_vec(features, features + feature_count);
        
        // Process features
        auto processed_features = model->process_features(feature_vec);
        
        // Split LSTM state into hidden and cell states
        int hidden_size = model->get_hidden_size();
        
        // Create tensors from the LSTM state arrays
        auto hidden_state = torch::from_blob(
            lstm_state, 
            {1, 1, hidden_size}, 
            torch::kFloat
        ).clone();
        
        auto cell_state = torch::from_blob(
            lstm_state + hidden_size,
            {1, 1, hidden_size},
            torch::kFloat
        ).clone();
        
        // Convert processed features to tensor
        auto features_tensor = torch::from_blob(
            processed_features.data(),
            {1, 1, static_cast<int64_t>(processed_features.size())},
            torch::kFloat
        ).clone();
        
        // Run prediction
        auto [probs, new_hidden, new_cell] = model->predict(
            features_tensor,
            hidden_state,
            cell_state
        );
        
        // Copy probabilities to output array
        auto probs_acc = probs.accessor<float,2>();
        for (int i = 0; i < model->get_action_count(); ++i) {
            output_probs[i] = static_cast<double>(probs_acc[0][i]);
        }
        
        // Update LSTM state
        auto hidden_acc = new_hidden.accessor<float,3>();
        auto cell_acc = new_cell.accessor<float,3>();
        
        for (int i = 0; i < hidden_size; ++i) {
            lstm_state[i] = static_cast<double>(hidden_acc[0][0][i]);
            lstm_state[i + hidden_size] = static_cast<double>(cell_acc[0][0][i]);
        }
        
        return true;
    }
    catch (const std::exception& e) {
        log_error(std::string("Prediction failed: ") + e.what());
        return false;
    }
}

__declspec(dllexport) bool GetModelProperties(
    void* handle,
    int* feature_count,
    int* hidden_size,
    int* action_count
) {
    if (!handle || !feature_count || !hidden_size || !action_count) {
        log_error("Invalid parameters in GetModelProperties");
        return false;
    }
    
    try {
        auto* model = static_cast<ModelWrapper*>(handle);
        *feature_count = model->get_feature_count();
        *hidden_size = model->get_hidden_size();
        *action_count = model->get_action_count();
        return true;
    }
    catch (const std::exception& e) {
        log_error(std::string("Failed to get model properties: ") + e.what());
        return false;
    }
}

// Global storage for feature names to ensure proper memory management
struct FeatureNamesData {
    char** names;
    int count;
};

__declspec(dllexport) const char* const* GetFeatureNames(
    void* handle,
    int* count
) {
    if (!handle || !count) {
        log_error("Invalid parameters in GetFeatureNames");
        return nullptr;
    }
    
    try {
        auto* model = static_cast<ModelWrapper*>(handle);
        const auto& names = model->get_feature_names();
        *count = static_cast<int>(names.size());
        
        // Create a new array of C strings (we need to allocate both the array and the strings)
        auto* data = new FeatureNamesData();
        data->count = *count;
        data->names = new char*[names.size()];
        
        for (size_t i = 0; i < names.size(); ++i) {
            // Allocate memory for each string copy and copy the string content
            data->names[i] = new char[names[i].length() + 1];
            strcpy(data->names[i], names[i].c_str());
        }
        
        return reinterpret_cast<const char* const*>(data);
    }
    catch (const std::exception& e) {
        log_error(std::string("Failed to get feature names: ") + e.what());
        *count = 0;
        return nullptr;
    }
}

__declspec(dllexport) void FreeFeatureNames(const char* const* names_ptr) {
    if (names_ptr) {
        auto* data = reinterpret_cast<FeatureNamesData*>(const_cast<char**>(names_ptr));
        
        // Free each string
        for (int i = 0; i < data->count; ++i) {
            delete[] data->names[i];
        }
        
        // Free the array itself
        delete[] data->names;
        
        // Free the container
        delete data;
    }
}

} // extern "C"
