#include "model_wrapper.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

ModelWrapper::ModelWrapper(const std::string& model_path, const std::string& config_path) 
    : device(torch::kCPU) {
    try {
        // Load TorchScript model
        module = torch::jit::load(model_path);
        module.to(device);
        module.eval();
        
        // Load configuration
        load_config(config_path);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize model: " + std::string(e.what()));
    }
}

void ModelWrapper::load_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }
    
    json config;
    f >> config;
    
    // Load model parameters
    feature_count = config["feature_count"];
    hidden_size = config["hidden_size"];
    action_count = config["action_count"];
    
    // Load feature names
    feature_names = config["feature_names"].get<std::vector<std::string>>();
    
    // Load feature normalization parameters
    feature_means = config["feature_means"].get<std::vector<float>>();
    feature_stds = config["feature_stds"].get<std::vector<float>>();
    
    // Validate sizes
    if (feature_names.size() != feature_count ||
        feature_means.size() != feature_count ||
        feature_stds.size() != feature_count) {
        throw std::runtime_error("Inconsistent feature dimensions in config");
    }
}

std::vector<float> ModelWrapper::process_features(const std::vector<double>& raw_features) {
    if (raw_features.size() != feature_count) {
        throw std::runtime_error("Invalid feature count: " + 
                               std::to_string(raw_features.size()) + 
                               " (expected " + std::to_string(feature_count) + ")");
    }
    
    // Convert double to float
    std::vector<float> features;
    features.reserve(feature_count);
    
    for (size_t i = 0; i < raw_features.size(); ++i) {
        features.push_back(static_cast<float>(raw_features[i]));
    }
    
    return features;
}

torch::Tensor ModelWrapper::normalize_features(const std::vector<float>& features) {
    // Features are already normalized in the Python environment (TradingEnv)
    // Just convert to tensor with proper shape for the model
    return torch::from_blob(const_cast<float*>(features.data()), 
                           {static_cast<int64_t>(features.size())}, 
                           torch::kFloat).clone();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ModelWrapper::predict(const torch::Tensor& features,
                     const torch::Tensor& hidden_state,
                     const torch::Tensor& cell_state) {
    torch::NoGradGuard no_grad;
    
    try {
        // Features are already normalized in the TradingEnv, so we can use them directly
        // Just ensure proper shape for the model input: [1, 1, feature_count]
        auto features_reshaped = features.view({1, 1, -1});
        
        // Ensure hidden states have the correct dtype for maximum precision
        auto hidden_state_f32 = hidden_state.to(torch::kFloat32);
        auto cell_state_f32 = cell_state.to(torch::kFloat32);
        
        // Prepare inputs for the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features_reshaped);
        inputs.push_back(hidden_state_f32);
        inputs.push_back(cell_state_f32);
        
        // Forward pass
        auto output = module.forward(inputs).toTuple();
        
        // Extract outputs
        auto action_probs = output->elements()[0].toTensor();
        auto new_hidden = output->elements()[1].toTensor();
        auto new_cell = output->elements()[2].toTensor();
        
        return std::make_tuple(action_probs, new_hidden, new_cell);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Prediction failed: " + std::string(e.what()));
    }
}

torch::Tensor ModelWrapper::get_initial_hidden_state() {
    torch::NoGradGuard no_grad;
    
    try {
        // Try to call the model's get_initial_state method
        std::vector<torch::jit::IValue> inputs = {1}; // batch_size = 1
        auto states = module.get_method("get_initial_state")(inputs).toTuple();
        return states->elements()[0].toTensor();
    } catch (const std::exception& e) {
        // Fallback: create zeros tensor with expected dimensions
        // Assumed dimensions: [num_layers(2), batch_size(1), hidden_size(256)]
        return torch::zeros({2, 1, 256}, torch::kFloat32);
    }
}

torch::Tensor ModelWrapper::get_initial_cell_state() {
    torch::NoGradGuard no_grad;
    
    try {
        // Try to call the model's get_initial_state method
        std::vector<torch::jit::IValue> inputs = {1}; // batch_size = 1
        auto states = module.get_method("get_initial_state")(inputs).toTuple();
        return states->elements()[1].toTensor();
    } catch (const std::exception& e) {
        // Fallback: create zeros tensor with expected dimensions
        // Assumed dimensions: [num_layers(2), batch_size(1), hidden_size(256)]
        return torch::zeros({2, 1, 256}, torch::kFloat32);
    }
}

int ModelWrapper::get_action(const torch::Tensor& action_probs) {
    // Get the index of the maximum probability
    auto max_idx = action_probs.argmax(-1).item<int>();
    return max_idx;
}
