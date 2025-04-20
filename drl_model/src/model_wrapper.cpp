#include "model_wrapper.h"
#include <fstream>
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
    std::vector<float> normalized;
    normalized.reserve(features.size());
    
    for (size_t i = 0; i < features.size(); ++i) {
        normalized.push_back((features[i] - feature_means[i]) / feature_stds[i]);
    }
    
    return torch::from_blob(normalized.data(), 
                          {1, 1, static_cast<int64_t>(normalized.size())},
                          torch::kFloat).clone().to(device);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ModelWrapper::predict(const torch::Tensor& features,
                     const torch::Tensor& hidden_state,
                     const torch::Tensor& cell_state) {
    torch::NoGradGuard no_grad;
    
    try {
        // Normalize features
        auto normalized_features = normalize_features(
            std::vector<float>(features.data_ptr<float>(),
                             features.data_ptr<float>() + feature_count)
        );
        
        // Prepare inputs for the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(normalized_features);
        inputs.push_back(hidden_state);
        inputs.push_back(cell_state);
        
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
