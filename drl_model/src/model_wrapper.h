#pragma once
#include <torch/script.h>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

class ModelWrapper {
private:
    torch::jit::script::Module module;
    torch::Device device;
    int feature_count;
    int hidden_size;
    int action_count;
    
    // Feature processing parameters
    std::vector<std::string> feature_names;
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    
public:
    ModelWrapper(const std::string& model_path, const std::string& config_path);
    ~ModelWrapper() = default;
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    predict(const torch::Tensor& features,
            const torch::Tensor& hidden_state,
            const torch::Tensor& cell_state);
            
    // Process and normalize features
    std::vector<float> process_features(const std::vector<double>& raw_features);
    
    // Get action from probabilities
    int get_action(const torch::Tensor& action_probs);
    
    // Getters for model properties
    int get_feature_count() const { return feature_count; }
    int get_hidden_size() const { return hidden_size; }
    int get_action_count() const { return action_count; }
    std::vector<std::string> get_feature_names() const { return feature_names; }
    
private:
    void load_config(const std::string& config_path);
    torch::Tensor normalize_features(const std::vector<float>& features);
};
