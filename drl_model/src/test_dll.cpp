#include "common.h"
#include "model_wrapper.h"

using namespace drl::utils;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.pt> <model_config.json>" << std::endl;
        return 1;
    }
    
    try {
        // Check files exist
        fs::path model_path = argv[1];
        fs::path config_path = argv[2];
        
        check_file_exists(model_path);
        check_file_exists(config_path);
        
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Config path: " << config_path << std::endl;
        
        // Create model wrapper
        std::cout << "\nLoading model..." << std::endl;
        ModelWrapper model(model_path.string(), config_path.string());
        
        // Get model properties
        std::cout << "\nModel properties:" << std::endl;
        std::cout << "Feature count: " << model.get_feature_count() << std::endl;
        std::cout << "Hidden size: " << model.get_hidden_size() << std::endl;
        std::cout << "Action count: " << model.get_action_count() << std::endl;
        
        // Get feature names
        std::cout << "\nFeature names:" << std::endl;
        const auto& names = model.get_feature_names();
        for (size_t i = 0; i < names.size(); ++i) {
            std::cout << i << ": " << names[i] << std::endl;
        }
        
        // Create dummy features
        std::vector<double> features(model.get_feature_count(), 0.0);
        for (size_t i = 0; i < features.size(); ++i) {
            features[i] = (i % 2 == 0) ? 0.5 : -0.5;  // Alternating values
        }
        
        // Process features
        std::cout << "\nProcessing features..." << std::endl;
        auto processed = model.process_features(features);
        print_array("Raw features", features);
        print_array("Processed", processed);
        
        // Create tensors for prediction - don't normalize here
        // as normalization will happen inside predict()
        auto features_tensor = torch::from_blob(
            processed.data(),
            {1, 1, static_cast<int64_t>(processed.size())},
            torch::kFloat
        ).clone();
        
        print_tensor_shape("Features", features_tensor);
        
        auto hidden_state = torch::zeros({1, 1, model.get_hidden_size()});
        auto cell_state = torch::zeros({1, 1, model.get_hidden_size()});
        
        print_tensor_shape("Hidden state", hidden_state);
        print_tensor_shape("Cell state", cell_state);
        
        // Run prediction
        std::cout << "\nRunning prediction..." << std::endl;
        auto [probs, new_hidden, new_cell] = model.predict(
            features_tensor,
            hidden_state,
            cell_state
        );
        
        // Print probabilities
        std::cout << "Action probabilities:" << std::endl;
        auto probs_acc = probs.accessor<float, 2>();
        for (int i = 0; i < model.get_action_count(); ++i) {
            std::cout << "Action " << i << ": " << std::fixed 
                     << std::setprecision(6) << probs_acc[0][i] << std::endl;
        }
        
        // Test multiple steps
        std::cout << "\nTesting multiple prediction steps..." << std::endl;
        hidden_state = new_hidden;
        cell_state = new_cell;
        
        for (int step = 0; step < 3; ++step) {
            std::cout << "\nStep " << step + 2 << ":" << std::endl;
            
            auto [step_probs, step_hidden, step_cell] = model.predict(
                features_tensor,
                hidden_state,
                cell_state
            );
            
            auto step_acc = step_probs.accessor<float, 2>();
            for (int i = 0; i < model.get_action_count(); ++i) {
                std::cout << "Action " << i << ": " << std::fixed 
                         << std::setprecision(6) << step_acc[0][i] << std::endl;
            }
            
            hidden_state = step_hidden;
            cell_state = step_cell;
        }
        
        std::cout << "\nTest version " << DRL_MODEL_VERSION 
                 << " (" << DRL_MODEL_BUILD_TYPE << ") "
                 << "completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
