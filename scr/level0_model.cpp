#include "level0_model.h"
#include <iostream>

Level0Model::Level0Model(int input_dim, int output_dim)
    : input_dim(input_dim), output_dim(output_dim), 
      fusion_module(input_dim), level1_layers(input_dim, output_dim) {}

void Level0Model::train(const std::vector<std::vector<float>>& data, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        for (const auto& sample : data) {
            auto spatial_features = sample; // Placeholder for spatial input
            auto temporal_features = sample; // Placeholder for temporal input
            
            auto fused_features = fusion_module.fuse(spatial_features, temporal_features);
            auto predictions = level1_layers.forward(fused_features);

            // Example loss calculation (mean squared error)
            for (size_t i = 0; i < predictions.size(); ++i) {
                float target = 1.0f; // Placeholder target
                float error = predictions[i] - target;
                epoch_loss += error * error;
            }
        }
        std::cout << "Epoch " << epoch + 1 << " Loss: " << epoch_loss / data.size() << std::endl;
    }
}

std::vector<float> Level0Model::predict(const std::vector<float>& input) {
    auto fused_features = fusion_module.fuse(input, input); // Placeholder fusion
    return level1_layers.forward(fused_features);
}
