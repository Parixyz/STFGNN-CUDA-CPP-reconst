#include "level1_layers.h"

Level1Layers::Level1Layers(int input_dim, int output_dim)
    : stfgnn_layer(input_dim, output_dim), gated_conv(input_dim, output_dim) {}

std::vector<float> Level1Layers::forward(const std::vector<float>& input) {
    auto spatial_output = stfgnn_layer.forward(input);
    return gated_conv.forward(spatial_output);
}
