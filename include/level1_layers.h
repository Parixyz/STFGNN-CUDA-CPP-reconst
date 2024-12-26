#ifndef LEVEL1_LAYERS_H
#define LEVEL1_LAYERS_H

#include <vector>
#include "stfgnn_layer.h"
#include "gated_conv.h"

class Level1Layers {
public:
    Level1Layers(int input_dim, int output_dim);
    std::vector<float> forward(const std::vector<float>& input);

private:
    STFGNNLayer stfgnn_layer;
    GatedConv gated_conv;
};

#endif // LEVEL1_LAYERS_H
