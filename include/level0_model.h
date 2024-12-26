#ifndef LEVEL0_MODEL_H
#define LEVEL0_MODEL_H

#include <vector>
#include "fusion_module.h"
#include "level1_layers.h"

class Level0Model {
public:
    Level0Model(int input_dim, int output_dim);
    void train(const std::vector<std::vector<float>>& data, int epochs, float learning_rate);
    std::vector<float> predict(const std::vector<float>& input);

private:
    FusionModule fusion_module;
    Level1Layers level1_layers;
    int input_dim;
    int output_dim;
};

#endif // LEVEL0_MODEL_H
