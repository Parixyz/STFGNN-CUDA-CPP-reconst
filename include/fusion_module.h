#ifndef FUSION_MODULE_H
#define FUSION_MODULE_H

#include <vector>

class FusionModule {
public:
    FusionModule(int input_dim);
    std::vector<float> fuse(const std::vector<float>& spatial, const std::vector<float>& temporal);

private:
    int input_dim;
    std::vector<float> fusion_weights;
};

#endif // FUSION_MODULE_H
