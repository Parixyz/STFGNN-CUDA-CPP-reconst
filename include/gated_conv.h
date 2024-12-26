#ifndef GATED_CONV_H
#define GATED_CONV_H

#include <Eigen/Dense>

class GatedConvLayer {
public:
    GatedConvLayer(size_t inputDim, size_t kernelSize, size_t dilation);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& nodeFeatures);

private:
    size_t inputDim;
    size_t kernelSize;
    size_t dilation;

    Eigen::MatrixXd weights_tanh;
    Eigen::MatrixXd weights_sigmoid;

    Eigen::VectorXd bias_tanh;
    Eigen::VectorXd bias_sigmoid;

    Eigen::MatrixXd applyConvolution(const Eigen::MatrixXd& nodeFeatures, const Eigen::MatrixXd& weights, const Eigen::VectorXd& bias);
};

#endif
