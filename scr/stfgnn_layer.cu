#include "stfgnn_layer.h"
#include <cuda_runtime.h>

__global__ void spatial_temporal_convolution(float* input, float* output, float* weights, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        float result = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            result += input[i] * weights[i * output_dim + idx];
        }
        output[idx] = result;
    }
}

STFGNNLayer::STFGNNLayer(int input_dim, int output_dim)
    : input_dim(input_dim), output_dim(output_dim) {
    weights.resize(input_dim * output_dim, 0.5f); // Example initialization
}

std::vector<float> STFGNNLayer::forward(const std::vector<float>& input) {
    std::vector<float> output(output_dim);
    float* d_input;
    float* d_output;
    float* d_weights;

    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output, output.size() * sizeof(float));
    cudaMalloc(&d_weights, weights.size() * sizeof(float));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (output_dim + threads_per_block - 1) / threads_per_block;

    spatial_temporal_convolution<<<blocks, threads_per_block>>>(d_input, d_output, d_weights, input_dim, output_dim);

    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weights);

    return output;
}
