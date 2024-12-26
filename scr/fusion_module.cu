#include "fusion_module.h"
#include <cuda_runtime.h>

__global__ void fusion_operation(float* spatial, float* temporal, float* output, float* weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = weights[idx] * (spatial[idx] + temporal[idx]);
    }
}

FusionModule::FusionModule(int input_dim) : input_dim(input_dim) {
    fusion_weights.resize(input_dim, 0.5f); // Example initialization
}

std::vector<float> FusionModule::fuse(const std::vector<float>& spatial, const std::vector<float>& temporal) {
    std::vector<float> output(input_dim);
    float* d_spatial;
    float* d_temporal;
    float* d_output;
    float* d_weights;

    cudaMalloc(&d_spatial, input_dim * sizeof(float));
    cudaMalloc(&d_temporal, input_dim * sizeof(float));
    cudaMalloc(&d_output, input_dim * sizeof(float));
    cudaMalloc(&d_weights, fusion_weights.size() * sizeof(float));

    cudaMemcpy(d_spatial, spatial.data(), input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temporal, temporal.data(), input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, fusion_weights.data(), fusion_weights.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (input_dim + threads_per_block - 1) / threads_per_block;

    fusion_operation<<<blocks, threads_per_block>>>(d_spatial, d_temporal, d_output, d_weights, input_dim);

    cudaMemcpy(output.data(), d_output, input_dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_spatial);
    cudaFree(d_temporal);
    cudaFree(d_output);
    cudaFree(d_weights);

    return output;
}
