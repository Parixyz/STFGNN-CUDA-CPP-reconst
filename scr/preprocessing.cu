#include "preprocessing.h"
#include <cuda_runtime.h>

__global__ void zscore_normalize(float* data, int size, float mean, float stddev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (data[idx] - mean) / stddev;
    }
}

void preprocess_data(std::vector<float>& data, float mean, float stddev) {
    float* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(float));
    cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (data.size() + threads_per_block - 1) / threads_per_block;
    zscore_normalize<<<blocks, threads_per_block>>>(d_data, data.size(), mean, stddev);

    cudaMemcpy(data.data(), d_data, data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
