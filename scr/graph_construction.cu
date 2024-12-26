#include "graph_construction.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// CUDA kernel for constructing a spatial graph
__global__ void construct_spatial_graph(float* data, int size, int* graph) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; ++j) {
            if (abs(data[idx] - data[j]) < 1.0f) {  // Example condition
                graph[idx * size + j] = 1;
            } else {
                graph[idx * size + j] = 0;
            }
        }
    }
}

// CUDA kernel for constructing a temporal graph
__global__ void construct_temporal_graph(float* data, int size, int* graph) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        graph[idx * size + idx + 1] = 1;  // Temporal connection
    }
}

// CUDA kernel for fusion graph
__global__ void construct_fusion_graph(int* spatial_graph, int* temporal_graph, int size, int* fusion_graph) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        fusion_graph[idx] = spatial_graph[idx] || temporal_graph[idx];
    }
}

// CUDA kernel for constructing dynamic graphs
__global__ void construct_dynamic_graph(float* data, int size, int* graph, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int j = 0; j < size; ++j) {
            graph[idx * size + j] = (data[idx] + data[j]) > threshold ? 1 : 0;
        }
    }
}

GraphConstruction::GraphConstruction(const std::string& config_path) : config_path(config_path) {}

std::vector<std::vector<int>> GraphConstruction::build_spatial_graph(const std::vector<float>& data) {
    int size = data.size();
    std::vector<std::vector<int>> spatial_graph(size, std::vector<int>(size));
    float* d_data;
    int* d_graph;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_graph, size * size * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    construct_spatial_graph<<<blocks, threads_per_block>>>(d_data, size, d_graph);

    cudaMemcpy(spatial_graph.data()->data(), d_graph, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_graph);

    return spatial_graph;
}

std::vector<std::vector<int>> GraphConstruction::build_temporal_graph(const std::vector<float>& data) {
    int size = data.size();
    std::vector<std::vector<int>> temporal_graph(size, std::vector<int>(size, 0));
    int* d_graph;
    cudaMalloc(&d_graph, size * size * sizeof(int));

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    construct_temporal_graph<<<blocks, threads_per_block>>>(nullptr, size, d_graph);

    cudaMemcpy(temporal_graph.data()->data(), d_graph, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_graph);

    return temporal_graph;
}

std::vector<std::vector<int>> GraphConstruction::build_fusion_graph(const std::vector<float>& spatial_graph, const std::vector<float>& temporal_graph) {
    int size = spatial_graph.size();
    std::vector<std::vector<int>> fusion_graph(size, std::vector<int>(size));
    int* d_spatial_graph;
    int* d_temporal_graph;
    int* d_fusion_graph;

    cudaMalloc(&d_spatial_graph, size * size * sizeof(int));
    cudaMalloc(&d_temporal_graph, size * size * sizeof(int));
    cudaMalloc(&d_fusion_graph, size * size * sizeof(int));

    cudaMemcpy(d_spatial_graph, spatial_graph.data()->data(), size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temporal_graph, temporal_graph.data()->data(), size * size * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (size * size + threads_per_block - 1) / threads_per_block;
    construct_fusion_graph<<<blocks, threads_per_block>>>(d_spatial_graph, d_temporal_graph, size, d_fusion_graph);

    cudaMemcpy(fusion_graph.data()->data(), d_fusion_graph, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_spatial_graph);
    cudaFree(d_temporal_graph);
    cudaFree(d_fusion_graph);

    return fusion_graph;
}

std::vector<std::vector<int>> GraphConstruction::build_dynamic_graph(const std::vector<float>& data) {
    int size = data.size();
    std::vector<std::vector<int>> dynamic_graph(size, std::vector<int>(size));
    float* d_data;
    int* d_graph;

    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_graph, size * size * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    construct_dynamic_graph<<<blocks, threads_per_block>>>(d_data, size, d_graph, 1.0f);

    cudaMemcpy(dynamic_graph.data()->data(), d_graph, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_graph);

    return dynamic_graph;
}
