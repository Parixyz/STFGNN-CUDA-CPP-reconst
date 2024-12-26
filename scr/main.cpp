#include "dataloader.h"
#include "preprocessing.h"
#include <iostream>

int main() {
    // Filepath to the PeMS dataset
    std::string filePath = "PeMS4.csv";

    try {
        // Initialize the DataLoader
        DataLoader dataLoader(filePath);

        // Access training data
        const auto& trainSet = dataLoader.getTrainSet();
        const size_t numRows = trainSet.size();
        const size_t numCols = trainSet[0].size();

        // Prepare data for CUDA
        float* data = new float[numRows * numCols];
        float* mean = new float[numCols];
        float* stdDev = new float[numCols];

        // Flatten data for CUDA
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                data[i * numCols + j] = trainSet[i][j];
            }
        }

        // Retrieve normalization parameters
        const auto& meanVec = dataLoader.getMean();
        const auto& stdDevVec = dataLoader.getStdDev();

        for (size_t i = 0; i < numCols; ++i) {
            mean[i] = meanVec[i];
            stdDev[i] = stdDevVec[i];
        }

        // Normalize data using CUDA
        normalizeDataCUDA(data, mean, stdDev, numRows, numCols);

        // Print normalized data (optional)
        std::cout << "Normalized Training Data:\n";
        for (size_t i = 0; i < numRows; ++i) {
            for (size_t j = 0; j < numCols; ++j) {
                std::cout << data[i * numCols + j] << " ";
            }
            std::cout << "\n";
        }

        // Cleanup
        delete[] data;
        delete[] mean;
        delete[] stdDev;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
