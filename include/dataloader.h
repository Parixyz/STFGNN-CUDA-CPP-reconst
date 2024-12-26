#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <unordered_map>

class DataLoader {
public:
    DataLoader(const std::string& dataset_path);
    std::vector<std::vector<float>> load_data();
    std::unordered_map<std::string, float> compute_statistics();

private:
    std::string dataset_path;
    std::vector<std::vector<float>> data;
    std::unordered_map<std::string, float> stats;
};

#endif // DATALOADER_H
