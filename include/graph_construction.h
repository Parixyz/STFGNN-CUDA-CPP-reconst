#ifndef GRAPH_CONSTRUCTION_H
#define GRAPH_CONSTRUCTION_H

#include <vector>
#include <string>

class GraphConstruction {
public:
    GraphConstruction(const std::string& config_path);
    std::vector<std::vector<int>> build_spatial_graph(const std::vector<float>& data);
    std::vector<std::vector<int>> build_temporal_graph(const std::vector<float>& data);
    std::vector<std::vector<int>> build_fusion_graph(const std::vector<float>& spatial_graph, const std::vector<float>& temporal_graph);
    std::vector<std::vector<int>> build_dynamic_graph(const std::vector<float>& data);

private:
    std::string config_path;
};

#endif // GRAPH_CONSTRUCTION_H
