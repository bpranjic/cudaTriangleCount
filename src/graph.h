#ifndef GRAPH_H
#define GRAPH_H

#include <filesystem>

#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

// SELECTED ALGORITHM
// 1 thread per edge = 0
// 1 warp node, merge = 1
// 1 warp per node, binary search = 2
// 1 warp per edge = 3
// 2 kernels hybrid approach = 4
enum Algorithm
{
    ONE_THREAD_PER_EDGE = 0,
    ONE_WARP_PER_NODE_MERGE = 1,
    ONE_WARP_PER_NODE_BINARY = 2,
    ONE_WARP_PER_EDGE = 3,
    HYBRID = 4,
    DENSE = 5
};

namespace fs = std::filesystem;

template <typename T>
class Graph
{
public:
    Graph(fs::path filepath, T n, T e, T format, uint64_t actualResult);
    ~Graph();
    void read();
    uint64_t launchAlgorithm();
    uint64_t actualTriangleCount;

private:
    T numberOfNodes;
    T numberOfEdges;
    fs::path filePath;
    // GRAPH FORMATS
    // u v c, starts with 0 type = 0
    // u v c, start with 1  type = 1
    // u v, starts with 0   type = 2
    // u v, starts with 1   type = 3
    T inputFormat;
    std::vector<std::vector<T>> sparseGraph;

    // Dense Implementation
    T *denseGraph;
    T nodesMax;
    T nodesMin = 0;

    // Metrics
    T maxDegree = 0;
    uint64_t nodesSquared;
    double density;
    uint64_t sumOfDegrees = 0;
    double avgDegree = 0.0;
    T median = 0;
    T mode = 0;
    double meanSigma = 0.0;
    double modeSigma = 0.0;
    std::vector<T> degrees = std::vector<T>(numberOfNodes, 0);
    Algorithm selectedAlgorithm = ONE_THREAD_PER_EDGE;

    // Read methods
    void readDense();
    void readSparse();

    // Algorithms
    uint64_t launchDense();
    uint64_t launchSparseEdgeMerge();
    uint64_t launchSparseNodeMerge();
    uint64_t launchSparseNodeBinary();
    uint64_t launchSparseEdgeBinary();
};

#endif