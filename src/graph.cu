#include <iostream>
#include "graph.h"
#include "algorithm/dense.cuh"
#include "algorithm/sparse.cuh"

template class Graph<uint32_t>;
template class Graph<uint64_t>;

/**
 * @brief Creates a new Graph instance for a graph made up
 * of @param n nodes and @param e edges. @param filepath denotes the
 * location of the input file for the graph. @param actualResult is the
 * true triangle count for the graph. @param format describes the
 * format of the graph:
 * u v c, 0-indexed -> 0
 * u v c, 1-indexed -> 1
 * u v,   0-indexed -> 2
 * u v,   1-indexed -> 3
 * @param filepath
 * @param n
 * @param e
 * @param format
 * @param actualResult
 */
template <typename T>
Graph<T>::Graph(fs::path path, T n, T e, T format, uint64_t actualResult) : filePath(path), numberOfNodes(n), numberOfEdges(e), inputFormat(format), actualTriangleCount(actualResult)
{
    nodesMax = n;
    nodesSquared = 1ULL * n * n;
    density = 1.0 * e / nodesSquared;
}

/**
 * @brief Will free memory allocated on the device.
 *
 */
template <typename T>
Graph<T>::~Graph()
{
    if (denseGraph)
    {
        cudaFree(denseGraph);
    }
}

/**
 * @brief Will read the graph into an adjacency list or adjacency matrix
 * depending on the ratio between edges and nodes in the graph.
 */
template <typename T>
void Graph<T>::read()
{
    if (density >= 0.03)
    {
        selectedAlgorithm = DENSE;
        readDense();
    }
    else
    {
        selectedAlgorithm = ONE_THREAD_PER_EDGE;
        readSparse();
    }
}

template <typename T>
uint64_t Graph<T>::launchAlgorithm()
{
    std::cout << selectedAlgorithm << std::endl;
    switch (selectedAlgorithm)
    {
    case DENSE:
        return launchDense();
    case ONE_THREAD_PER_EDGE:
        return launchSparseEdgeMerge();
    case ONE_WARP_PER_NODE_MERGE:
        return launchSparseNodeMerge();
    case ONE_WARP_PER_NODE_BINARY:
        return launchSparseNodeBinary();
    case ONE_WARP_PER_EDGE:
        return launchSparseEdgeBinary();
    default:
        return 0ULL;
    }
}

/**
 * @brief This method will read the graph into a dense adjacency matrix.
 * This method efficiently uses the memory by assigning 1 bit in
 * a row to a neighbour instead of one byte.
 * This method will allocate memory on the device.
 */
template <typename T>
void Graph<T>::readDense()
{
    nodesMin = (nodesMax + 31) / 32;
    T memSize = nodesMax * nodesMin * sizeof(T);
    T *hostGraph = (T *)malloc(memSize);
    cudaMalloc(&denseGraph, memSize);
    for (T i = 0; i < nodesMax * nodesMin; i++)
    {
        hostGraph[i] = 0ULL;
    }

    std::ifstream inputStream(filePath);
    T u, v;
    std::string c;
    bool oneIndexed = (inputFormat % 2 == 1);
    bool weightedGraph = (inputFormat < 2);
    while (inputStream >> u >> v)
    {
        if (weightedGraph)
        {
            inputStream >> c;
        }

        if (oneIndexed)
        {
            --u;
            --v;
        }
        T left = std::min(u, v);
        T right = std::max(u, v);

        T offset = right / 32;
        T shiftAmount = right % 32;
        hostGraph[nodesMin * left + offset] |= (1ULL << shiftAmount);
    }
    cudaMemcpy(denseGraph, hostGraph, memSize, cudaMemcpyHostToDevice);
}

/**
 * @brief This method will read the graph into a sparse adjacency list.
 * It first reads neighbours into a vector and then orders them using
 * std::sort and removes duplicates with erase(). While reading the graph,
 * it collects various metrics for choosing the optimal algorithm on the GPU.
 */
template <typename T>
void Graph<T>::readSparse()
{
    for (T i = 0; i < numberOfNodes; i++)
    {
        std::vector<T> u;
        sparseGraph.push_back(u);
    }
    std::ifstream inputStream(filePath);
    T u, v;
    std::string c;
    bool oneIndexed = (inputFormat % 2 == 1);
    bool weightedGraph = (inputFormat < 2);
    T count = 0;

    while (inputStream >> u >> v)
    {
        if (weightedGraph)
        {
            inputStream >> c;
        }

        if (oneIndexed)
        {
            --u;
            --v;
        }

        T left = std::min(u, v);
        T right = std::max(u, v);
        sparseGraph.at(left).push_back(right);
        degrees.at(left)++;
        degrees.at(right)++;
    }

    for (T i = 0; i < numberOfNodes; i++)
    {
        std::sort(sparseGraph.at(i).begin(), sparseGraph.at(i).end());
        sparseGraph.at(i).erase(std::unique(sparseGraph.at(i).begin(), sparseGraph.at(i).end()), sparseGraph.at(i).end());
        count += sparseGraph.at(i).size();
        sumOfDegrees += degrees.at(i);
        if (maxDegree < degrees.at(i))
        {
            maxDegree = degrees.at(i);
        }
    }
    numberOfEdges = count;

    // Metrics
    avgDegree = 1.0 * sumOfDegrees / numberOfNodes;

    // mode
    std::vector<T> histogram(maxDegree + 1, 0);
    for (T i = 0; i < numberOfNodes; i++)
    {
        histogram.at(degrees.at(i))++;
    }
    mode = std::distance(histogram.begin(), std::max_element(histogram.begin(), histogram.end()));

    // median
    auto m = degrees.begin() + degrees.size() / 2;
    std::nth_element(degrees.begin(), m, degrees.end());
    median = degrees.at(degrees.size() / 2);

    // mean and mode variance
    for (T i = 0; i < numberOfNodes; i++)
    {
        meanSigma += abs(degrees.at(i) - avgDegree);
        modeSigma += abs((degrees.at(i) - mode) * 1.0);
    }
    meanSigma /= numberOfNodes;
    modeSigma /= numberOfNodes;

    if (mode < 32)
    {
        selectedAlgorithm = ONE_THREAD_PER_EDGE;
    }
    else
    {
        if (avgDegree >= median - 5 && avgDegree <= median + 5 && avgDegree >= mode - 5 &&
            avgDegree <= mode + 5 && median >= mode - 5 && median <= mode + 5)
        {
            selectedAlgorithm = ONE_WARP_PER_NODE_MERGE;
        }
    }
}

template <typename T>
u_int64_t Graph<T>::launchDense()
{
    T memSize = nodesMax * sizeof(T);
    T *deviceSum;
    cudaMalloc(&deviceSum, memSize);
    T *sumHost = (T *)malloc(memSize);
    for (T i = 0; i < nodesMax; i++)
    {
        sumHost[i] = 0;
    }
    cudaMemcpy(deviceSum, sumHost, memSize, cudaMemcpyHostToDevice);

    T threadsPerBlock = 128;
    T blocksPerGrid = (32 * nodesMax + threadsPerBlock - 1) / threadsPerBlock;
    denseKernel<<<blocksPerGrid, threadsPerBlock>>>(nodesMax, nodesMin, deviceSum, denseGraph);
    cudaDeviceSynchronize();

    cudaMemcpy(sumHost, deviceSum, memSize, cudaMemcpyDeviceToHost);
    u_int64_t res = 0;
    for (T i = 0; i < nodesMax; i++)
    {
        res += sumHost[i];
    }
    cudaFree(deviceSum);
    free(sumHost);
    return res;
}

template <typename T>
u_int64_t Graph<T>::launchSparseEdgeMerge()
{
    T *hostNodes, *hostEdgeOffsets, *hostEdgeNeighbours, *hostSum,
        *deviceNodes, *deviceEdgeOffsets, *deviceEdgeNeighbours, *deviceSum;
    cudaMalloc(&deviceEdgeOffsets, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceEdgeNeighbours, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceNodes, (numberOfNodes + 1) * sizeof(T));
    cudaMalloc(&deviceSum, numberOfEdges * sizeof(T));
    hostNodes = (T *)malloc((numberOfNodes + 1) * sizeof(T));
    hostEdgeOffsets = (T *)malloc(numberOfEdges * sizeof(T));
    hostEdgeNeighbours = (T *)malloc(numberOfEdges * sizeof(T));
    hostSum = (T *)malloc(numberOfEdges * sizeof(T));

    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostSum[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostEdgeOffsets[i] = 0;
        hostEdgeNeighbours[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = 0;
    }
    hostNodes[numberOfNodes] = numberOfEdges;
    T it = 0;
    for (T i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = it;
        for (T j = 0; j < sparseGraph.at(i).size(); j++)
        {
            hostEdgeOffsets[it] = i;
            hostEdgeNeighbours[it] = sparseGraph.at(i).at(j);
            it++;
        }
    }

    cudaMemcpy(deviceEdgeOffsets, hostEdgeOffsets,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdgeNeighbours, hostEdgeNeighbours,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNodes, hostNodes,
               (numberOfNodes + 1) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSum, hostSum,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);

    unsigned int threadsPerBlock = 128;
    unsigned int blocksPerGrid = (numberOfEdges + threadsPerBlock - 1) / threadsPerBlock;
    sparseEdgeMergeKernel<<<blocksPerGrid, threadsPerBlock>>>(numberOfNodes, numberOfEdges, deviceNodes,
                                                              deviceEdgeOffsets, deviceEdgeNeighbours, deviceSum);
    cudaDeviceSynchronize();
    cudaMemcpy(hostSum, deviceSum, numberOfEdges * sizeof(T), cudaMemcpyDeviceToHost);

    u_int64_t res = 0;
    for (T i = 0; i < numberOfEdges; i++)
    {
        res += hostSum[i];
    }
    return res;
}

template <typename T>
u_int64_t Graph<T>::launchSparseNodeMerge()
{
    T *hostEdges, *hostNodes, *hostSum, *deviceEdges, *deviceNodes, *deviceSum;
    hostEdges = (T *)malloc(numberOfEdges * sizeof(T));
    hostNodes = (T *)malloc((numberOfNodes + 1) * sizeof(T));
    hostSum = (T *)malloc(numberOfEdges * sizeof(T));
    cudaMalloc(&deviceEdges, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceNodes, (numberOfNodes + 1) * sizeof(T));
    cudaMalloc(&deviceSum, numberOfEdges * sizeof(T));
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostSum[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostEdges[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = 0;
    }
    hostNodes[numberOfNodes] = numberOfEdges;

    T idx = 0;
    for (T i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = idx;
        for (T j = 0; j < sparseGraph.at(i).size(); j++)
        {
            hostEdges[idx] = sparseGraph.at(i).at(j);
            idx++;
        }
    }

    cudaMemcpy(deviceEdges, hostEdges, numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNodes, hostNodes, (numberOfNodes + 1) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSum, hostSum, numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);

    unsigned int threadsPerBlock = 128;
    unsigned int blocksPerGrid = (32 * numberOfNodes + threadsPerBlock - 1) / threadsPerBlock;
    sparseNodeMergeKernel<<<blocksPerGrid, threadsPerBlock>>>(numberOfNodes, numberOfEdges, deviceNodes, deviceEdges, deviceSum);
    cudaDeviceSynchronize();
    cudaMemcpy(hostSum, deviceSum, numberOfEdges * sizeof(T), cudaMemcpyDeviceToHost);

    u_int64_t res = 0;
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        res += hostSum[i];
    }
    return res;
}

template <typename T>
u_int64_t Graph<T>::launchSparseNodeBinary()
{
    T *hostEdges, *hostNodes, *hostSum, *deviceEdges, *deviceNodes, *deviceSum;
    hostEdges = (T *)malloc(numberOfEdges * sizeof(T));
    hostNodes = (T *)malloc((numberOfNodes + 1) * sizeof(T));
    hostSum = (T *)malloc(numberOfEdges * sizeof(T));
    cudaMalloc(&deviceEdges, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceNodes, (numberOfNodes + 1) * sizeof(T));
    cudaMalloc(&deviceSum, numberOfEdges * sizeof(T));
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostSum[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostEdges[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = 0;
    }
    hostNodes[numberOfNodes] = numberOfEdges;

    T idx = 0;
    for (T i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = idx;
        for (T j = 0; j < sparseGraph.at(i).size(); j++)
        {
            hostEdges[idx] = sparseGraph.at(i).at(j);
            idx++;
        }
    }

    cudaMemcpy(deviceEdges, hostEdges, numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNodes, hostNodes, (numberOfNodes + 1) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSum, hostSum, numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);

    unsigned int threadsPerBlock = 128;
    unsigned int blocksPerGrid = (32 * numberOfNodes + threadsPerBlock - 1) / threadsPerBlock;
    sparseNodeBinaryKernel<<<blocksPerGrid, threadsPerBlock>>>(numberOfNodes, numberOfEdges, deviceNodes, deviceEdges, deviceSum);
    cudaDeviceSynchronize();
    cudaMemcpy(hostSum, deviceSum, numberOfEdges * sizeof(T), cudaMemcpyDeviceToHost);

    u_int64_t res = 0;
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        res += hostSum[i];
    }
    return res;
}

template <typename T>
u_int64_t Graph<T>::launchSparseEdgeBinary()
{
    T *hostNodes, *hostEdgeOffsets, *hostEdgeNeighbours, *hostSum,
        *deviceNodes, *deviceEdgeOffsets, *deviceEdgeNeighbours, *deviceSum;
    cudaMalloc(&deviceEdgeOffsets, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceEdgeNeighbours, numberOfEdges * sizeof(T));
    cudaMalloc(&deviceNodes, (numberOfNodes + 1) * sizeof(T));
    cudaMalloc(&deviceSum, numberOfEdges * sizeof(T));
    hostNodes = (T *)malloc((numberOfNodes + 1) * sizeof(T));
    hostEdgeOffsets = (T *)malloc(numberOfEdges * sizeof(T));
    hostEdgeNeighbours = (T *)malloc(numberOfEdges * sizeof(T));
    hostSum = (T *)malloc(numberOfEdges * sizeof(T));

    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostSum[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfEdges; i++)
    {
        hostEdgeOffsets[i] = 0;
        hostEdgeNeighbours[i] = 0;
    }
    for (unsigned int i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = 0;
    }
    hostNodes[numberOfNodes] = numberOfEdges;
    T it = 0;
    for (T i = 0; i < numberOfNodes; i++)
    {
        hostNodes[i] = it;
        for (T j = 0; j < sparseGraph.at(i).size(); j++)
        {
            hostEdgeOffsets[it] = i;
            hostEdgeNeighbours[it] = sparseGraph.at(i).at(j);
            it++;
        }
    }

    cudaMemcpy(deviceEdgeOffsets, hostEdgeOffsets,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEdgeNeighbours, hostEdgeNeighbours,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNodes, hostNodes,
               (numberOfNodes + 1) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSum, hostSum,
               numberOfEdges * sizeof(T), cudaMemcpyHostToDevice);

    uint64_t warpCount = 32ULL * numberOfEdges;
    T threadsPerBlock = 128;
    uint64_t blocksPerGrid = (warpCount + threadsPerBlock - 1) / threadsPerBlock;
    sparseEdgeBinaryKernel<<<blocksPerGrid, threadsPerBlock>>>(numberOfNodes, numberOfEdges, deviceNodes,
                                                               deviceEdgeOffsets, deviceEdgeNeighbours, deviceSum);
    cudaDeviceSynchronize();
    cudaMemcpy(hostSum, deviceSum, numberOfEdges * sizeof(T), cudaMemcpyDeviceToHost);

    u_int64_t res = 0;
    for (T i = 0; i < numberOfEdges; i++)
    {
        res += hostSum[i];
    }
    return res;
}