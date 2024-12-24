/**
 * @brief It accepts a dense graph and counts the number of intersections 
 * between neighbouring lists of two nodes u,v that are connected by an edge u -> v. 
 * The result is stored per node in the sum array.
 * 
 * @tparam T uint32_t or uint64_t
 * @param nodesMax total number of nodes
 * @param nodesMin compressed number of nodes
 * @param sum pointer to array that stores results for each node
 * @param graph input graph
 */
template <typename T>
__global__ void denseKernel(T nodesMax, T nodesMin, T *sum, T *graph)
{
    T threadId = blockDim.x * blockIdx.x + threadIdx.x;
    T u = threadId / 32;
    T laneId = threadId % 32;
    T mask = 1ULL << laneId;
    T uNeighbours = nodesMin * u;
    T count = 0;
    for (T i = 0; i < nodesMin; i++)
    {
        T currNeighbour = graph[uNeighbours + i];
        T v = 32 * i + laneId;
        T vNeighbours = nodesMin * v;
        if (currNeighbour & mask)
        {
            for (T j = 0; j < nodesMin; j++)
            {
                count += __popcll(graph[uNeighbours + j] & graph[vNeighbours + j]);
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[u] = count;
}
