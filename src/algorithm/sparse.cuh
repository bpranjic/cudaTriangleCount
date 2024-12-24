/**
 * @brief Method that returns the number of intersections between two
 * sorted arrays.
 * 
 * @tparam T uint32_t or uint64_t
 * @param lu left bound for u
 * @param ru right bound for u
 * @param lv left bound for v
 * @param rv right bound for v
 * @return returns the number of intersections
 */
template <typename T>
__device__ T mergeIntersectCount(T* lu, T* ru, T* lv, T* rv){
    T count = 0;
    while(lu != ru && lv != rv){
        if(*lu < *lv){
            ++lu;
        }
        else{
            if(!(*lv < *lu)){
               count++;
                *lu++;
            }
            ++lv;
        }
    } 
    return count;
}

/**
 * @brief Returns true if x is found in the array search region bound by [l, r]
 * 
 * @tparam T uint32_t or uint64_t
 * @param arr search array
 * @param l left bound of search region
 * @param r right bound of search region
 * @param x element to be found
 * @return Returns true if x in arr[l:r], else false
 */
template <typename T>
__device__ bool binarySearch(T* arr, T l, T r, T x){
    while(l <= r){
        int m = l + (r - l + 1) / 2;
        if(arr[m] == x){
            return true;
        }
        if(arr[m] < x){
            l = m + 1;
        }
        else{
            r = m - 1;
        }
    }
    return false;
}

/**
 * @brief Each thread counts the number of intersections between
 * the neighbourhoods of the nodes u, v for an edge u -> v.
 * Edges are stored with 3 arrays, one array that holds all nodes,
 * one array that holds v for every edge u -> v and one array that
 * stores the starting offset for each neighbourhood for every node.
 * Intersection is performed merge-based.
 * 
 * @tparam T uint32_t or uint64_t
 * @param numberOfNodes number of nodes in the graph
 * @param numberOfEdges number of edges in the graph
 * @param nodes pointer to an array of nodes
 * @param edgeOffsets Starting offset of the neighbours for each node u
 * @param edgeNeighbours array of v's for all edges u -> v
 * @param sum pointer to array that stores results for each edge
 */
template <typename T>
__global__ void sparseEdgeMergeKernel(T numberOfNodes, T numberOfEdges, T *nodes, T *edgeOffsets, T*edgeNeighbours, T *sum){
    T threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadId < numberOfEdges){
        T u = edgeOffsets[threadId];
        T v = edgeNeighbours[threadId];
        T* lu = &edgeNeighbours[nodes[u]];
        T* lv = &edgeNeighbours[nodes[v]];
        T* ru = &edgeNeighbours[nodes[u + 1]];
        T* rv = &edgeNeighbours[nodes[v + 1]];
        sum[threadId] = mergeIntersectCount(lu, ru, lv, rv);
    }
}

/**
 * @brief Launches one warp per node. Each thread in a warp will count
 * the intersections between two neighbourhoods of two nodes.
 * Intersection is merge-based. 
 * 
 * @param numberOfNodes number of nodes
 * @param numberOfEdges number of edges
 * @param nodes nodes in the graph
 * @param edges edges in the graph 
 * @param sum pointer to array that stores result for each node
 */
template <typename T>
__global__ void sparseNodeMergeKernel(T numberOfNodes, T numberOfEdges, T *nodes, T*edges, T *sum){
    T threadId = blockIdx.x * blockDim.x + threadIdx.x;
    T warpId = threadId / 32;
    T laneId = threadId % 32;
    if(warpId < numberOfNodes){
        T u = warpId;
        T uSize = nodes[u + 1] - nodes[u];
        T* lu = &edges[nodes[u]];
        T* ru = lu + uSize;
        T count = 0;
        for(T i = laneId; i < uSize; i+=32){
            T v = *(lu+i);
            T vSize = nodes[v + 1] - nodes[v];
            T* lv = &edges[nodes[v]];
            T* rv = lv + vSize;
            count += mergeIntersectCount(lu, ru, lv, rv);
        }
        __syncthreads();
        for(int offset = 16; offset > 0; offset /= 2){
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        sum[u] = count;
    }
}

/**
 * @brief Launches one warp per node. Each thread in a warp will count
 * the intersections between two neighbourhoods of two nodes.
 * Intersection is based on a binary search, each thread will look in parallel
 * for the existence of one neighbour from u in the neighbourhood of v.
 * 
 * @param numberOfNodes number of nodes
 * @param numberOfEdges number of edges
 * @param nodes nodes in the graph
 * @param edges edges in the graph 
 * @param sum pointer to array that stores result for each node
 */
template <typename T>
__global__ void sparseNodeBinaryKernel(T numberOfNodes, T numberOfEdges, T *nodes, T*edges, T *sum){
    T threadId = blockIdx.x * blockDim.x + threadIdx.x;
    T warpId = threadId / 32;
    T laneId = threadId % 32;
    if(warpId < numberOfNodes){
        T u = warpId;
        T uSize = nodes[u + 1] - nodes[u];
        T* lu = &edges[nodes[u]];
        T* ru = lu + uSize;
        T count = 0;
        for(T i = 0; i < uSize; i++){
            T v = *(lu + i);
            T* lv = &edges[nodes[v]];
            T vSize = nodes[v + 1] - nodes[v];
            for(T j = laneId; j < vSize; j+=32){
                count += binarySearch(lu, (T)0, uSize - 1, *(lv + j));
            }
        }
        __syncthreads();
        for(int offset = 16; offset > 0; offset /= 2){
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        sum[u] = count;
    }
}

//1 warp per edge, parallel binary search based intersection with strided access
/**
 * @brief Launches one warp per edge. Each thread in a warp will count
 * the intersections between two neighbourhoods of two nodes.
 * Intersection is based on a binary search, each thread will look in parallel
 * for the existence of one neighbour from u in the neighbourhood of v
 * 
 * @tparam T uint32_t or uint64_t
 * @param numberOfNodes number of nodes in the graph
 * @param numberOfEdges number of edges in the graph
 * @param nodes pointer to an array of nodes
 * @param edgeOffsets Starting offset of the neighbours for each node u
 * @param edgeNeighbours array of v's for all edges u -> v
 * @param sum pointer to array that stores results for each edge
 */
template <typename T>
__global__ void sparseEdgeBinaryKernel(T numberOfNodes, T numberOfEdges, T *nodes, T *edgeOffsets, T*edgeNeighbours, T *sum){
    u_int64_t threadId = 1ULL * blockDim.x * blockIdx.x + threadIdx.x ;
    u_int64_t warpId = threadId / 32;
    T laneId = threadId % 32;
    T u = edgeOffsets[warpId];
    T v = edgeNeighbours[warpId];
    T uSize = nodes[u + 1] - nodes[u];
    T vSize = nodes[v + 1] - nodes[v];
    T *lu = &edgeNeighbours[nodes[u]];
    T *lv = &edgeNeighbours[nodes[v]];
    T count = 0;
    for(T i = laneId; i < vSize; i+=32){
        count += binarySearch(lu, (T)0, uSize - 1, *(lv + i));
    }
    for(int offset = 16; offset > 0; offset /= 2){
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    sum[warpId] = count;
}
