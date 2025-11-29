#include "scan.cuh"

#include <iostream>

#include "reduce.cuh"


__global__ void scan_block(cuda::std::span<int> reduced_blocks, cuda::std::span<int> d_output)
{
    extern __shared__ int s_data[];
    int tid = threadIdx.x;

    // Load data into shared memory (exclusive scan -> shift by 1)
    s_data[tid] = (tid > 0) ? reduced_blocks[tid - 1] : 0;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int val = 0;
        if (tid >= offset)
        {
            val = s_data[tid - offset];
        }
        __syncthreads();

        s_data[tid] += val;
        __syncthreads();
    }

    // Write result to global memory
    d_output[tid] = s_data[tid];
}


__global__ void propagate(cuda::std::span<int> d_data, cuda::std::span<int> d_scanned_blocks, cuda::std::span<int> d_output)
{
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    // Load block data into shared memory (pad out-of-range with 0)
    s_data[tid] = (i < d_data.size()) ? d_data[i] : 0;
    __syncthreads();

    // Inclusive scan (Kogge-Stone style) in shared memory
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int val = 0;
        if (tid >= offset)
        {
            val = s_data[tid - offset];
        }
        __syncthreads();

        s_data[tid] += val;
        __syncthreads();
    }

    // Convert inclusive scan to exclusive by shifting right by one
    int exclusive = (tid == 0) ? 0 : s_data[tid - 1];

    // Add block prefix (scanned_blocks) and write to output.
    // Support two output layouts:
    // - If `d_output.size() == d_data.size() + 1` -> write shifted result (output[0] = 0, output[i+1] = sum up to i)
    // - If `d_output.size() == d_data.size()` -> write exclusive scan in-place (output[i] = sum of elements before i)
    if (i < d_data.size())
    {
        if (d_output.size() == d_data.size() + 1)
        {
            d_output[i + 1] = exclusive + d_data[i] + d_scanned_blocks[bid];
        }
        else
        {
            d_output[i] = exclusive + d_scanned_blocks[bid];
        }
    }
}


