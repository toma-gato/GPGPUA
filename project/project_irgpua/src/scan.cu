#include "scan.cuh"

#include <iostream>

#include "reduce.cuh"

/**
 * @brief Kogge-stone scan on reduced blocks
 *
 * @param reduced_blocks
 * @param d_output
 * @return
 */
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

/**
 * @brief
 *
 * @param d_data
 * @param d_scanned_blocks
 * @param d_output
 * @return __global__
 */
__global__ void propagate(cuda::std::span<int> d_data, cuda::std::span<int> d_scanned_blocks, cuda::std::span<int> d_output)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    if (i < d_data.size())
    {
        d_output[i] = d_data[i] + d_scanned_blocks[bid];
    }
}

/**
 * @brief
 *
 * @param d_data
 * @param d_output
 */
void exclusive_scan(rmm::device_vector<int> &d_data, rmm::device_vector<int> &d_output)
{
    size_t num_elements = d_data.size();
    size_t block_size = 256;
    size_t block_count = (num_elements + block_size - 1) / block_size;

    cuda::std::span<int> data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());

    rmm::device_vector<int> reduced_blocks(block_count);
    cuda::std::span<int> reduced_blocks_span(thrust::raw_pointer_cast(reduced_blocks.data()), reduced_blocks.size());

    reduce_block<<<block_count, block_size>>>(data_span, reduced_blocks_span);

    rmm::device_vector<int> d_scanned_blocks(block_count);
    cuda::std::span<int> scanned_blocks_span(thrust::raw_pointer_cast(d_scanned_blocks.data()), d_scanned_blocks.size());
    size_t shared_memsize = block_count * sizeof(int);
    scan_block<<<1, block_count, shared_memsize>>>(reduced_blocks_span, scanned_blocks_span);

    cuda::std::span<int> d_output_span(thrust::raw_pointer_cast(d_output.data()), d_output.size());

    propagate<<<block_count, block_size>>>(data_span, scanned_blocks_span, d_output_span);
}
