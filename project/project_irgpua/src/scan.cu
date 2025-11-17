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

    // Add block prefix (scanned_blocks) and write to output shifted by 1
    if (i < d_data.size())
    {
        d_output[i + 1] = exclusive + d_data[i] + d_scanned_blocks[bid];
    }
}

void exclusive_scan_byhand(rmm::device_vector<int> &d_data, rmm::device_vector<int> &d_output)
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

    // Ensure output[0] == 0
    int zero = 0;
    cudaMemcpy(thrust::raw_pointer_cast(d_output.data()), &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Launch propagate with shared memory for block-local scan
    propagate<<<block_count, block_size, block_size * sizeof(int)>>>(data_span, scanned_blocks_span, d_output_span);
}
