#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

#include "reduce.cuh"

__global__ void scan_block(cuda::std::span<int> reduced_blocks, cuda::std::span<int> d_output);
__global__ void propagate(cuda::std::span<int> d_data, cuda::std::span<int> d_scanned_blocks, cuda::std::span<int> d_output);

template <size_t BLOCK_SIZE = 256>
void exclusive_scan_byhand(rmm::device_vector<int> &d_data, rmm::device_vector<int> &d_output)
{
    const size_t num_elements = d_data.size();
    const size_t GRID_SIZE = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cuda::std::span<int> data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());
    rmm::device_vector<int> reduced_blocks(GRID_SIZE);
    cuda::std::span<int> reduced_blocks_span(thrust::raw_pointer_cast(reduced_blocks.data()), reduced_blocks.size());
    // Allocate dynamic shared memory for the block reduction (one int per thread)
    reduce_block<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(data_span, reduced_blocks_span);

    rmm::device_vector<int> d_scanned_blocks(GRID_SIZE);
    cuda::std::span<int> scanned_blocks_span(thrust::raw_pointer_cast(d_scanned_blocks.data()), d_scanned_blocks.size());
    constexpr size_t MAX_GRID_SIZE = 1024;
    if (GRID_SIZE <= MAX_GRID_SIZE)
    {
        scan_block<<<1, GRID_SIZE, GRID_SIZE * sizeof(int)>>>(reduced_blocks_span, scanned_blocks_span);
    }
    else
    {
        // Recursion
        rmm::device_vector<int> temp_scan(GRID_SIZE + 1);
        exclusive_scan_byhand<BLOCK_SIZE>(reduced_blocks, temp_scan);

        cudaMemcpy(thrust::raw_pointer_cast(d_scanned_blocks.data()),
                   thrust::raw_pointer_cast(temp_scan.data()),
                   GRID_SIZE * sizeof(int),
                   cudaMemcpyDeviceToDevice);
    }

    cuda::std::span<int> d_output_span(thrust::raw_pointer_cast(d_output.data()), d_output.size());

    propagate<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(data_span, scanned_blocks_span, d_output_span);
    cudaDeviceSynchronize();
}
