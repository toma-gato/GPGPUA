#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void reduce_block(cuda::std::span<int> d_data, cuda::std::span<int> d_output);
__global__ void reduce_final(cuda::std::span<int> d_data, cuda::std::span<int> d_result);

template <size_t BLOCK_SIZE = 256>
int reduce_byhand(rmm::device_vector<int> &d_data)
{
    size_t num_elements = d_data.size();

    const size_t GRID_SIZE = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cuda::std::span<int> d_data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());

    rmm::device_vector<int> d_intermediate(GRID_SIZE, 0);
    cuda::std::span<int> d_intermediate_span(thrust::raw_pointer_cast(d_intermediate.data()), d_intermediate.size());

    rmm::device_vector<int> d_result(1, 0);
    cuda::std::span<int> d_result_span(thrust::raw_pointer_cast(d_result.data()), 1);

    reduce_block<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data_span, d_intermediate_span);
    cudaDeviceSynchronize();

    reduce_final<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_intermediate_span, d_result_span);
    cudaDeviceSynchronize();

    int result = 0;
    cudaMemcpy(&result, thrust::raw_pointer_cast(d_result.data()), sizeof(int), cudaMemcpyDeviceToHost);

    return result;
}
