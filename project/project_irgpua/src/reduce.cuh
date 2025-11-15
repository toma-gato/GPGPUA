#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void reduce_block(cuda::std::span<int> d_data, cuda::std::span<int> d_output);
__global__ void reduce_final(cuda::std::span<int> d_data, cuda::std::span<int> d_result);
int reduce(rmm::device_vector<int> d_data);

