#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void scan_block(cuda::std::span<int> reduced_blocks, cuda::std::span<int> d_output);
__global__ void propagate(cuda::std::span<int> d_data, cuda::std::span<int> d_scanned_blocks, cuda::std::span<int> d_output);
void exclusive_scan_byhand(rmm::device_vector<int> &d_data, rmm::device_vector<int> &d_output);
