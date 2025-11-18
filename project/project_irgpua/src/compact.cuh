#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void predicate(cuda::std::span<int> d_input, cuda::std::span<int> d_predicate, int flag);
__global__ void scatter(cuda::std::span<int> d_input,
                        cuda::std::span<int> d_output,
                        cuda::std::span<int> d_predicate,
                        cuda::std::span<int> d_scanned_predicate);
void compact_byhand(rmm::device_vector<int> &input, rmm::device_vector<int> &output, int flag);
