#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void histogram_kernel(cuda::std::span<int> d_data, cuda::std::span<int> d_hist);
__global__ void build_inclusive_cdf(cuda::std::span<int> d_exclusive, cuda::std::span<int> d_hist, cuda::std::span<int> d_cdf); 
__global__ void build_lut(cuda::std::span<int> d_cdf, cuda::std::span<int> d_lut, int total_pixels, int cdf_min);
__global__ void apply_lut(cuda::std::span<int> d_data, cuda::std::span<int> d_lut);
void histogram_equalize_byhand(rmm::device_vector<int> &d_data);
