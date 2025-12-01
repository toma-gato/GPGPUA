#pragma once

#include <rmm/device_uvector.hpp>
#include <raft/core/device_span.hpp>

void remove_garbage(rmm::device_uvector<int> &buffer, cudaStream_t stream);

void apply_pattern_treatment(rmm::device_uvector<int> &buffer, cudaStream_t stream);

__global__ void apply_histogram_equalization_kernel(raft::device_span<int> data, size_t n, const raft::device_span<int> cumulative_histo, int cdf_min, int total_pixels)

__global__ void find_first_nonzero_kernel(const raft::device_span<int> histo, raft::device_span<int> result, int size);

void histogram_equalization_gpu(rmm::device_uvector<int> &buffer, cudaStream_t stream);

void fix_image_gpu_indus(rmm::device_uvector<int> &buffer, cudaStream_t stream);