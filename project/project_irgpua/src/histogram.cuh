#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

#include <thrust/host_vector.h>
#include "scan.cuh"

__global__ void histogram_kernel(cuda::std::span<int> d_data, cuda::std::span<int> d_hist);
__global__ void build_inclusive_cdf(cuda::std::span<int> d_exclusive, cuda::std::span<int> d_hist, cuda::std::span<int> d_cdf);
__global__ void build_lut(cuda::std::span<int> d_cdf, cuda::std::span<int> d_lut, int total_pixels, int cdf_min);
__global__ void apply_lut(cuda::std::span<int> d_data, cuda::std::span<int> d_lut);

template <size_t BLOCK_SIZE = 256>
void histogram_equalize_byhand(rmm::device_vector<int> &d_data, size_t num_elements)
{
    const unsigned NUM_BINS = 256;

    rmm::device_vector<int> d_hist(NUM_BINS, 0);
    cuda::std::span<int> data_span(thrust::raw_pointer_cast(d_data.data()), num_elements);
    cuda::std::span<int> hist_span(thrust::raw_pointer_cast(d_hist.data()), d_hist.size());

    // #1 Compute histogram
    const size_t computed_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr size_t MAX_GRID_SIZE = 1024;
    const size_t GRID_SIZE = computed_blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : computed_blocks;
    histogram_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(data_span, hist_span);

    // #2 Compute exclusive scan of histogram to get CDF
    rmm::device_vector<int> d_hist_exclusive(NUM_BINS, 0);
    exclusive_scan_byhand<BLOCK_SIZE>(d_hist, d_hist_exclusive);

    // #3 Build inclusive CDF
    rmm::device_vector<int> d_cdf(NUM_BINS, 0);
    cuda::std::span<int> excl_span(thrust::raw_pointer_cast(d_hist_exclusive.data()), d_hist_exclusive.size());
    cuda::std::span<int> cdf_span(thrust::raw_pointer_cast(d_cdf.data()), d_cdf.size());
    const int bin_threads = (NUM_BINS < BLOCK_SIZE) ? NUM_BINS : BLOCK_SIZE;
    const int bin_blocks = (NUM_BINS + bin_threads - 1) / bin_threads;
    build_inclusive_cdf<<<bin_blocks, bin_threads>>>(excl_span, hist_span, cdf_span);

    // Copy CDF to host to find first non-zero
    thrust::host_vector<int> h_cdf(NUM_BINS);
    cudaMemcpy(thrust::raw_pointer_cast(h_cdf.data()), thrust::raw_pointer_cast(d_cdf.data()), NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    int cdf_min = 0;
    for (int i = 0; i < NUM_BINS; ++i)
    {
        if (h_cdf[i] > 0)
        {
            cdf_min = h_cdf[i];
            break;
        }
    }

    // #4 Build LUT on device
    rmm::device_vector<int> d_lut(NUM_BINS, 0);
    cuda::std::span<int> lut_span(thrust::raw_pointer_cast(d_lut.data()), d_lut.size());
    build_lut<<<bin_blocks, bin_threads>>>(cdf_span, lut_span, static_cast<int>(num_elements), cdf_min);

    // #5 Apply LUT
    cuda::std::span<int> lut_span2(thrust::raw_pointer_cast(d_lut.data()), d_lut.size());
    const int apply_threads = BLOCK_SIZE;
    const int apply_blocks = (num_elements + apply_threads - 1) / apply_threads;
    int apply_g = apply_blocks > MAX_GRID_SIZE ? MAX_GRID_SIZE : apply_blocks;
    apply_lut<<<apply_g, apply_threads>>>(data_span, lut_span2);
}
