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
void histogram_equalize_byhand(rmm::device_vector<int> &d_data)
{
    const int NUM_BINS = 256;
    size_t num_elements = d_data.size();

    // allocate histogram (initialized to 0)
    rmm::device_vector<int> d_hist(NUM_BINS, 0);
    cuda::std::span<int> data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());
    cuda::std::span<int> hist_span(thrust::raw_pointer_cast(d_hist.data()), d_hist.size());

    // launch histogram kernel
    // Unified block/grid sizing for all kernel launches in this function.
    // BLOCK_SIZE can be tuned; keep 256 as a sensible default.
    const int computed_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Use a reasonable grid even for large images (cap grid size)
    const int max_blocks = 1024;
    const size_t GRID_SIZE = computed_blocks > max_blocks ? max_blocks : computed_blocks;

    // Launch histogram kernel using unified BLOCK_SIZE and GRID_SIZE
    histogram_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(data_span, hist_span);
    cudaDeviceSynchronize();

    // compute exclusive scan (prefix sums) of histogram
    rmm::device_vector<int> d_hist_exclusive(NUM_BINS, 0);
    exclusive_scan_byhand<BLOCK_SIZE>(d_hist, d_hist_exclusive);

    // build inclusive cdf = exclusive + hist
    rmm::device_vector<int> d_cdf(NUM_BINS, 0);
    cuda::std::span<int> excl_span(thrust::raw_pointer_cast(d_hist_exclusive.data()), d_hist_exclusive.size());
    cuda::std::span<int> cdf_span(thrust::raw_pointer_cast(d_cdf.data()), d_cdf.size());
    // For bin-based kernels (NUM_BINS elements) derive thread/block layout from BLOCK_SIZE
    const int bin_threads = (NUM_BINS < BLOCK_SIZE) ? NUM_BINS : BLOCK_SIZE;
    const int bin_blocks = (NUM_BINS + bin_threads - 1) / bin_threads;
    build_inclusive_cdf<<<bin_blocks, bin_threads>>>(excl_span, hist_span, cdf_span);
    cudaDeviceSynchronize();

    // copy cdf to host to find cdf_min (first non-zero)
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

    // build LUT on device
    rmm::device_vector<int> d_lut(NUM_BINS, 0);
    cuda::std::span<int> lut_span(thrust::raw_pointer_cast(d_lut.data()), d_lut.size());
    build_lut<<<bin_blocks, bin_threads>>>(cdf_span, lut_span, static_cast<int>(num_elements), cdf_min);
    cudaDeviceSynchronize();

    // apply LUT
    cuda::std::span<int> lut_span2(thrust::raw_pointer_cast(d_lut.data()), d_lut.size());
    const int apply_threads = BLOCK_SIZE;
    const int apply_blocks = (num_elements + apply_threads - 1) / apply_threads;
    int apply_g = apply_blocks > max_blocks ? max_blocks : apply_blocks;
    apply_lut<<<apply_g, apply_threads>>>(data_span, lut_span2);
    cudaDeviceSynchronize();
}
