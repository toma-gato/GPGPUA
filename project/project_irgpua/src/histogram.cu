#include "histogram.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <rmm/device_vector.hpp>

#include "scan.cuh"

// Simple histogram kernel: each thread processes a stride of elements
__global__ void histogram_kernel(cuda::std::span<int> d_data, cuda::std::span<int> d_hist)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < (size_t)d_data.size(); i += stride)
    {
        int v = d_data[i];
        if (v < 0)
            v = 0;
        if (v > 255)
            v = 255;
        atomicAdd(&d_hist[v], 1);
    }
}

// Build inclusive CDF: cdf[i] = exclusive[i] + hist[i]
__global__ void build_inclusive_cdf(cuda::std::span<int> d_exclusive, cuda::std::span<int> d_hist, cuda::std::span<int> d_cdf)
{
    int i = threadIdx.x;
    if (i < d_hist.size())
    {
        d_cdf[i] = d_exclusive[i] + d_hist[i];
    }
}

// Build LUT from inclusive cdf, using cdf_min and total pixels N
__global__ void build_lut(cuda::std::span<int> d_cdf, cuda::std::span<int> d_lut, int total_pixels, int cdf_min)
{
    int i = threadIdx.x;
    if (i < d_cdf.size())
    {
        int cdf = d_cdf[i];
        int mapped = 0;
        int denom = total_pixels - cdf_min;
        if (denom > 0)
        {
            float val = float(cdf - cdf_min) / float(denom) * 255.0f;
            int r = static_cast<int>(val + 0.5f);
            if (r < 0)
                r = 0;
            if (r > 255)
                r = 255;
            mapped = r;
        }
        else
        {
            // all pixels identical -> leave values unchanged
            mapped = i;
        }
        d_lut[i] = mapped;
    }
}

// Apply LUT to image in-place
__global__ void apply_lut(cuda::std::span<int> d_data, cuda::std::span<int> d_lut)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < (size_t)d_data.size(); i += stride)
    {
        int v = d_data[i];
        if (v < 0)
            v = 0;
        if (v > 255)
            v = 255;
        d_data[i] = d_lut[v];
    }
}

void histogram_equalize_byhand(rmm::device_vector<int> &d_data)
{
    const int NUM_BINS = 256;
    size_t num_elements = d_data.size();
    if (num_elements == 0)
        return;

    // allocate histogram (initialized to 0)
    rmm::device_vector<int> d_hist(NUM_BINS, 0);
    cuda::std::span<int> data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());
    cuda::std::span<int> hist_span(thrust::raw_pointer_cast(d_hist.data()), d_hist.size());

    // launch histogram kernel
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    // Use a reasonable grid even for large images (cap grid size)
    const int max_blocks = 1024;
    int g = blocks > max_blocks ? max_blocks : blocks;
    histogram_kernel<<<g, threads>>>(data_span, hist_span);
    cudaDeviceSynchronize();

    // compute exclusive scan (prefix sums) of histogram
    rmm::device_vector<int> d_hist_exclusive(NUM_BINS, 0);
    exclusive_scan_byhand(d_hist, d_hist_exclusive);

    // build inclusive cdf = exclusive + hist
    rmm::device_vector<int> d_cdf(NUM_BINS, 0);
    cuda::std::span<int> excl_span(thrust::raw_pointer_cast(d_hist_exclusive.data()), d_hist_exclusive.size());
    cuda::std::span<int> cdf_span(thrust::raw_pointer_cast(d_cdf.data()), d_cdf.size());
    build_inclusive_cdf<<<1, NUM_BINS>>>(excl_span, hist_span, cdf_span);
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
    build_lut<<<1, NUM_BINS>>>(cdf_span, lut_span, static_cast<int>(num_elements), cdf_min);
    cudaDeviceSynchronize();

    // apply LUT
    cuda::std::span<int> lut_span2(thrust::raw_pointer_cast(d_lut.data()), d_lut.size());
    const int apply_threads = 256;
    const int apply_blocks = (num_elements + apply_threads - 1) / apply_threads;
    int apply_g = apply_blocks > max_blocks ? max_blocks : apply_blocks;
    apply_lut<<<apply_g, apply_threads>>>(data_span, lut_span2);
    cudaDeviceSynchronize();
}
