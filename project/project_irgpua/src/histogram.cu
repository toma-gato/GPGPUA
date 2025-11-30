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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_hist.size())
    {
        d_cdf[i] = d_exclusive[i] + d_hist[i];
    }
}

// Build LUT from inclusive cdf, using cdf_min and total pixels N
__global__ void build_lut(cuda::std::span<int> d_cdf, cuda::std::span<int> d_lut, int total_pixels, int cdf_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
