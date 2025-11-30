#include "histogram.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <rmm/device_vector.hpp>

#include "scan.cuh"

__global__ void histogram_kernel(cuda::std::span<int> d_data, cuda::std::span<int> d_hist)
{
    __shared__ unsigned int s_hist[256];

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    for (int b = tid; b < 256; b += block_threads)
        s_hist[b] = 0u;
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    const size_t n = (size_t)d_data.size();
    for (size_t i = idx; i < n; i += stride)
    {
        int v = d_data[i];
        if ((unsigned)v > 255u) {
            v = (v < 0) ? 0 : 255;
        }
        atomicAdd(&s_hist[(unsigned)v], 1u);
    }

    __syncthreads();

    for (int b = tid; b < 256; b += block_threads)
    {
        unsigned int val = s_hist[b];
        if (val != 0u)
        {
            atomicAdd(reinterpret_cast<unsigned int*>(&d_hist[b]), val);
        }
    }
}

__global__ void build_inclusive_cdf(cuda::std::span<int> d_exclusive, cuda::std::span<int> d_hist, cuda::std::span<int> d_cdf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_hist.size())
    {
        d_cdf[i] = d_exclusive[i] + d_hist[i];
    }
}

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
            mapped = i;
        }
        d_lut[i] = mapped;
    }
}

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
