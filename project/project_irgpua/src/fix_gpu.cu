#include "fix_gpu.cuh"

#include "compact.cuh"
#include "histogram.cuh"

__global__ void apply_pattern_kernel_optimized(cuda::std::span<int> data)
{
    const int adjustments[4] = {1, -5, 3, -8};

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < data.size())
    {
        data[idx] += adjustments[idx % 4];
    }
}

void fix_image_gpu(rmm::device_vector<int> &buffer)
{
    // #1 Compact
    rmm::device_vector<int> d_compact_result = buffer;
    compact_byhand(buffer, d_compact_result, -27);
    buffer = d_compact_result;

    // #2 Apply pattern
    const int threads_per_block = 256;
    const int blocks = (buffer.size() + threads_per_block - 1) / threads_per_block;
    apply_pattern_kernel_optimized<<<blocks, threads_per_block>>>(cuda::std::span<int>(thrust::raw_pointer_cast(buffer.data()), buffer.size()));
    cudaDeviceSynchronize();

    // #3 Histogram equalization (in-place)
    histogram_equalize_byhand(buffer);
    cudaDeviceSynchronize();
}
