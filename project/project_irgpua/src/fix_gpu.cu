#include "fix_gpu.cuh"

#include "compact.cuh"
#include "histogram.cuh"

#include <thrust/copy.h>

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
    rmm::device_vector<int> d_compact_result(buffer.size());
    int compacted_size = compact_byhand<256>(buffer, d_compact_result, -27);

    // #2 Apply pattern
    constexpr size_t BLOCK_SIZE = 256;
    const size_t GRID_SIZE = (compacted_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_pattern_kernel_optimized<<<GRID_SIZE, BLOCK_SIZE>>>(cuda::std::span<int>(d_compact_result.data().get(), compacted_size));

    // #3 Histogram equalization (in-place)
    histogram_equalize_byhand(d_compact_result, static_cast<size_t>(compacted_size));

    buffer = d_compact_result;
}
