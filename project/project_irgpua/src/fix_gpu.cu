#include "fix_gpu.cuh"

#include "compact.cuh"

void fix_image_gpu(rmm::device_vector<int> &buffer)
{
    // #1 Compact
    rmm::device_vector<int> d_compact_result(buffer.size(), 0);
    compact_byhand(buffer, d_compact_result, -27);
    buffer = d_compact_result;
}
