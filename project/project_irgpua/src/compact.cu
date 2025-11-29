#include "compact.cuh"


__global__ void predicate(cuda::std::span<int> d_input, cuda::std::span<int> d_predicate, int flag)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_input.size())
    {
        d_predicate[idx] = (d_input[idx] != flag) ? 1 : 0;
    }
}

__global__ void scatter(cuda::std::span<int> d_input, cuda::std::span<int> d_output, cuda::std::span<int> d_scanned_predicate, cuda::std::span<int> d_predicate)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_input.size())
    {
        if (d_predicate[idx] == 1)
        {
            int output_idx = d_scanned_predicate[idx];
            d_output[output_idx] = d_input[idx];
        }
    }
}
