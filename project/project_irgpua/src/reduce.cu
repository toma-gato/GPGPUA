#include "reduce.cuh"

__global__ void reduce_block(cuda::std::span<int> d_data, cuda::std::span<int> d_output)
{
    extern __shared__ int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] = (i < d_data.size()) ? d_data[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_output[blockIdx.x] = s_data[0];
    }
}
