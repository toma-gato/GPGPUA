#include "reduce.cuh"

/**
 * Performs a reduction (sum) on the input device vector using a 
 */


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

__global__ void reduce_final(cuda::std::span<int> d_data, cuda::std::span<int> d_result)
{
    extern __shared__ int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x;

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
        d_result[0] = s_data[0];
    }
}

int reduce(rmm::device_vector<int> d_data)
{
    size_t num_elements = d_data.size();

    size_t block_size = 256;
    size_t grid_size = (num_elements + block_size - 1) / block_size;

    cuda::std::span<int> d_data_span(thrust::raw_pointer_cast(d_data.data()), d_data.size());

    rmm::device_vector<int> d_intermediate(grid_size, 0);
    cuda::std::span<int> d_intermediate_span(thrust::raw_pointer_cast(d_intermediate.data()), d_intermediate.size());

    rmm::device_vector<int> d_result(1, 0);
    cuda::std::span<int> d_result_span(thrust::raw_pointer_cast(d_result.data()), 1);

    reduce_block<<<grid_size, block_size>>>(d_data_span, d_intermediate_span);
    reduce_final<<<1, block_size>>>(d_intermediate_span, d_result_span);

    int result = 0;
    cudaMemcpy(&result, thrust::raw_pointer_cast(d_result.data()), sizeof(int), cudaMemcpyDeviceToHost);

    return result;
}
