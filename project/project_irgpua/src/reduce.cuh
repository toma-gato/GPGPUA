#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

__global__ void reduce_block(cuda::std::span<int> d_data, cuda::std::span<int> d_output);
__global__ void reduce_final(cuda::std::span<int> d_data, cuda::std::span<int> d_result);

template <size_t BLOCK_SIZE = 256>
__device__ void warp_reduce(cuda::std::span<int> sdata, int tid)
{
    if constexpr (BLOCK_SIZE >= 64)
    {
        sdata[tid] += sdata[tid + 32];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 32)
    {
        sdata[tid] += sdata[tid + 16];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 16)
    {
        sdata[tid] += sdata[tid + 8];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 8)
    {
        sdata[tid] += sdata[tid + 4];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 4)
    {
        sdata[tid] += sdata[tid + 2];
        __syncwarp();
    }
    if constexpr (BLOCK_SIZE >= 2)
    {
        sdata[tid] += sdata[tid + 1];
        __syncwarp();
    }
}

template <size_t BLOCK_SIZE = 256>
__global__ void reduce_kernel(cuda::std::span<int> buffer, cuda::std::span<int> total)
{
    extern __shared__ int sdata[];

    const unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (BLOCK_SIZE * 2) + tid;

    const size_t n = buffer.size();
    size_t gridSize = BLOCK_SIZE * 2 * gridDim.x;

    int sum = 0;
    while (i < n)
    {
        sum += buffer[i];

        if (i + BLOCK_SIZE < n)
            sum += buffer[i + BLOCK_SIZE];

        i += gridSize;
    }

    sdata[tid] = sum;
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        warp_reduce<BLOCK_SIZE>(cuda::std::span<int>(sdata, BLOCK_SIZE), tid);
    }

    if (tid == 0)
        total[blockIdx.x] = sdata[0];
}

template <size_t BLOCK_SIZE = 256>
int reduce_byhand(rmm::device_vector<int> &d_data)
{
    const size_t num_elements = d_data.size();
    constexpr size_t SHARED_MEMSIZE = BLOCK_SIZE * sizeof(int);

    size_t num_block = ((num_elements + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2));
    rmm::device_vector<int> tmp(num_block);

    reduce_kernel<BLOCK_SIZE><<<num_block, BLOCK_SIZE,
                                SHARED_MEMSIZE>>>(
        cuda::std::span<int>(thrust::raw_pointer_cast(d_data.data()), num_elements),
        cuda::std::span<int>(thrust::raw_pointer_cast(tmp.data()), num_block));

    while (num_block > 1)
    {
        size_t blocks = (num_block + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

        rmm::device_vector<int> tmp2(blocks);

        reduce_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE,
                                    SHARED_MEMSIZE>>>(
            cuda::std::span<int>(thrust::raw_pointer_cast(tmp.data()), num_block),
            cuda::std::span<int>(thrust::raw_pointer_cast(tmp2.data()), blocks));

        tmp = std::move(tmp2);
        num_block = blocks;
    }

    int result = 0;
    cudaMemcpy(&result,
               thrust::raw_pointer_cast(tmp.data()),
               sizeof(int),
               cudaMemcpyDeviceToHost);

    return result;
}
