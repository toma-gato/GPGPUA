#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

template <typename T>
__global__
void kernel_scan_baseline(raft::device_span<T> buffer)
{
    for (int i = 1; i < buffer.size(); ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(rmm::device_uvector<int>& buffer)
{
	kernel_scan_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

inline __device__ int warp_reduce(int val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(~0, val, offset);
    }
    return val;
}

template <typename T, int BLOCK_SIZE>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;
    while(i < buffer.size()) {
        if (i + BLOCK_SIZE < buffer.size())
            sdata[tid] += buffer[i] + buffer[i + BLOCK_SIZE];
        else
            sdata[tid] += buffer[i];
        i += gridSize;
    }
    
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 1024) {
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 64) {
        if (tid < 32)
            sdata[tid] += sdata[tid + 32];
        __syncthreads();
    }

    if (tid < 32)
        sdata[tid] += warp_reduce(sdata[tid]);

    if (tid == 0) total[blockIdx.x] = sum;
}


template <typename T>
__global__
void kernel_your_scan(raft::device_span<T> buffer)
{
    // TODO
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < buffer.size(); i*=2) {
        T val = 0;
        if (tid >= i) {
            val = buffer[idx - i];
        }
        __syncthreads();

        buffer[idx] += val;
        __syncthreads();
    }
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    // TODO
    // rmm::device_uvector<int> tmp(2, buffer.stream());

    // kernel_your_reduce<int, 64><<<2, 64, 0, buffer.stream()>>>(
    //     raft::device_span<const int>(buffer.data(), buffer.size()),
    //     raft::device_span<int>(tmp.data(), 1));

	kernel_your_scan<int><<<1, 64, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}