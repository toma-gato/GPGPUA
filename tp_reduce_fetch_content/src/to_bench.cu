#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    for (int i = 0; i < buffer.size(); ++i)
        *total.data() += buffer[i];
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
	kernel_reduce_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__device__ void warp_reduce(raft::device_span<T> sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32]; __syncwarp();
    sdata[tid] += sdata[tid + 16]; __syncwarp();
    sdata[tid] += sdata[tid + 8]; __syncwarp();
    sdata[tid] += sdata[tid + 4]; __syncwarp();
    sdata[tid] += sdata[tid + 2]; __syncwarp();
    sdata[tid] += sdata[tid + 1]; __syncwarp();
}

template <typename T>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    // Help: odd size
    // When treating an odd size think about two things
    // 1. How could a thread sum two values and have the second (that we don't want for the odd case) not have any impact on the sum?
    // 2. Once 1. is achived, could we use a fixed even size while still achieving the same 

    // TODO
    // Your reduce code
    extern __shared__ T sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if (i < buffer.size()) {
        if (i + blockDim.x < buffer.size())
            sdata[tid] = buffer[i] + buffer[i + blockDim.x];
        else
            sdata[tid] = buffer[i];
    }
    else
        sdata[tid] = 0;
    
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    warp_reduce(sdata, tid);

    if (tid == 0) total[blockIdx.x] = sdata[0];
}

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    // Help: more than 1 thread block
    // When treating the 2 thread block case you need to create a temporary array
    // To do so use the following API : rmm::device_uvector<int> tmp(<SIZE>, buffer.stream())

    // Help: very large case
    // Using only 2 kernels, what is the biggest buffer size we can handle?

    // TODO fill in blocks, threads, and shared memory
    // Help: To properly compute the amount of block, use the following API: (<PROBLEM_SIZE> + <BLOCK_SIZE> - 1) / <BLOCK_SIZE>

    rmm::device_uvector<int> tmp(1024, buffer.stream());

    kernel_your_reduce<int><<<1024, 1024, 1024 * sizeof(int), buffer.stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(tmp.data(), 1));

    kernel_your_reduce<int><<<1, 1024, 1024 * sizeof(int), buffer.stream()>>>(
        raft::device_span<const int>(tmp.data(), tmp.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}