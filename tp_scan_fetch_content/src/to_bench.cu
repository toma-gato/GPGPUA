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

template <typename T>
__global__
void kernel_your_scan(raft::device_span<T> buffer)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 1; i < buffer.size(); i*=2) {
        if (tid >= i) {
            buffer[idx] += buffer[idx - i];
        }
        __syncthreads();
    }
}

void your_scan(rmm::device_uvector<int>& buffer)
{
    // TODO
    // ...

	kernel_your_scan<int><<<1, 64, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}