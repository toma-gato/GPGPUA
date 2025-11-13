#include "fix_gpu_indus.cuh"

#include "../cuda_tools/cuda_error_checking.cuh"

#include <cub/cub.cuh>
#include <raft/core/device_span.hpp>

struct NotGarbage {
    __device__ bool operator()(int val) const { return val != -27; }
};

void remove_garbage(rmm::device_uvector<int> &buffer)
{
    size_t n = buffer.size();
    cudaStream_t stream = buffer.stream();
    
    rmm::device_uvector<int> temp(n, stream);
    rmm::device_uvector<int> n_selected(1, stream);
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(
        d_temp_storage, temp_storage_bytes,
        buffer.data(), temp.data(),
        n_selected.data(),
        n,
        NotGarbage(),
        stream
    );
    
    rmm::device_uvector<char> temp_storage(temp_storage_bytes, stream);
    cub::DeviceSelect::If(
        temp_storage.data(), temp_storage_bytes,
        buffer.data(), temp.data(),
        n_selected.data(),
        n,
        NotGarbage(),
        stream
    );
    
    int num_selected = n_selected.element(0, stream);
    printf("Number of garbage elements removed: %d, should be 19595 for Image #0\n", n - num_selected);
    
    cudaMemcpyAsync(buffer.data(), temp.data(), num_selected * sizeof(int), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    
    buffer.resize(num_selected, stream);
}

__global__ void apply_pattern_kernel_optimized(raft::device_span<int> data, size_t n)
{
    const int adjustments[4] = {1, -5, 3, -8};
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
    {
        data[idx] += adjustments[idx % 4];
    }
}

void apply_pattern_treatment(rmm::device_uvector<int> &buffer)
{
    size_t n = buffer.size();
    cudaStream_t stream = buffer.stream();
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    apply_pattern_kernel_optimized<<<num_blocks, threads_per_block, 0, stream>>>(
        raft::device_span<int>(buffer.data(), buffer.size()), n
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void fix_image_gpu_indus(rmm::device_uvector<int> &buffer)
{
    remove_garbage(buffer);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

    apply_pattern_treatment(buffer);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
