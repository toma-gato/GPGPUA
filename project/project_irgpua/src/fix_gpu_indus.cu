#include "fix_gpu_indus.cuh"

#include <cub/cub.cuh>
#include <raft/core/device_span.hpp>

void remove_garbage(raft::device_span<int> buffer, cudaStream_t stream)
{
    const int garbage_val = -27;
    size_t n = buffer.size();

    rmm::device_uvector<int> temp(n, stream);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Flagged(
        d_temp_storage,
        temp_storage_bytes,
        buffer.data(),
        temp.data(),
        nullptr,
        n,
        [=] __device__ (int val) { return val != garbage_val; }
    );

    rmm:device_uvector<char> cub_storage(temp_storage_bytes, stream);
    d_temp_storage = cub_storage.data();

    rmm::device_scalar<int> n_selected(0, stream);
    cub::DeviceSelect::Flagged(
        d_temp_storage,
        temp_storage_bytes,
        buffer.data(),
        temp.data(),
        n_selected.data(),
        n,
        [=] __device__ (int val) { return val != garbage_val; }
    );

    cudaMemcpyAsync(buffer.data(), temp.data(), n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
}

void fix_image_gpu_indus(rmm::device_uvector<int> &buffer)
{
    remove_garbage(raft::device_span<int>(buffer.data(), buffer.size()), buffer.stream());
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
