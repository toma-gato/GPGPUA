#include "fix_gpu_indus.cuh"

#include "../cuda_tools/cuda_error_checking.cuh"

#include <cub/cub.cuh>
#include <raft/core/device_span.hpp>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

struct NotGarbage {
    __device__ bool operator()(int val) const { return val != -27; }
};

void remove_garbage(rmm::device_uvector<int> &buffer, cudaStream_t stream)
{
    size_t n = buffer.size();
    
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
    
    int num_selected;
    
    cudaMemcpyAsync(&num_selected, n_selected.data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaMemcpyAsync(buffer.data(), temp.data(), num_selected * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    buffer.resize(num_selected, stream);

}

// __global__ void apply_pattern_kernel_optimized(raft::device_span<int> data, size_t n)
// {
//     const int adjustments[4] = {1, -5, 3, -8};
    
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < n)
//     {
//         data[idx] += adjustments[idx % 4];
//     }
// }

// void apply_pattern_treatment(rmm::device_uvector<int> &buffer, cudaStream_t stream)
// {
//     size_t n = buffer.size();
    
//     int threads_per_block = 256;
//     int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
//     apply_pattern_kernel_optimized<<<num_blocks, threads_per_block, 0, stream>>>(
//         raft::device_span<int>(buffer.data(), buffer.size()), n
//     );
    
//     CUDA_CHECK_ERROR(cudaGetLastError());
// }

struct pattern_functor
{
    __host__ __device__
    int operator()(int value, size_t idx) const
    {
        const int adjustments[4] = {1, -5, 3, -8};
        return value + adjustments[idx & 3];
    }
};

void apply_pattern_treatment(rmm::device_uvector<int> &buffer, cudaStream_t stream)
{
    thrust::transform(
        thrust::cuda::par.on(stream),
        buffer.begin(),
        buffer.end(),
        thrust::make_counting_iterator<size_t>(0),
        buffer.begin(),
        pattern_functor()
    );
}

__global__ void apply_histogram_equalization_kernel(
    int* data, 
    size_t n, 
    const int* cumulative_histo,
    int cdf_min,
    int total_pixels)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n)
    {
        int pixel = data[idx];
        pixel = min(max(pixel, 0), 255);
        float normalized = ((cumulative_histo[pixel] - cdf_min) / 
                           static_cast<float>(total_pixels - cdf_min)) * 255.0f;
        data[idx] = static_cast<int>(roundf(normalized));
    }
}

// Kernel optimisé pour trouver le minimum
__global__ void find_cdf_min_kernel(const int* cumulative_histo, const int* histogram, int* cdf_min, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size && histogram[idx] > 0)
    {
        atomicMin(cdf_min, cumulative_histo[idx]);
    }
}

void histogram_equalization_gpu_async(rmm::device_uvector<int> &buffer, cudaStream_t stream)
{
    size_t n = buffer.size();
    const int num_bins = 256;
    
    // Étape 1 : Histogramme
    rmm::device_uvector<int> histogram(num_bins, stream);
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes,
        buffer.data(), histogram.data(),
        num_bins + 1, 0, 256, n, stream
    );
    
    rmm::device_uvector<char> temp_storage_histo(temp_storage_bytes, stream);
    cub::DeviceHistogram::HistogramEven(
        temp_storage_histo.data(), temp_storage_bytes,
        buffer.data(), histogram.data(),
        num_bins + 1, 0, 256, n, stream
    );
    
    // Étape 2 : Scan inclusif
    rmm::device_uvector<int> cumulative_histo(num_bins, stream);
    
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes,
        histogram.data(), cumulative_histo.data(),
        num_bins, stream
    );
    
    rmm::device_uvector<char> temp_storage_scan(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSum(
        temp_storage_scan.data(), temp_storage_bytes,
        histogram.data(), cumulative_histo.data(),
        num_bins, stream
    );
    
    // Étape 3 : Trouver cdf_min de manière plus efficace
    rmm::device_uvector<int> d_cdf_min(1, stream);
    int init_value = INT_MAX;
    cudaMemcpyAsync(d_cdf_min.data(), &init_value, sizeof(int), 
                    cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (num_bins + threads - 1) / threads;
    find_cdf_min_kernel<<<blocks, threads, 0, stream>>>(cumulative_histo.data(), histogram.data(), d_cdf_min.data(), num_bins);
    
    int cdf_min;
    cudaMemcpyAsync(&cdf_min, d_cdf_min.data(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    blocks = (n + threads - 1) / threads;
    apply_histogram_equalization_kernel<<<blocks, threads, 0, stream>>>(
        buffer.data(), n, cumulative_histo.data(), cdf_min, static_cast<int>(n)
    );
    
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void fix_image_gpu_indus(rmm::device_uvector<int> &buffer, cudaStream_t stream)
{
    remove_garbage_async(buffer, stream);
    apply_pattern_treatment_async(buffer, stream);
    histogram_equalization_gpu_async(buffer, stream);    
}