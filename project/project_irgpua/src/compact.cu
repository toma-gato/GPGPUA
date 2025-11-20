#include "compact.cuh"

#include "scan.cuh"
#include "reduce.cuh"

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

void compact_byhand(rmm::device_vector<int> &input, rmm::device_vector<int> &output, int flag)
{
    rmm::device_vector<int> d_predicate(input.size());
    cuda::std::span<int> d_input_span(input.data().get(), input.size());
    cuda::std::span<int> d_predicate_span(d_predicate.data().get(), d_predicate.size());

    size_t block_size = 256;
    size_t block_count = (input.size() + block_size - 1) / block_size;
    predicate<<<block_count, block_size, 0>>>(d_input_span, d_predicate_span, flag);
    cudaDeviceSynchronize();

    rmm::device_vector<int> d_scanned_predicate(input.size());
    exclusive_scan_byhand(d_predicate, d_scanned_predicate);

    cuda::std::span<int> d_output_span(output.data().get(), output.size());
    cuda::std::span<int> d_scanned_predicate_span(d_scanned_predicate.data().get(), d_scanned_predicate.size());
    scatter<<<block_count, block_size, 0>>>(d_input_span, d_output_span, d_scanned_predicate_span, d_predicate_span);
    cudaDeviceSynchronize();
}
