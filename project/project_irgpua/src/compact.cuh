#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

#include "reduce.cuh"
#include "scan.cuh"

__global__ void predicate(cuda::std::span<int> d_input, cuda::std::span<int> d_predicate, int flag);
__global__ void scatter(cuda::std::span<int> d_input,
                        cuda::std::span<int> d_output,
                        cuda::std::span<int> d_predicate,
                        cuda::std::span<int> d_scanned_predicate);

template <size_t BLOCK_SIZE = 256>
int compact_byhand(rmm::device_vector<int> &input, rmm::device_vector<int> &output, int flag)
{
    rmm::device_vector<int> d_predicate(input.size());
    cuda::std::span<int> d_input_span(input.data().get(), input.size());
    cuda::std::span<int> d_predicate_span(d_predicate.data().get(), d_predicate.size());

    const size_t GRID_SIZE = (input.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    predicate<<<GRID_SIZE, BLOCK_SIZE, 0>>>(d_input_span, d_predicate_span, flag);
    cudaDeviceSynchronize();

    rmm::device_vector<int> d_scanned_predicate(input.size());
    exclusive_scan_byhand<BLOCK_SIZE>(d_predicate, d_scanned_predicate);

    cuda::std::span<int> d_output_span(output.data().get(), output.size());
    cuda::std::span<int> d_scanned_predicate_span(d_scanned_predicate.data().get(), d_scanned_predicate.size());
    scatter<<<GRID_SIZE, BLOCK_SIZE, 0>>>(d_input_span, d_output_span, d_scanned_predicate_span, d_predicate_span);
    cudaDeviceSynchronize();

    // Determine compacted size: last_scanned + last_predicate
    int last_pred = 0;
    int last_scan = 0;
    if (input.size() > 0)
    {
        // copy the last elements from device
        cudaMemcpy(&last_pred,
                   thrust::raw_pointer_cast(d_predicate.data()) + (input.size() - 1),
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(&last_scan,
                   thrust::raw_pointer_cast(d_scanned_predicate.data()) + (input.size() - 1),
                   sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    return last_scan + last_pred;
}
