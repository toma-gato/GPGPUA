#include "fix_gpu_indus.cuh"

#include "../cuda_tools/cuda_error_checking.cuh"

#include <cub/cub.cuh>
#include <raft/core/device_span.hpp>

struct NotGarbage {
    __device__ bool operator()(int val) const { return val != -27; }
};

void remove_garbage(raft::device_span<int> buffer, cudaStream_t stream)
{
    size_t n = buffer.size();
    rmm::device_uvector<int> temp(n, stream);
    rmm::device_uvector<int> n_selected(1, stream);
    
    // Première passe : calculer la taille du stockage temporaire
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
    
    // Allouer le stockage temporaire
    rmm::device_uvector<char> temp_storage(temp_storage_bytes, stream);
    
    // Deuxième passe : effectuer la sélection
    cub::DeviceSelect::If(
        temp_storage.data(), temp_storage_bytes,
        buffer.data(), temp.data(),
        n_selected.data(),
        n,
        NotGarbage(),
        stream
    );
    
    // Obtenir le nombre d'éléments sélectionnés
    int num_selected = n_selected.element(0, stream);
    printf("Number of garbage elements removed: %d\n", n - num_selected);
    
    // Copier SEULEMENT les éléments valides
    cudaMemcpyAsync(buffer.data(), temp.data(), num_selected * sizeof(int), 
                    cudaMemcpyDeviceToDevice, stream);
}

void fix_image_gpu_indus(rmm::device_uvector<int> &buffer)
{
    size_t original_size = buffer.size();
    remove_garbage(raft::device_span<int>(buffer.data(), buffer.size()), buffer.stream());
    
    // Note : Vous devrez redimensionner le buffer ou retourner la nouvelle taille
    // buffer.resize(nouvelle_taille, buffer.stream()); 
    
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
