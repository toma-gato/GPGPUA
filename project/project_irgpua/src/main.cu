#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu_indus.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#include <rmm/device_uvector.hpp>

#include <chrono>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    auto start = std::chrono::high_resolution_clock::now();

    #ifdef USE_GPU
        std::cout << "Using GPU..." << std::endl;
    #else
        std::cout << "Using CPU..." << std::endl;
    #endif

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
    //for (const auto& dir_entry : recursive_directory_iterator("/home/thomas.galateau/image_test"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #ifdef USE_GPU
    
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            images[i] = pipeline.get_image(i);
            size_t elems = static_cast<size_t>(images[i].size());
            
            rmm::device_uvector<int> d_buf(elems, rmm::cuda_stream_default);
            cudaMemcpyAsync(d_buf.data(), images[i].buffer, elems * sizeof(int), cudaMemcpyHostToDevice, rmm::cuda_stream_default);
            
            fix_image_gpu_indus(d_buf, rmm::cuda_stream_default);
            
            size_t new_elems = d_buf.size();
            cudaMemcpyAsync(images[i].buffer, d_buf.data(), new_elems * sizeof(int), cudaMemcpyDeviceToHost, rmm::cuda_stream_default);
            cudaStreamSynchronize(rmm::cuda_stream_default);
        }
    #else
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            images[i] = pipeline.get_image(i);
            
            fix_image_cpu(images[i]);
        }
    #endif

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #ifdef USE_GPU        std::vector<int> h_results(nb_images);
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            auto& image = images[i];
            const int image_size = image.width * image.height;
            
            int total = thrust::reduce(
                thrust::cuda::par.on(stream),
                image.buffer.begin(),
                image.buffer.end(),
                0,
                thrust::plus<int>()
            );
            
            images[i].to_sort.total = total;
        }
    #else
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            auto& image = images[i];
            const int image_size = image.width * image.height;
            
            image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
        }
    #endif

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i) {
        free(images[i].buffer);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Temps total: %.3f ms\n", ms);

    return 0;
}
