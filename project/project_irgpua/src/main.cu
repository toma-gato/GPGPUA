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
    std::vector<Image> images_cpu(nb_images);
    std::vector<Image> images_gpu(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course

        Image original = pipeline.get_image(i);
        size_t elems = static_cast<size_t>(original.size());

        images_cpu[i] = original;
        images_gpu[i] = original;
    
        rmm::device_uvector<int> d_buf(elems, rmm::cuda_stream_default);
        cudaMemcpyAsync(d_buf.data(), images_gpu[i].buffer, elems * sizeof(int), 
                        cudaMemcpyHostToDevice, rmm::cuda_stream_default);
        fix_image_gpu_indus(d_buf);
        cudaMemcpyAsync(images_gpu[i].buffer, d_buf.data(), elems * sizeof(int), 
                        cudaMemcpyDeviceToHost, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);
        
        // VERSION CPU
        fix_image_cpu(images_cpu[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image_cpu = images_cpu[i];
        auto& image_gpu = images_gpu[i];
        const int image_size = image_cpu.width * image_cpu.height;
        
        image_cpu.to_sort.total = std::reduce(image_cpu.buffer, image_cpu.buffer + image_size, 0);
        image_gpu.to_sort.total = std::reduce(image_gpu.buffer, image_gpu.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images_cpu] () mutable
    {
        return images_cpu[n++].to_sort;
    });
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images_gpu] () mutable
    {
        return images_gpu[n++].to_sort;
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
        std::cout << "CPU Image #" << images_cpu[i].to_sort.id << " total : " << images_cpu[i].to_sort.total << std::endl;
        std::cout << "GPU Image #" << images_gpu[i].to_sort.id << " total : " << images_gpu[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "CPU_Image#" << images_cpu[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images_cpu[i].write(str);

        oss << "GPU_Image#" << images_gpu[i].to_sort.id << ".pgm";
        str = oss.str();
        images_gpu[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i) {
        free(images_cpu[i].buffer);
        free(images_gpu[i].buffer);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Temps total (CPU + GPU sync): %.3f ms\n", ms);

    return 0;
}
