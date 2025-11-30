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

#include "fix_gpu.cuh"
#include "reduce.cuh"

#include <thrust/device_ptr.h>

#include <chrono>

int main(int argc, char *argv[])
{
    // -- Pipeline initialization

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "Using GPU..." << std::endl;
    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    std::string images_dir = "./images_projet";
    for (int ai = 1; ai < argc; ++ai)
    {
        std::string arg = argv[ai];
        const std::string prefix = "--images-dir=";
        if (arg.rfind(prefix, 0) == 0)
            images_dir = arg.substr(prefix.size());
        else if (arg == "-d" && ai + 1 < argc)
            images_dir = argv[++ai];
        else if (ai == 1)
            images_dir = arg;
    }

    for (const auto &dir_entry : recursive_directory_iterator(images_dir))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        size_t elems = static_cast<size_t>(images[i].size());

        // allocate device buffer and copy host -> device
        rmm::device_vector<int> d_buf(elems);
        cudaMemcpyAsync(d_buf.data().get(), images[i].buffer, elems * sizeof(int),
                        cudaMemcpyHostToDevice, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);

        fix_image_gpu(d_buf);

        // copy back the compacted result (may be smaller)
        size_t new_elems = d_buf.size();
        cudaMemcpyAsync(images[i].buffer, d_buf.data().get(), new_elems * sizeof(int),
                        cudaMemcpyDeviceToHost, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);

        // ensure host buffer keeps expected size: zero the tail if compacting removed pixels
        images[i].actual_size = static_cast<int>(new_elems);
        if (new_elems < elems)
            std::fill(images[i].buffer + new_elems, images[i].buffer + elems, 0);
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
        auto &image = images[i];
        const int image_size = image.width * image.height;

        rmm::device_vector<int> d_buf(image_size);
        cudaMemcpyAsync(d_buf.data().get(), images[i].buffer, image_size * sizeof(int),
                        cudaMemcpyHostToDevice, rmm::cuda_stream_default);
        cudaStreamSynchronize(rmm::cuda_stream_default);

        image.to_sort.total = reduce_byhand(d_buf);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images]() mutable
                  { return images[n++].to_sort; });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b)
              { return a.total < b.total; });

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
    for (int i = 0; i < nb_images; ++i)
    {
        free(images[i].buffer);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Temps total: %.3f ms\n", ms);

    return 0;
}
