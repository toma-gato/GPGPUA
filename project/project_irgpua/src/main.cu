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
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


#include <chrono>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    auto start = std::chrono::high_resolution_clock::now();

    int expected_images_total[29] = {
        27805567,
        185010925,
        342970490,
        33055988,
        390348481,
        91297791,
        10825197,
        118842538,
        72434629,
        191735142,
        182802772,
        78632198,
        491605096,
        8109782,
        111786760,
        406461934,
        80671811,
        70004942,
        104275727,
        30603818,
        6496225,
        207334021,
        268424419,
        432916359,
        51973720,
        24489209,
        80124196,
        29256842,
        25803206,
        34550754
    };

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
        const int num_streams = std::min(nb_images, 16);
    
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; ++i)
        {
            cudaStreamCreate(&streams[i]);
        }
        
        std::vector<rmm::device_uvector<int>> d_buffers;
        d_buffers.reserve(nb_images);

        for (int i = 0; i < nb_images; ++i)
        {
            images[i] = pipeline.get_image(i);
            size_t elems = static_cast<size_t>(images[i].size());
            cudaStream_t stream = streams[i % num_streams];

            d_buffers.emplace_back(elems, stream);

            cudaMemcpyAsync(d_buffers[i].data(), images[i].buffer, elems * sizeof(int), cudaMemcpyHostToDevice, stream);
            
            fix_image_gpu_indus(d_buffers[i], stream);
        }
        for (auto& stream : streams)
        {
            cudaStreamSynchronize(stream);
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
    #ifdef USE_GPU
        rmm::device_uvector<int> d_results(nb_images, streams[0]);

        for (int i = 0; i < nb_images; ++i)
        {
            const int image_size = images[i].width * images[i].height;
            cudaStream_t stream = streams[i % num_streams];

            thrust::device_ptr<int> result_ptr = thrust::device_pointer_cast(d_results.data() + i);
            
            *result_ptr = thrust::reduce(
                thrust::cuda::par.on(stream),
                d_buffers[i].data(),
                d_buffers[i].data() + image_size,
                0,
                thrust::plus<int>()
            );            
        }
        for (auto& stream : streams)
        {
            cudaStreamSynchronize(stream);
        }

        std::vector<int> h_results(nb_images);
        cudaMemcpy(h_results.data(), d_results.data(), 
                   nb_images * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copier les images fixées
        for (int i = 0; i < nb_images; ++i)
        {
            size_t new_elems = d_buffers[i].size();
            cudaMemcpy(images[i].buffer, d_buffers[i].data(), 
                       new_elems * sizeof(int), cudaMemcpyDeviceToHost);
            images[i].to_sort.total = h_results[i];
        }

        for (auto& stream : streams)
        {
            cudaStreamDestroy(stream);
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
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << " and should be : " << expected_images_total[i] <<std::endl;
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
