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
            // per-thread reusable resources
            thread_local cudaStream_t thread_stream = nullptr;
            thread_local bool stream_created = false;
            thread_local rmm::device_uvector<int> d_buf(0, rmm::cuda_stream_default);
            thread_local int* host_pinned = nullptr;
            thread_local size_t host_pinned_bytes = 0;
            thread_local size_t dev_buf_elems = 0;

            if (!stream_created) {
                if (cudaStreamCreate(&thread_stream) != cudaSuccess) {
                    std::cerr << "cudaStreamCreate failed, using default stream\n";
                    thread_stream = rmm::cuda_stream_default;
                }
                stream_created = true;
            }

            // get image and sizes
            images[i] = pipeline.get_image(i);
            size_t elems = static_cast<size_t>(images[i].size());
            size_t bytes = elems * sizeof(int);

            // ensure device buffer big enough (RMM pool makes resize cheap after first alloc)
            if (dev_buf_elems < elems) {
                d_buf.resize(elems, thread_stream);
                dev_buf_elems = elems;
            }

            // ensure host-pinned buffer big enough and reuse it
            if (host_pinned_bytes < bytes) {
                if (host_pinned) {
                    cudaFreeHost(host_pinned);
                    host_pinned = nullptr;
                }
                if (cudaMallocHost(reinterpret_cast<void**>(&host_pinned), bytes) != cudaSuccess) {
                    // fallback: heap buffer (slower)
                    host_pinned = reinterpret_cast<int*>(malloc(bytes));
                    if (!host_pinned) {
                        std::cerr << "host buffer allocation failed\n";
                        continue;
                    }
                }
                host_pinned_bytes = bytes;
            }

            // copy input into pinned staging buffer
            std::memcpy(host_pinned, images[i].buffer, bytes);

            // async H2D on this thread's stream
            CUDA_CHECK(cudaMemcpyAsync(d_buf.data(), host_pinned, bytes,
                                    cudaMemcpyHostToDevice, thread_stream));

            // call GPU pipeline (must use d_buf.stream() or the passed stream)
            fix_image_gpu_indus(d_buf);

            // async D2H on same stream into pinned host buffer
            CUDA_CHECK(cudaMemcpyAsync(host_pinned, d_buf.data(), bytes,
                                    cudaMemcpyDeviceToHost, thread_stream));

            // wait for the stream to finish this image (keeps correctness; can be removed for more advanced overlap)
            CUDA_CHECK(cudaStreamSynchronize(thread_stream));

            // copy pinned -> final CPU buffer
            std::memcpy(images[i].buffer, host_pinned, bytes);
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
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

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
