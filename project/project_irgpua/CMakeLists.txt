cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

option(USE_GPU "Use GPU acceleration" ON)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_reduce)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

include_directories(src)

add_executable(main src/main.cu)

if(USE_GPU)
    target_compile_definitions(main PRIVATE USE_GPU)
    message(STATUS "Building with GPU support enabled")
else()
    message(STATUS "Building with CPU only")
endif()

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

# Create a library target containing the core sources (so tests can link against it)
add_library(project_core STATIC
        src/fix_cpu.cu
        src/fix_gpu.cu
        src/fix_gpu_indus.cu
        src/reduce.cu
        src/scan.cu
        src/compact.cu
)

target_include_directories(project_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(project_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(project_core PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)

# Link the core library into the main executable
target_link_libraries(main PUBLIC project_core)

# GoogleTest setup
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Add tests
add_executable(reduce_tests tests/reduce_tests.cu)
target_include_directories(reduce_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(reduce_tests PRIVATE gtest_main project_core)
set_target_properties(reduce_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(scan_tests tests/scan_tests.cu)
target_include_directories(scan_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(scan_tests PRIVATE gtest_main project_core)
set_target_properties(scan_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(compact_tests tests/compact_tests.cu)
target_include_directories(compact_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(compact_tests PRIVATE gtest_main project_core)
set_target_properties(compact_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(GoogleTest)
gtest_discover_tests(reduce_tests)
gtest_discover_tests(scan_tests)
gtest_discover_tests(compact_tests)
