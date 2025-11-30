# IRGPUA PROJECT

# Group members
- Thomas Galateau
- Nathan Sue
- Théo Hénon

# Quick start
To compile the project, run:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

There are three executables in the `build` directory:
- `main` which is the CPU implementation
- `main_gpu` which is the GPU byhand implementation
- `main_gpu_indus` which is the industrial GPU implementation

To run the executables, use:
```bash
./build/main -d ./images
./build/main_gpu -d ./images
./build/main_gpu_indus -d ./images
```
You may replace `./images` with the path to your own images directory.

The executables will write output images in the current directory.

# Tests
There are tests located in the `tests` directory, for byhand GPU implementation, especially for the reduce, scan & compact implementations.

To run the tests, use:
```bash
cd build
ctest --output-on-failure -j "$(nproc)"
```

There is also a script to compare output between CPU, byhand GPU and industrial GPU implementations:
```bash
IMAGES_DIR=./images ./tests/test_diff.sh
```

You may replace `./images` with the path to your own images directory.

This will produce 5 files :
- `checksum_cpu.out` : output of the CPU implementation
- `checksum_byhand.out` : output of the byhand GPU implementation
- `checksum_indus.out` : output of the industrial GPU implementation
- `byhand_vs_cpu.diff` : diff between CPU and byhand GPU outputs
- `indus_vs_cpu.diff` : diff between CPU and industrial GPU outputs

# Benchmark & Profiling
There are two scripts to launch benchmarks and profiling in `tests` directory:
- `benchmark_ncu.sh` to benchmark and profile both byhand and industrial GPU implementation using NVIDIA Nsight Compute.
To run the benchmark script, use:
```bash
IMAGES_DIR=./images VERSION=1 ./tests/benchmark_ncu.sh
```

You may replace `./images` with the path to your own images directory, and `VERSION` with the version of your benchmark reports (e.g., `1.0`, `1.1`, etc.)

This will produce two report files:
- `bench_v${VERSION}-byhand.ncu-rep` for byhand GPU implementation
- `bench_v${VERSION}-industrial.ncu-rep` for industrial GPU implementation

Those reports contains detailed performance on those metrics :
- **LaunchStats** 
- **ComputeWorkloadAnalysis** 
- **MemoryWorkloadAnalysis** 
- **SpeedOfLight**

- `profile_nsys.sh` to profile both byhand and industrial GPU implementation using NVIDIA Nsight Systems.
To run the profiling script, use:
```bash
IMAGES_DIR=./images VERSION=1 ./tests/profile_nsys.sh
```

You may replace `./images` with the path to your own images directory, and `VERSION` with the version of your benchmark reports (e.g., `1.0`, `1.1`, etc.)

This will produce two report files:
- `profile_v${VERSION}-byhand.nsys-rep` for byhand GPU implementation
- `profile_v${VERSION}-industrial.nsys-rep` for industrial GPU implementation
