#!/bin/bash
# This script runs a focused Nsight Compute benchmark
# It collects only: execution time (launch stats), compute & memory throughput,
# and the GPU Speed-Of-Light (SOL) analysis sections.

BUILD_DIR="${BUILD_DIR:-"$(dirname "$0")/../build"}"
BYHAND_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu"}"
INDUS_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu_indus"}"
BYHAND_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${BYHAND_PROGRAM_NAME}")"}"
INDUS_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${INDUS_PROGRAM_NAME}")"}"
VERSION="${VERSION:-"1"}"
IMAGES_DIR="${IMAGES_DIR:-"${BUILD_DIR}/images_projet"}"

set -e

NCU_SECTIONS=(LaunchStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis SpeedOfLight)

section_args=()
for s in "${NCU_SECTIONS[@]}"; do
	section_args+=("--section" "$s")
done

# Benchmark byhand version
sudo ncu -o "bench_v${VERSION}-byhand" "${section_args[@]}" \
	-k "regex:.*(reduce|scan|propagate|histogram|histogram_kernel|build_inclusive_cdf|build_lut|apply_lut|histogram_equalize_byhand).*" \
	"$BYHAND_PROGRAM_PATH" -d "$IMAGES_DIR"

# Benchmark industrial version
sudo ncu -o "bench_v${VERSION}-industrial" "${section_args[@]}" \
	"$INDUS_PROGRAM_PATH" -d "$IMAGES_DIR"


