#!/bin/bash
# This script runs a focused Nsight Compute benchmark
# It collects only: execution time (launch stats), compute & memory throughput,
# and the GPU Speed-Of-Light (SOL) analysis sections.

BUILD_DIR="${BUILD_DIR:-"$(dirname "$0")/../build"}"
PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu"}"
PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${PROGRAM_NAME}")"}"
VERSION="${VERSION:-"1"}"

set -e

cd "$BUILD_DIR"

NCU_SECTIONS=(LaunchStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis SpeedOfLight)

section_args=()
for s in "${NCU_SECTIONS[@]}"; do
	section_args+=("--section" "$s")
done

sudo ncu -o "bench_v${VERSION}" "${section_args[@]}" \
	-k "regex:.*(regex|scan|propagate|histogram|histogram_kernel|build_inclusive_cdf|build_lut|apply_lut|histogram_equalize_byhand).*" \
	"$PROGRAM_PATH"

cd -



