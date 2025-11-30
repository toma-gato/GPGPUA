#!/bin/bash
# This script runs a focused Nsight System profiling

BUILD_DIR="${BUILD_DIR:-"$(dirname "$0")/../build"}"
BYHAND_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu"}"
INDUS_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu_indus"}"
BYHAND_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${BYHAND_PROGRAM_NAME}")"}"
INDUS_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${INDUS_PROGRAM_NAME}")"}"
VERSION="${VERSION:-"1"}"
IMAGES_DIR="${IMAGES_DIR:-"$(realpath "${BUILD_DIR}/images_projet")"}"

set -e
nsys profile -o "profile_v${VERSION}-byhand" "$BYHAND_PROGRAM_PATH" -d "$IMAGES_DIR"
nsys profile -o "profile_v${VERSION}-industrial" "$INDUS_PROGRAM_PATH" -d "$IMAGES_DIR"
