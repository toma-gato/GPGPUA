#!/bin/bash

BUILD_DIR="${BUILD_DIR:-"$(dirname "$0")/../build"}"
BYHAND_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu"}"
INDUS_PROGRAM_NAME="${PROGRAM_NAME:-"main_gpu_indus"}"
BYHAND_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${BYHAND_PROGRAM_NAME}")"}"
INDUS_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/${INDUS_PROGRAM_NAME}")"}"
CPU_PROGRAM_PATH="${PROGRAM_PATH:-"$(realpath "${BUILD_DIR}/main")"}"
IMAGES_DIR="${IMAGES_DIR:-"${BUILD_DIR}/images_projet"}"
CPU_OUTPUT="${CPU_OUTPUT:-"checksum_cpu.out"}"
BYHAND_OUTPUT="${BYHAND_OUTPUT:-"checksum_byhand.out"}"
INDUS_OUTPUT="${INDUS_OUTPUT:-"checksum_indus.out"}"

$CPU_PROGRAM_PATH -d "$IMAGES_DIR" > "$CPU_OUTPUT"
$BYHAND_PROGRAM_PATH -d "$IMAGES_DIR" > "$BYHAND_OUTPUT"
$INDUS_PROGRAM_PATH -d "$IMAGES_DIR" > "$INDUS_OUTPUT"
diff "$CPU_OUTPUT" "$BYHAND_OUTPUT" > byhand_vs_cpu.diff
diff "$CPU_OUTPUT" "$INDUS_OUTPUT" > indus_vs_cpu.diff
exit 0
