#pragma once

#include <rmm/device_uvector.hpp>

void fix_image_gpu_indus(rmm::device_uvector<int>& buffer, cudaStream_t stream);
