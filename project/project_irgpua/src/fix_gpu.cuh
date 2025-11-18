#pragma once

#include <rmm/device_vector.hpp>

void fix_image_gpu(rmm::device_vector<int> &buffer);
