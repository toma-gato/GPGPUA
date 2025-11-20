#pragma once

#include <cuda/std/span>
#include <rmm/device_vector.hpp>

// Compute histogram (256 bins) on device and perform histogram equalization in-place
// on the provided `d_data` device vector (pixel values expected in [0,255]).
void histogram_equalize_byhand(rmm::device_vector<int> &d_data);
