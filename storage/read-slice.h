#pragma once

#include <highfive/H5DataSet.hpp>

#include "HalideBuffer.h"

namespace storage {

using slice_t = Halide::Runtime::Buffer<const uint16_t, 2>;
using cx_fcube_t = Halide::Runtime::Buffer<const float, 4>;

slice_t readSlice(const HighFive::DataSet& dataset, size_t well_id, size_t z, size_t width,
                  size_t height);

cx_fcube_t readQPILayers(const HighFive::DataSet& dataset, size_t well_id, size_t width,
                         size_t height, size_t n_layers = 4);

}  // namespace storage