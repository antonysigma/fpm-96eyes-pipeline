#include "read-slice.h"

#include <highfive/H5File.hpp>

// Patch to encode std::complex<float> in HDF5 file.
#include "complex_float_support.hpp"

namespace storage {

slice_t
readSlice(const HighFive::DataSet& dataset, size_t well_id, size_t z, size_t width, size_t height) {
    Halide::Runtime::Buffer<uint16_t, 2> image(width, height);
    dataset.select({z, well_id, 0, 0}, {1, 1, height, width}).read(image.data());
    return image;
}

cx_fcube_t
readQPILayers(const HighFive::DataSet& dataset, size_t well_id, size_t width, size_t height,
              size_t n_layers) {
    Halide::Runtime::Buffer<float, 4> raw(2, width, height, n_layers);

    dataset.select({0, well_id, 0, 0}, {n_layers, 1, height, width})
        .read(reinterpret_cast<std::complex<float>*>(raw.data()));

    return raw;
}

}  // namespace storage