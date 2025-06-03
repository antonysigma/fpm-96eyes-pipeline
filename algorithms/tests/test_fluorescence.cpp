#include <armadillo>
#include <chrono>
#include <iostream>
#include <string>

#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"

// We want to continue to use our Halide::Buffer with AOT-compiled
// code, so we explicitly include it. It's a header-only class, and
// doesn't require libHalide.
#include "HalideBuffer.h"
#include "plls.h"
#include "raw2bgr.h"
#include "read-slice.h"

namespace {
using namespace HighFive;
using Halide::Runtime::Buffer;

enum channel_RGB { dapi = 2, egfp = 1, txred = 0 };

constexpr size_t width = 2592;
constexpr size_t height = 1944;
constexpr int well_id = 7;
constexpr int wsize = 96;
constexpr size_t zsize = 11;

using storage::slice_t;

arma::Cube<uint8_t>
decodeImage(slice_t egfp_buffer, slice_t txred_buffer) {
    arma::Cube<uint8_t> normalized_image(width, height, 3);

    // No DAPI (blue) channel

    // Retrieve focussed image at EGFP
    Buffer<uint8_t> output(normalized_image.memptr(), width, height, 3);

    // Goes through image processing pipeline
    // TODO(Antony): Output the interleaved image, not planar.
    egfp_buffer.set_host_dirty();
    txred_buffer.set_host_dirty();
    const auto error = raw2bgr(egfp_buffer, txred_buffer, output);
    assert(!error && "Halide error.");

    output.copy_to_host();
    return normalized_image;
}

}  // namespace

int
main() {
    // Load data
    constexpr char filename[]{HDF5_FILE_PATH};
    auto file = File(filename, File::ReadOnly);

    using namespace arma;

    Col<size_t>::fixed<2> plane_id;
    {
        arma::Mat<uint8_t> autofocus_plane(2, wsize);
        std::cout << "Reading pre-computed autofocus values..." << std::endl;
        file.getDataSet("autofocus_plane").read(autofocus_plane.memptr());
        plane_id(0) = autofocus_plane(0, well_id);
    }

    // Repeat for txred, with focal shift
    constexpr auto focal_shift = 2;
    plane_id(1) = std::min(plane_id(0) + focal_shift, zsize - 1);
    std::cout << "Saving focused image at z[egfp] = " << plane_id(0)
              << "; z[txred] = " << plane_id(1) << "..." << std::endl;

    using storage::readSlice;
    auto dataset = file.getDataSet("fluorescence");
    std::cout << "Reading EGFP slice..." << std::endl;
    const auto egfp = readSlice(dataset, well_id, plane_id(0), width, height);

    std::cout << "Reading TXRED slice..." << std::endl;
    const auto txred = readSlice(dataset, well_id, plane_id(1), width, height);

    // Save focal plane to local folder
    {
        const auto normalized_image = decodeImage(egfp, txred);

        const std::string output_filename("fluorescence_channel.ppm");
        normalized_image.save(output_filename, arma::ppm_binary);
        std::cout << "Saved to " << output_filename << std::endl;
    }

#ifdef RUN_BENCHMARK
    {
        std::cout << "Running benchmark of " << wsize << " channels, " << zsize
                  << " focal planes:" << std::endl;

        using namespace std::chrono;

        auto t0 = high_resolution_clock::now();
        for (size_t w = 0; w < wsize; ++w) {
            fluorescence.select({z, w, 0, 0}, {zsize, 1, height, width})
                .read((uint16_t****)image.memptr());

            plls(input, output);
            auto plane_id = arma::index_max(rolloff);
        }

        auto diff = duration_cast<milliseconds>(high_resolution_clock::now() - t0);
        float toc = diff.count();
        std::cout << toc << "ms enlapsed, " << (toc / wsize) << "ms per compute, "
                  << (toc / wsize / zsize) << "ms per plane" << std::endl;
    }
#endif

    return 0;
}
