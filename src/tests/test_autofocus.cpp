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

namespace {
using namespace HighFive;
using Halide::Runtime::Buffer;

constexpr int width = 2592;
constexpr int height = 1944;
constexpr int well_id = 7;
constexpr int wsize = 96;
constexpr int z = 0;
constexpr int zsize = 11;

// Read the entire focal stack, simulating the autofocus mechanism on the fly.
arma::Cube<uint16_t>
readFluorescenceZStack(File& file) {
    arma::Cube<uint16_t> z_stack(width, height, zsize);

    file.getDataSet("fluorescence")
        .select({z, well_id, 0, 0}, {zsize, 1, height, width})
        .read(z_stack.memptr());

    return z_stack;
}

arma::fcolvec
autofocus(const arma::Cube<uint16_t>& z_stack) {
    using namespace arma;

    // Memory map to buffer
    Buffer<const uint16_t> input(z_stack.memptr(), width, height, zsize);

    fcolvec rolloff(zsize);
    Buffer<float> output(rolloff.memptr(), zsize);

    // Compute the phase log-log slope metric
    input.set_host_dirty();
    const auto error = plls(input, output);
    assert(!error && "Halide error.");

    output.copy_to_host();
    return rolloff;
}

}  // namespace

int
main() {
    arma::Cube<uint16_t> image(width, height, zsize);

    // Load data
    constexpr char filename[] = (HDF5_FILE_PATH);
    auto file = File(filename, File::ReadOnly);

    std::cout << "Reading z-stack..." << std::endl;
    const auto z_stack = readFluorescenceZStack(file);

    {
        arma::Mat<uint8_t> autofocus_plane(2, wsize);
        std::cout << "Reading pre-computed autofocus values..." << std::endl;
        file.getDataSet("autofocus_plane").read(autofocus_plane.memptr());
        std::cout << "Expected focal plane = " << autofocus_plane(0, well_id) << std::endl;
    }

    std::cout << "Autofocusing..." << std::endl;
    const auto rolloff = autofocus(z_stack);
    rolloff.print("Power log-log slope = ");

    const arma::uword plane_id = rolloff.index_max();

    // Save focal plane to local folder
    {
        using namespace arma;
        std::cout << "Saving focused image at z=" << plane_id << "..." << std::endl;

        // Extract green channel
        auto x_index = regspace<uvec>(0, 2, width - 1);
        auto y_index = regspace<uvec>(0, 2, height - 1);
        const auto focussed_image =
            conv_to<fmat>::from(z_stack.slice(plane_id).submat(x_index + 1, y_index) +
                                z_stack.slice(plane_id).submat(x_index, y_index + 1));

        const auto normalized_image = arma::conv_to<arma::Mat<uint8_t>>::from(
            focussed_image.t() * 255.f / focussed_image.max());

        const std::string output_filename("in_focus.pgm");
        normalized_image.save(output_filename, arma::pgm_binary);
        std::cout << "Saved to " << output_filename << std::endl;
    }

//#define RUN_BENCHMARK
#ifdef RUN_BENCHMARK
    {
        std::cout << "Running benchmark of " << zsize << " focal planes by " << wsize
                  << " times:" << std::endl;

        using namespace std::chrono;
        const int N = wsize;

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < N; i++) plls(input, output);

        auto diff = duration_cast<milliseconds>(high_resolution_clock::now() - t0);
        float toc = diff.count();
        std::cout << toc << "ms enlapsed, " << (toc / N) << "ms per compute, " << (toc / N / zsize)
                  << "ms per plane" << std::endl;
    }

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
