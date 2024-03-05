#include <armadillo>
#include <chrono>
#include <complex>
#include <iostream>
#include <string>

#include "HalideBuffer.h"
#include "get_phase.h"
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"
#include "read-slice.h"

namespace {
using namespace HighFive;
using Halide::Runtime::Buffer;

constexpr size_t width = 2592;
constexpr size_t height = 1944;
constexpr size_t wsize = 96;
constexpr size_t zsize = 4;

using cx_fcube = arma::Cube<std::complex<float>>;
using storage::cx_fcube_t;

arma::Mat<uint8_t>
raw2phase(cx_fcube_t& raw) {
    arma::Mat<uint8_t> phase(width, height);
    Buffer<uint8_t> output(phase.memptr(), width, height);

    // Roll-off
    raw.set_host_dirty();
    const auto error = get_phase(raw, output);
    assert(!error && "Halide error.");

    output.copy_to_host();

    return phase;
}

}  // namespace

int
main() {
    // Load data
    constexpr char filename[] = (HDF5_FILE_PATH);
    auto file = File(filename, File::ReadOnly);

    constexpr int well_id = 5;
    auto dataset = file.getDataSet("himr");
    auto raw = storage::readQPILayers(dataset, well_id, width, height);
    const auto phase = raw2phase(raw);

    // Memory map to buffer
    const std::string output_filename("phase_channel.pgm");
    phase.save(output_filename, arma::pgm_binary);
    std::cout << "Saved to " << output_filename << std::endl;

    //#define RUN_BENCHMARK
#ifdef RUN_BENCHMARK
    {
        const int N = wsize;
        std::cout << "Running benchmark by " << N << " times:" << std::endl;

        using namespace std::chrono;

        auto t0 = high_resolution_clock::now();
        for (int i = 0; i < N; i++) get_phase(input, output);

        auto diff = duration_cast<milliseconds>(high_resolution_clock::now() - t0);
        float toc = diff.count();
        std::cout << toc << "ms enlapsed, " << (toc / N) << "ms per compute" << std::endl;
    }
#endif

    return 0;
}
