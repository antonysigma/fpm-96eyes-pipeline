#include "Halide.h"

namespace {
using namespace Halide;

const size_t maxG = 4;

/** Compute power log-log slope values for all focal depths */
class autofocus : public Generator<autofocus> {
    Var x{"x"}, y{"y"}, z{"z"};

    /** Blur and downsample the image by half of its width and height */
    Func downsample(Func) const;

    /** Interpolate the image by twice its width and height */
    Func upsample(Func) const;

    /** Gaussian pyramid */
    Func gPyramid[maxG];

    /** Average pixel value in the Laplacian pyramid at particular level */
    Func mean[maxG - 1];

    /** Standard deviation of pixel value in the Laplacian pyramid at particular level */
    Func half_sigma[maxG - 1];

   public:
    /** Number of levels */
    // Input<uint8_t> maxG{"maxLevel"};

    /** Input image */
    Input<Buffer<uint16_t>> input{"input", 3};

    /** Output value */
    Output<Buffer<float>> rolloff{"rolloff", 1};

    /** Algorithm definition */
    void generate();

    /** Algorithm schedule */
    void schedule();
};

Func
autofocus::downsample(Func f) const {
    Func downx{"downx"}, downy{"downy"};

    downx(x, y, z) = (f(2 * x - 2, y, z) + 4.0f * (f(2 * x - 1, y, z) + f(2 * x + 1, y, z)) +
                      6.0f * f(2 * x, y, z) + f(2 * x + 2, y, z)) /
                     16.0f;
    downy(x, y, z) =
        (downx(x, 2 * y - 2, z) + 4.0f * (downx(x, 2 * y - 1, z) + downx(x, 2 * y + 1, z), z) +
         6.0f * downx(x, 2 * y, z) + downx(x, 2 * y + 2, z)) /
        16.0f;
    return downy;
}

Func
autofocus::upsample(Func f) const {
    Func upx("upx"), upy("upy");

    upx(x, y, z) = select(x % 2 == 1, (f(x / 2, y, z) + f(x / 2 + 1, y, z)) / 2.0f,
                          (f(x / 2 - 1, y, z) + f(x / 2 + 1, y, z) + 6.0f * f(x / 2, y, z)) / 8.0f);

    upy(x, y, z) =
        select(y % 2 == 1, (upx(x, y / 2, z) + upx(x, y / 2 + 1, z)) / 2.0f,
               (upx(x, y / 2 - 1, z) + upx(x, y / 2 + 1, z) + 6.0f * upx(x, y / 2, z)) / 8.0f);

    return upy;
}

void
autofocus::generate() {
    // Boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Pixel binning (2x2) of green channel
    Func pixel_binning_green("pixel_binning");
    pixel_binning_green(x, y, z) = clamped(2 * x + 1, 2 * y, z) + clamped(2 * x, 2 * y + 1, z);

    Func lPyramid[maxG - 1];              // Laplacian pyramid
    Func spatial_freq_density[maxG - 1];  // Signal strength of a particular spatial frequency band

    gPyramid[0](x, y, z) = cast<float>(pixel_binning_green(x, y, z));

    for (int i = 1; i < maxG; ++i) {
        gPyramid[i](x, y, z) = downsample(gPyramid[i - 1])(x, y, z);

        auto& L = lPyramid[i - 1];
        L(x, y, z) = gPyramid[i - 1](x, y, z) - upsample(gPyramid[i])(x, y, z);

        // Compute standard deviation
        Expr width = input.width() >> i;
        Expr height = input.height() >> i;
        Expr pixel_count = width * height;
        auto& Mean = mean[i - 1];
        auto& Half_sigma = half_sigma[i - 1];

        RDom r(0, width, 0, height);
        Mean(z) = sum(L(r.x, r.y, z)) / pixel_count;
        Half_sigma(z) =
            sqrt(sum((L(r.x, r.y, z) - Mean(z)) * (L(r.x, r.y, z) - Mean(z))) / pixel_count / 4);

        // Compute spatial frequency density (SFD)
        Func magnitude;
        magnitude(x, y, z) = abs(L(x, y, z) - Mean(z));

        spatial_freq_density[i - 1](z) =
            sum(select(magnitude(r.x, r.y, z) > Half_sigma(z), magnitude(r.x, r.y, z), 0)) /
            pixel_count;
    }

    // Compute SFD roll off, all Func other than the specified levels will be optimized out
    // TODO: set min_level and max_level at run time

    {
        Expr width = input.width() >> maxG;
        Expr height = input.height() >> maxG;
        RDom r(0, width, 0, height);

        Expr pixel_count = width * height;
        Func average_brightness;

        average_brightness(z) = sum(gPyramid[maxG - 1](r.x, r.y, z)) / pixel_count;
        rolloff(z) = (spatial_freq_density[maxG - 2](z) - spatial_freq_density[1](z)) /
                     average_brightness(z);
    }
}

void
autofocus::schedule() {
    if (using_autoscheduler()) {
        input.dim(0).set_estimate(0, 2592);
        input.dim(1).set_estimate(0, 1944);
        input.dim(2).set_estimate(0, 21);

        rolloff.dim(0).set_estimate(0, 21);

        // const int kParallelism = 32;
        // const int kLastLevelCacheSize = 16 * 1024 * 1024;
        // const int kBalance = 40;
        // MachineParams machine_params(kParallelism, kLastLevelCacheSize, kBalance);
    } else {
        for (int i = 0; i < maxG - 1; ++i) {
            gPyramid[i].compute_root();
            mean[i].compute_root();
            half_sigma[i].compute_root();
            rolloff.parallel(z);
        }
    }
}
}  // namespace
// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:

HALIDE_REGISTER_GENERATOR(autofocus, plls)
