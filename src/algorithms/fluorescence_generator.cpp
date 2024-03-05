#include "Halide.h"
#include "constants.hpp"

namespace {

#include "linear_ops.hpp"
using namespace Halide;

using vars::c;
using vars::x;
using vars::y;

class fluorescenceImage : public Generator<fluorescenceImage> {
    /** Normalize colors */
    Func normalize(Func, Expr, Expr, Expr, Expr);

   public:
    /** Raw green channel image */
    Input<Buffer<uint16_t, 2>> egfp{"green"};

    /** Raw red channel image */
    Input<Buffer<uint16_t, 2>> txred{"txred"};

    /** RGB image */
    Output<Buffer<uint8_t, 3>> output{"output"};

    /** Algorithm definition */
    void generate();

    /** Algorithm schedule */
    void schedule();
};

////////////////////////////////////////////////////////////////////////////////
void
fluorescenceImage::generate() {
    using linear_ops::deinterleave;
    using linear_ops::hotPixelSuppression;

    // Boundary condition
    const auto clamped_g = BoundaryConditions::repeat_edge(egfp);
    const auto clamped_r = BoundaryConditions::repeat_edge(txred);

    const auto denoised_g = hotPixelSuppression(clamped_g);
    const auto denoised_r = hotPixelSuppression(clamped_r);

    const auto deinterleaved = deinterleave(denoised_g, denoised_r);

    const Func normalized = normalize(deinterleaved, egfp.width() / 2, egfp.height() / 2, 2, 99);

    Func quantized{"quantized"};
    quantized(x, y, c) = saturating_cast<uint8_t>(normalized(x, y, c));

    // Scale the image by 2X.
    output(x, y, c) = quantized(x / 2, y / 2, c);
}

void
fluorescenceImage::schedule() {
    using constants::height;
    using constants::width;
    // Estimates
    for (auto* p : {&egfp, &txred}) {
        p->dim(0).set_bounds(0, width).set_stride(1);
        p->dim(1).set_bounds(0, height).set_stride(width);

        p->dim(0).set_estimate(0, width);
        p->dim(1).set_estimate(0, height);
    }

    output.dim(0).set_bounds(0, width).set_stride(1);
    output.dim(1).set_bounds(0, height).set_stride(width);
    output.dim(2).set_bounds(0, 3).set_stride(width * height);

    output.dim(0).set_estimate(0, width);
    output.dim(1).set_estimate(0, height);
    output.dim(2).set_estimate(0, 3);

    if (using_autoscheduler()) {
        // Do nothing
        return;
    }

    // CPU
    output.compute_root().unroll(c);
}

Func
fluorescenceImage::normalize(Func input, Expr width, Expr height, Expr percentile_min,
                             Expr percentile_max) {
    const RDom r{0, width, 0, height};

    // Compute the histogram
    Func histogram{"histogram"};
    const Var i{"i"};
    histogram(c, i) = 0;
    histogram(c, input(r.x, r.y, c)) += 1;

    // Integrate it to introduce a cdf
    Func cdf{"cdf"};
    cdf(c, i) = histogram(c, 0);
    const RDom idx(1, width * height - 1);
    cdf(c, idx.x) = cdf(c, idx.x - 1) + histogram(c, idx.x);

    Func vmin{"vmin"};
    Func vmax{"vmax"};
    // vmin(c) = cast<float>(minimum(input(c, r.x, r.y)));
    // vmax(c) = cast<float>(maximum(input(c, r.x, r.y)));
    vmin(c) = argmax(cdf(c, idx.x) >= width * height * percentile_min / 100)[0];
    vmax(c) = argmax(cdf(c, idx.x) >= width * height * percentile_max / 100)[0];

    Func normalized{"normalized"};
    normalized(x, y, c) = (input(x, y, c) - vmin(c)) * 255 / (vmax(c) - vmin(c) + 1e-3f);

    return normalized;
}
}  // namespace
// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:

HALIDE_REGISTER_GENERATOR(fluorescenceImage, raw2bgr)
