#include "Halide.h"
#include "complex.h"
namespace {

#include "linear_ops.h"

using namespace Halide;

class stitchPhase : public Generator<stitchPhase> {
    Var x{"x"}, y{"y"}, z{"z"};
    Var k{"k"};

    Func erode(Func);
    Func distance_transform(Func);
    Func featherEdge(Func);
    ComplexFunc stitch(ComplexFunc, Func);

   public:
    /** 4 layers of images with datatype = std::complex<float> */
    Input<Buffer<float>> input{"input", 4};

    Output<Buffer<uint8_t>> output{"output", 2};

    void generate();
    void schedule();
};

////////////////////////////////////////////////////////////////////////////////
Func
stitchPhase::erode(Func mask) {
    Func eroded{"eroded"};

    RDom r(-1, 3, -1, 3);
    eroded(x, y, z) = product(mask(x + r.x, y + r.y, z));

    return eroded;
}

Func
stitchPhase::featherEdge(Func input) {
    const Expr N = 33;
    RDom r(-N / 2, N);

    Func blurx{"blurx"}, blury{"blury"};
    blurx(x, y, z) += input(x + r.x, y, z);
    blury(x, y, z) += blurx(x, y + r.x, z);

    Func eroded{"eroded"};
    eroded(x, y, z) = max(blury(x, y, z) - N * N / 2, 0);
    return eroded;
}

Func
stitchPhase::distance_transform(Func mask) {
    Func alpha_blended{"alpha_blended"};

    // const Expr maxLevel = 16;
    // Func eroded_mask{"eroded_mask"};
    // Var k{"k"};

    // eroded_mask(x, y, z, k) = mask(x, y, z);

    // RDom r(1, maxLevel-1);
    // eroded_mask(x, y, z, r.x) = erode(eroded_mask)(x, y, z, r.x-1);

    // RDom r2(0, maxLevel);
    // alpha_blended(x, y, z) = sum(eroded_mask(x, y, z, r2.x));

    const int maxLevel = 64;
    Func eroded_mask[maxLevel];

    eroded_mask[0] = mask;
    alpha_blended(x, y, z) = mask(x, y, z);
    for (int k = 1; k < maxLevel; ++k) {
        eroded_mask[k](x, y, z) = erode(eroded_mask[k - 1])(x, y, z);
        alpha_blended(x, y, z) += eroded_mask[k](x, y, z);
    }

    return alpha_blended;
}

ComplexFunc
stitchPhase::stitch(ComplexFunc raw, Func alpha) {
    ComplexFunc stitched{"stitched"};

    RDom r(0, 4);
    stitched(x, y) = sum(raw(x, y, r.x) * alpha(x, y, r.x)) / sum(alpha(x, y, r.x));

    return stitched;
}

void
stitchPhase::generate() {
    ComplexFunc in;
    in(x, y, z) = ComplexExpr(input(0, x, y, z), input(1, x, y, z));

    Func mask{"mask"};
    mask(x, y, z) = select(abs(in(x, y, z)) > 0, uint16_t(1), uint16_t(0));

    const Expr width = input.dim(1).extent();
    const Expr height = input.dim(2).extent();
    Func clamped_mask;

    clamped_mask(x, y, z) = mask(clamp(x, 0, width - 1), clamp(y, 0, height - 1), z);

    // Func alpha = distance_transform(clamped_mask);
    Func alpha = featherEdge(clamped_mask);
    ComplexFunc stitched = stitch(in, alpha);

    Func phase{"phase"};
    phase(x, y) = arg(stitched(x, y));

    const float pi = 3.1415926f;
    const float gain = 127.f / pi * 8;

    output(x, y) = saturating_cast<uint8_t>(phase(x, y) * gain + 127.f);
}

void
stitchPhase::schedule() {
    input.dim(0).set_estimate(0, 2);
    input.dim(1).set_estimate(0, 2592);
    input.dim(2).set_estimate(0, 1944);
    input.dim(3).set_estimate(0, 4);

    output.dim(0).set_estimate(0, 2592);
    output.dim(1).set_estimate(0, 1944);
}
}  // namespace
// We compile this file along with tools/GenGen.cpp. That file defines
// an "int main(...)" that provides the command-line interface to use
// your generator class. We need to tell that code about our
// generator. We do this like so:

HALIDE_REGISTER_GENERATOR(stitchPhase, get_phase)
