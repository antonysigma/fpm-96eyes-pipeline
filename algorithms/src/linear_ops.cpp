#include "linear_ops.h"
#include "types.h"
#include "vars.hpp"

namespace linear_ops {

using vars::c;
using vars::i;
using vars::k;
using vars::x;
using vars::y;

using namespace types;
using namespace Halide;

Func
adjust_brightness(const Func im, Expr gamma, Expr vmin, Expr vmax) {
    Func gamma_adjusted{"gamma_adjusted"};

    const Expr vrange = vmax - vmin;
    gamma_adjusted(x, y, k) =
        select(x <= vmin, 0, pow(cast<float>(im(x, y) - vmin) / vrange, 1.0f / gamma) * 65535.0f);

    return gamma_adjusted;
}

Func
norm1(const Func im, const RDom& r) {
    Func norm1{"norm1"};

    norm1(k) = sum(abs(cast<float>(im(r.x, r.y, k))));

    return norm1;
}

Func
norm2Squared(const Func im, const RDom& r) {
    Func im_f32{"im_f32"};
    im_f32(x, y, k) = cast<float>(im(x, y, k));

    Func norm2sq{"norm2sq"};
    norm2sq(k) += im_f32(r.x, r.y, k) * im_f32(r.x, r.y, k);

    return norm2sq;
}

template <typename F>
std::pair<F, Func>
applyCheckerboard(const F& im) {
    static const Func sign = [=]() -> Func {
        Func sign{"sign"};
        sign(x, y) = ((x + y) % 2 == 1);

        return sign;
    }();

    F checkerboard{"checkerboard"};

    checkerboard(x, y, _) = select(sign(x, y), -im(x, y, _), im(x, y, _));

    return {checkerboard, sign};
}

std::pair<ComplexFunc, ComplexFunc>
shiftAndMask(const Func im, const Func offset, const ComplexFunc mask, const int tile_size) {
    ComplexFunc shifted{"shifted"};

    // Crop a tile, assuming the source image has repeating edge boundary condition.
    // const Expr new_x = clamp(x + offset(X, k), 0, tile_size - 1);
    // const Expr new_y = clamp(y + offset(Y, k), 0, tile_size - 1);
    const Expr new_x = x + offset(X, k);
    const Expr new_y = y + offset(Y, k);
    shifted(x, y, k) = im(new_x, new_y);

    ComplexFunc masked{"masked"};
    masked(x, y, k) = shifted(x, y, k) * mask(x, y);

    return {masked, shifted};
}

ComplexFunc
epryGradientDescent(const ComplexFunc p, const ComplexFunc q, Expr alpha, Expr delta,
                    const ComplexFunc in) {
    const Expr magn = abs(p(x, y));

    const Expr w = lerp(magn, alpha, delta) * alpha;

    ComplexFunc epry{"epry"};
    epry(x, y) = in(x, y) + p(x, y) * conj(q(x, y)) / w;

    return epry;
}

// First order update, constant step size
ComplexFunc
epryGradientDescent(const ComplexFunc p, const ComplexFunc q, Expr alpha_squared,
                    const ComplexFunc in) {
    ComplexFunc epry{"epry"};
    epry(x, y) = in(x, y) + p(x, y) * conj(q(x, y)) / alpha_squared;

    return epry;
}

ComplexFunc
epryPseudoNewton(const ComplexFunc p, const ComplexFunc q, Expr alpha, Expr delta,
                 const ComplexFunc in) {
    const Expr sumsq = re(p(x, y)) * re(p(x, y)) + im(p(x, y)) * im(p(x, y));

    const Expr w = sumsq / (sumsq * sumsq + delta) / alpha;

    ComplexFunc epry{"epry"};
    epry(x, y) = in(x, y) + p(x, y) * conj(q(x, y)) * w;

    return epry;
}

Func
hotPixelSuppression(const Func input) {
    const Expr a = max(input(x - 2, y), input(x + 2, y), input(x, y - 2), input(x, y + 2));

    Func denoised{"denoised"};
    denoised(x, y) = clamp(input(x, y), 0, a);

    return denoised;
}

Func
deinterleave(const Func raw_green, const Func raw_red) {
    using namespace types;

    // Deinterleave the color channels
    Func deinterleaved{"deinterleaved"};

    deinterleaved(x, y, c) =
        select(c == egfp, raw_green(2 * x + 1, 2 * y) + raw_green(2 * x, 2 * y + 1),  //
               c == txred, raw_red(2 * x + 1, 2 * y + 1),                             //
               0);

    return deinterleaved;
}
}  // namespace linear_ops