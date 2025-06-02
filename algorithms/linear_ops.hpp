#pragma once

#include "Halide.h"
#include "complex.h"
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

/** Compute
 * Y = [(X - vmin) / (vmax - vmin)] ^ (1/gamma) * (2^16 - 1)
 * where gamma>0 is a scalar, and
 * X,Y are vectors of unsigned integer
 *
 * @param[in] in integer vector
 * @param[in] gamma positive real number
 * @param[in] vmin max value of X, preferrably the dark frame value (default = 0)
 * @param[in] vmax max value of X [255 for uint8 (default), 65535 for uint16]
 * @return integer vector
 */
Func
adjust_brightness(const Func im, Expr gamma, Expr vmin = 0, Expr vmax = 255) {
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

// First order update, variable step size
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

/** Deinterleave and fuse two Bayer images to one RGB image. */
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

template <typename T>
std::tuple<ComplexFunc, Func, Func>
fft2C2C(const T& input, const int width, bool is_fwd = true, std::string&& label = "input_mux") {
    Func fft2_internal{is_fwd ? "fft2_mux" : "ifft2_mux"};

    const auto extern_func =
        (is_fwd) ? std::string{"externCufftFwd"} : std::string{"externCufftInv"};

    std::vector<ExternFuncArgument> input_args;

    Func input_func{label};
    if constexpr (!std::is_same_v<T, ComplexFunc> || std::is_same_v<T, Buffer<float, 4>>) {
        input_func = input;
    } else {  // T == ComplexFunc
        using vars_t = std::vector<Var>;
        const auto vars_dst = (input.dimensions() == 3) ? vars_t{i, x, y, k} : vars_t{i, x, y};
        const auto vars_src = (input.dimensions() == 3) ? vars_t{x, y, k} : vars_t{x, y};

        input_func(vars_dst) = mux(i, {
                                          input(vars_src).re(),
                                          input(vars_src).im(),
                                      });
        // input_func.bound(i, 0, 2);
    }

    const auto n_dim = input_func.dimensions();
    fft2_internal.define_extern(extern_func, {input_func}, halide_type_of<float>(), n_dim,
                                NameMangling::Default, DeviceAPI::CUDA);

    ComplexFunc transformed{"transformed"};
    transformed(x, y, _) = ComplexExpr{fft2_internal(0, x, y, _), fft2_internal(1, x, y, _)};

    return {transformed, fft2_internal, input_func};
}

}  // namespace linear_ops