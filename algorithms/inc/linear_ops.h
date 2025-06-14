#pragma once

#include "Halide.h"
#include "complex.h"
#include "vars.hpp"

namespace linear_ops {

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
adjust_brightness(const Func im, Expr gamma, Expr vmin = 0, Expr vmax = 255) ;

Func
norm1(const Func im, const RDom& r) ;

Func
norm2Squared(const Func im, const RDom& r) ;

template <typename F>
std::pair<F, Func>
applyCheckerboard(const F& im);

std::pair<ComplexFunc, ComplexFunc>
shiftAndMask(const Func im, const Func offset, const ComplexFunc mask, const int tile_size);

// First order update, variable step size
ComplexFunc
epryGradientDescent(const ComplexFunc p, const ComplexFunc q, Expr alpha, Expr delta,
                    const ComplexFunc in);

// First order update, constant step size
ComplexFunc
epryGradientDescent(const ComplexFunc p, const ComplexFunc q, Expr alpha_squared,
                    const ComplexFunc in);

ComplexFunc
epryPseudoNewton(const ComplexFunc p, const ComplexFunc q, Expr alpha, Expr delta,
                 const ComplexFunc in);

Func
hotPixelSuppression(const Func input);

/** Deinterleave and fuse two Bayer images to one RGB image. */
Func
deinterleave(const Func raw_green, const Func raw_red);

template <typename T>
std::tuple<ComplexFunc, Func, Func>
fft2C2C(const T& input, const int width, bool is_fwd = true, std::string&& label = "input_mux") {
    using vars::x;
    using vars::y;
    using vars::k;
    using vars::i;

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

    assert(input_func.dimensions() == 3);
    fft2_internal.function().extern_definition_proxy_expr() =
        input_func(0, 0, 0) + input_func(1, width - 1, width - 1);

    ComplexFunc transformed{"transformed"};
    transformed(x, y, _) = ComplexExpr{fft2_internal(0, x, y, _), fft2_internal(1, x, y, _)};

    return {transformed, fft2_internal, input_func};
}

}  // namespace linear_ops