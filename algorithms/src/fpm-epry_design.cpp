#include "constants.hpp"
#include "fpm-epry_generator.h"
#include "linear_ops.hpp"
#include "types.h"
#include "vars.hpp"

namespace {
using namespace Halide;

using vars::i;
using vars::k;
using vars::x;
using vars::y;

constexpr bool FORWARD = true;
constexpr bool INVERSE = !FORWARD;

enum axis_t { X = 0, Y = 1 };

ComplexFunc
generateLR(const ComplexFunc& high_res, const Func& offset, const int32_t k,
           const ComplexFunc& pupil, const Expr width) {
    const ComplexFunc shift_multiplied;

    const Expr new_x = clamp(x + offset(X, k), 0, width * 2 - 1);
    const Expr new_y = clamp(y + offset(Y, k), 0, width * 2 - 1);
    shift_multiplied(x, y) = high_res(new_x, new_y) * pupil(x, y);

    return shift_multiplied;
}

std::pair<ComplexFunc, Func>
replaceIntensity(const ComplexFunc& simulated, const Func& low_res,
                 const int32_t illumination_idx) {
    const Func magn{"magn_low_res"};
    magn(x, y) = abs(simulated(x, y)) + 1e-6f;

    const ComplexExpr phase_angle = simulated(x, y) / magn(x, y);

    ComplexFunc replaced;
    replaced(x, y) = phase_angle * low_res(x, y, illumination_idx);

    return {replaced, magn};
}

std::pair<Func, Func>
normInf(const ComplexFunc input, const RDom& r, const std::string& label) {
    Func sumsq{"sumsq_" + label};
    sumsq(x, y) = re(input(x, y)) * re(input(x, y)) + im(input(x, y)) * im(input(x, y));

    Func alpha{label};
    alpha(x) = 0.0f;
    alpha(x) = max(alpha(x), sumsq(r.x, r.y));
    alpha(x) = sqrt(alpha(x));

    return {alpha, sumsq};
}

std::pair<ComplexFunc, ComplexFunc>
updateHR(const ComplexFunc& high_res, const ComplexFunc& f_difference, const ComplexFunc& pupil,
         const Func& offset, const Expr alpha, const int32_t k, const Expr width) {
    const Expr pupil_sumsq = re(pupil(x, y)) * re(pupil(x, y)) + im(pupil(x, y)) * im(pupil(x, y));
    Func scale_factor{"scale_factor_newton"};
    scale_factor(x, y) = sqrt(pupil_sumsq) / (pupil_sumsq + 1e-6f) / alpha;

    ComplexFunc delta{"delta"};
    delta(x, y) = scale_factor(x, y) * conj(pupil(x, y)) * f_difference(x, y);

    const Expr in_x_range = (x >= offset(X, k)) && (x < (offset(X, k) + width));
    const Expr in_y_range = (y >= offset(Y, k)) && (y < (offset(Y, k) + width));

    ComplexFunc high_res_new{"high_res"};
    const Expr new_x = clamp(x - offset(X, k), 0, width - 1);
    const Expr new_y = clamp(y - offset(Y, k), 0, width - 1);

    high_res_new(x, y) = select(  //
        in_x_range && in_y_range, high_res(x, y) - delta(new_x, new_y), high_res(x, y));

    return {high_res_new, delta};
}

ComplexFunc
updatePupil(const ComplexFunc& current_pupil, const ComplexFunc& f_difference,
            const ComplexFunc& f_object, const Expr beta, const Expr weight = 1e-6f) {
    const Expr f_object_sumsq =
        re(f_object(x, y)) * re(f_object(x, y)) + im(f_object(x, y)) * im(f_object(x, y));
    Func scale_factor{"scale_factor_lerp"};
    scale_factor(x, y) = fast_inverse(lerp(beta, sqrt(f_object_sumsq), weight));

    ComplexFunc new_pupil{"pupil"};
    new_pupil(x, y) =
        current_pupil(x, y) - scale_factor(x, y) * conj(f_object(x, y)) * f_difference(x, y);

    return new_pupil;
}

}  // namespace

namespace algorithms {
void
FPMEpry::design() {
    assert(uint32_t(n_unroll) >= n_normalize);

    const int width = tile_size;

    const int32_t n_normalize_iter = int32_t(n_illumination) * n_normalize;
    const int32_t n_unroll_iter = int32_t(n_illumination) * n_unroll;

    {
        high_res.resize(n_unroll_iter + 1);
        ComplexFunc h{"high_res"};
        h(x, y) = {high_res_prev(0, x, y), high_res_prev(1, x, y)};
        high_res.front() = std::move(h);
    }

    {
        // The || x ||_00, aka peak value of the Fourier spectrum is the DC term.
        const Expr center_x = width;
        const Expr center_y = width;
        beta() = abs(high_res.front()(center_x, center_y));
    }

    {
        pupil.resize(1);
        ComplexFunc p{"pupil"};
        p(x, y) = {pupil_prev(0, x, y), pupil_prev(1, x, y)};
        pupil.front() = std::move(p);
    }

    r = RDom(0, width, 0, width);
    std::tie(alpha, sumsq_alpha) = normInf(pupil.front(), r, "alpha");

    const auto fpmIter = [&](const ComplexFunc& high_res_prev, const ComplexFunc& current_pupil,
                             const int32_t illumination_idx)
        -> std::tuple<ComplexFunc, ComplexFunc, ComplexFunc, Func, Func, Func, Func, ComplexFunc,
                      Func> {
        using linear_ops::fft2C2C;

        const auto f_estimated =
            generateLR(high_res_prev, k_offset, illumination_idx, current_pupil, width);

        ComplexFunc estimated;
        Func f_estimated_interleaved;
        Func ifft2;
        std::tie(estimated, ifft2, f_estimated_interleaved) =
            fft2C2C(f_estimated, width, INVERSE, "f_estimated_interleaved");

        const auto [replaced, magn_low_res] =
            replaceIntensity(estimated, low_res, illumination_idx);

        ComplexFunc f_replaced;
        Func fft2;
        Func replaced_interleaved;
        std::tie(f_replaced, fft2, replaced_interleaved) =
            fft2C2C(replaced, width, FORWARD, "replaced_interleaved");

        ComplexFunc f_difference{"f_difference"};
        f_difference(x, y) = f_replaced(x, y) - f_estimated(x, y);

        const auto [this_high_res, delta] = updateHR(high_res_prev, f_difference, current_pupil,
                                                     k_offset, alpha(0), illumination_idx, width);

        return {this_high_res,        f_difference, f_estimated, f_estimated_interleaved,
                replaced_interleaved, ifft2,        fft2,        delta,
                magn_low_res};
    };

    f_estimated_interleaved.resize(n_unroll_iter);
    replaced_interleaved.resize(n_unroll_iter);
    fft2.resize(n_unroll_iter);
    ifft2.resize(n_unroll_iter);
    delta.resize(n_unroll_iter);
    magn_low_res.resize(n_unroll_iter);

    for (int32_t i = 0; i < n_normalize_iter; i++) {
        using std::ignore;
        std::tie(high_res[i + 1], ignore, ignore, f_estimated_interleaved[i],
                 replaced_interleaved[i], ifft2[i], fft2[i], delta[i], magn_low_res[i]) =
            fpmIter(high_res[i], pupil.back(), i % n_illumination);
    }

    for (int32_t i = n_normalize_iter; i < n_unroll_iter; i++) {
        ComplexFunc f_diff;
        ComplexFunc f_estimated;
        std::tie(high_res.at(i + 1), f_diff, f_estimated, f_estimated_interleaved[i],
                 replaced_interleaved[i], ifft2[i], fft2[i], delta[i], magn_low_res[i]) =
            fpmIter(high_res[i], pupil.back(), i % n_illumination);
        pupil.emplace_back(updatePupil(pupil.back(), f_diff, f_estimated, beta()));
        f_difference.emplace_back(std::move(f_diff));
    }

    high_res_new(i, x, y) = mux(i, {re(high_res.back()(x, y)), im(high_res.back()(x, y))});
    pupil_new(i, x, y) = mux(i, {re(pupil.back()(x, y)), im(pupil.back()(x, y))});
}
}  // namespace algorithms

HALIDE_REGISTER_GENERATOR(algorithms::FPMEpry, fpm_epry)