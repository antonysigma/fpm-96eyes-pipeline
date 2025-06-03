#include <vector>

#include "Halide.h"
#include "complex.h"
#include "constants.hpp"

namespace {
#include "linear_ops.hpp"
#include "vars.hpp"

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

/** Simulate the low-resolution image */
class FPMEpry : public Generator<FPMEpry> {
    Input<Buffer<const float, 3>> low_res{"low_res"};
    Input<Buffer<float, 3>> high_res_prev{"high_res_prev"};
    Input<Buffer<float, 3>> pupil_prev{"pupil_prev"};

    Input<Buffer<const int32_t, 2>> k_offset{"k_offset"};

    Output<Buffer<float, 3>> high_res_new{"high_res_new"};
    Output<Buffer<float, 3>> pupil_new{"pupil_new"};

    GeneratorParam<int32_t> n_illumination{"n_illumination", 3, 9, 49};
    GeneratorParam<int32_t> n_unroll{"n_unroll", 1, 1, 30};
    GeneratorParam<int32_t> n_normalize{"n_normalize", 0, 0, 5};
    GeneratorParam<int32_t> tile_size{"tile_size", 128, 0, 256};

    RDom r;
    Func sumsq_alpha;
    Func alpha;
    Func beta{"beta"};

    std::vector<Func> f_estimated_interleaved;
    std::vector<Func> replaced_interleaved;
    std::vector<Func> magn_low_res;
    std::vector<Func> fft2;
    std::vector<Func> ifft2;
    std::vector<ComplexFunc> delta;
    std::vector<ComplexFunc> high_res;
    std::vector<ComplexFunc> f_difference;
    std::vector<ComplexFunc> pupil;

   public:
    void generate();
    void schedule();
};

void
FPMEpry::generate() {
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

void
FPMEpry::schedule() {
    const int W = tile_size;
    const int W2 = W * 2;

    low_res.dim(0).set_bounds(0, W).set_stride(1);
    low_res.dim(1).set_bounds(0, W).set_stride(W);
    low_res.dim(2).set_min(0).set_stride(W * W);
    const auto n_slides = low_res.dim(2).extent();

    k_offset.dim(0).set_bounds(0, 2).set_stride(1);
    k_offset.dim(1).set_bounds(0, n_slides).set_stride(2);

    const auto setComplexBound = [=](auto& p, const int w) {
        p.dim(0).set_bounds(0, 2).set_stride(1);
        p.dim(1).set_bounds(0, w).set_stride(2);
        p.dim(2).set_bounds(0, w).set_stride(2 * w);
    };

    setComplexBound(high_res_prev, W2);
    setComplexBound(high_res_new, W2);
    setComplexBound(pupil_prev, W);
    setComplexBound(pupil_new, W);

    if (using_autoscheduler()) {
        k_offset.set_estimates({{0, 2}, {0, n_illumination}});

        high_res_prev.set_estimates({{0, 2}, {0, W2}, {0, W2}});

        high_res_new.set_estimates({{0, 2}, {0, W2}, {0, W2}});

        pupil_prev.set_estimates({{0, 2}, {0, W}, {0, W}});

        pupil_new.set_estimates({{0, 2}, {0, W}, {0, W}});

        return;
    }

    const auto target = get_target();
    if (target.has_gpu_feature()) {
        const Var x_vo{"xo"}, y_o{"yo"}, x_vi{"xi"}, y_i{"yi"};

        pupil_new  //
            .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, W, 1)
            .unroll(i);

        high_res_new  //
            .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, W, 1)
            .unroll(i);

        for (auto& s : pupil) {
            s.compute_root()  //
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1);
        }

        for (auto& s : f_difference) {
            s.compute_root()  //
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1);
        }

        for (auto& s : high_res) {
            s.compute_root()  //
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1);
        }

        for (auto& s : delta) {
            s.compute_root()  //
                .bound(x, 0, W)
                .bound(y, 0, W)
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1);
        }

        for (auto& s : fft2) {
            s.compute_root();
        }

        for (auto& s : replaced_interleaved) {
            s.compute_root()  //
                .bound(x, 0, W)
                .bound(y, 0, W)
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1)
                .bound(i, 0, 2)
                .unroll(i);
        }

        for (size_t idx = 0; idx < replaced_interleaved.size(); idx++) {
            magn_low_res[idx].compute_at(replaced_interleaved[idx], x_vi);
        }

        for (auto& s : ifft2) {
            s.compute_root();
        }

        for (auto& s : f_estimated_interleaved) {
            s.compute_root()  //
                .bound(x, 0, W)
                .bound(y, 0, W)
                .gpu_tile(x, y, x_vo, y_o, x_vi, y_i, 128, 1)
                .bound(i, 0, 2)
                .unroll(i);
        }

        // Fuse zero-init, maximum(), and sqrt() into one single GPU kernel.
        alpha.compute_at(alpha.in(), x_vi);

        alpha.in().compute_root().split(x, x_vo, x_vi, 1).gpu(x_vo, x_vi);

        // Compute intermediate max values by columns.
        const RVar rxo{"rxo"}, ryo{"ryo"}, rxi{"rxi"}, ryi{"ryi"};
        alpha.update(0).tile(r.x, r.y, rxo, ryo, rxi, ryi, 1, W);

        // implement sqrt() in GPU thread
        alpha.update(1).gpu_threads(x);

        const Var u{"u"};
        const Var v{"v"};
        auto alpha_intm = alpha.update(0).rfactor({
            {rxo, u},
            {ryo, v},
        });

        // Zero-init at before iteration: alpha = max(alpha, value)
        alpha_intm.compute_at(alpha_intm.in(), u);

        // Iterate over rows via SIMD.
        alpha_intm.in().compute_at(alpha.in(), x_vo).gpu_threads(u);

        return;
    }
}

}  // namespace

HALIDE_REGISTER_GENERATOR(FPMEpry, fpm_epry)