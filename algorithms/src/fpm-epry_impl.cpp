#include "fpm-epry_generator.h"
#include "vars.hpp"

using namespace Halide;

using vars::i;
using vars::k;
using vars::x;
using vars::y;

namespace algorithms {
void
FPMEpry::setBounds() {
    const int W = tile_size;
    const int W2 = W * oversampling_factor;

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
}

void
FPMEpry::implementation() {
    const int W = tile_size;
    const int W2 = W * oversampling_factor;

    if (using_autoscheduler()) {
        k_offset.set_estimates({{0, 2}, {0, n_illumination}});

        high_res_prev.set_estimates({{0, 2}, {0, W2}, {0, W2}});

        high_res_new.set_estimates({{0, 2}, {0, W2}, {0, W2}});

        pupil_prev.set_estimates({{0, 2}, {0, W}, {0, W}});

        pupil_new.set_estimates({{0, 2}, {0, W}, {0, W}});

        return;
    }

    const auto target = get_target();
    assert(target.has_gpu_feature() && "GPU target required for manual implementation.");

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
}

}  // namespace algorithms