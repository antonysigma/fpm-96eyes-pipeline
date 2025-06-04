#pragma once

#include <Halide.h>
#include <complex.h>

#include <vector>

namespace algorithms {
using namespace Halide;

enum fpm_mode_t : int32_t { AUTO_BRIGHTNESS = 0, PUPIL_RECOVERY = 1 };

/** Simulate the low-resolution image */
class FPMEpry : public Generator<FPMEpry> {
    Input<Buffer<const float, 3>> low_res{"low_res"};
    Input<Buffer<float, 3>> high_res_prev{"high_res_prev"};
    Input<Buffer<float, 3>> pupil_prev{"pupil_prev"};

    Input<Buffer<const int32_t, 2>> k_offset{"k_offset"};

    Output<Buffer<float, 3>> high_res_new{"high_res_new"};
    Output<Buffer<float, 3>> pupil_new{"pupil_new"};

    GeneratorParam<uint32_t> n_illumination{"n_illumination", 3, 9, 49};
    GeneratorParam<int32_t> fpm_mode{"fpm_mode", PUPIL_RECOVERY, AUTO_BRIGHTNESS, PUPIL_RECOVERY};
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

    void setBounds();
    void design();
    void implementation();

   public:
    static constexpr auto oversampling_factor = 2;

    inline void generate() { design(); }
    inline void schedule() {
        setBounds();
        implementation();
    }
};

}  // namespace algorithms