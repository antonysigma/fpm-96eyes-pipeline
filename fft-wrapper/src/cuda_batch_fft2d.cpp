#include "cuda_batch_fft2d.h"

#include "cuda_assert.h"

CudaBatchFft2d::CudaBatchFft2d(unsigned batch, int width, int height) {
    if (batch > 1) {
        // Multi-plane 2D FFT
        const int rank[] = {height, width};

        cufftErrchk(cufftPlanMany(&_plan, 2, const_cast<int*>(rank), nullptr, 0, 0, nullptr, 0, 0,
                                  CUFFT_C2C, batch),
                    "cufftPlanMany");
    } else {
        // Single plane 2D FFT
        cufftErrchk(cufftPlan2d(&_plan, height, width, CUFFT_C2C), "cufftPlan2d");
    }
}

CudaBatchFft2d::CudaBatchFft2d() : _plan{0} {}

CudaBatchFft2d::~CudaBatchFft2d() {
    if (_plan == 0) return;
    // cufftErrchk(cufftDestroy(_plan), "cufftDestroy");
    //_plan = 0;
}

void
CudaBatchFft2d::dft2(const float2_t* src, float2_t* dst, cudaStream_t s) const {
    cufftErrchk(cufftSetStream(_plan, s), "cufftSetStream");
    cufftErrchk(cufftExecC2C(_plan, (cufftComplex*)src, reinterpret_cast<cufftComplex*>(dst),
                             CUFFT_FORWARD),
                "cufftExecF2Z");
}

void
CudaBatchFft2d::idft2(const float2_t* src, float2_t* dst, cudaStream_t s) const {
    // std::cerr << "idft\n";
    cufftErrchk(cufftSetStream(_plan, s), "cufftSetStream");
    cufftErrchk(cufftExecC2C(_plan, (cufftComplex*)src, reinterpret_cast<cufftComplex*>(dst),
                             CUFFT_INVERSE),
                "cufftExecZ2D");
}

bool
CudaBatchFft2d::isInitialized() const {
    return !(_plan == 0);
}