#pragma once
#include <complex>

#include "cufft.h"

#pragma pack(push, 8)
struct float2_t {
    float re;
    float im;
};
#pragma pack(pop)

class CudaBatchFft2d {
    /** Pre-allocated memory for the cuFFT routine */
    cufftHandle _plan = 0;

   public:
    /** Calculate and allocate memory for the cuFFT routine
     *
     * @param[in] batch number of image segments
     * @param[in] height height of the image
     * @param[in] width width of the image
     */
    CudaBatchFft2d(unsigned batch, int width, int height);

    ~CudaBatchFft2d();

    CudaBatchFft2d();
    CudaBatchFft2d(CudaBatchFft2d&) = delete;
    CudaBatchFft2d(const CudaBatchFft2d&) = delete;
    CudaBatchFft2d(CudaBatchFft2d&&) = default;

    /** Forward two-dimensional FFT.
     *
     * @param[in] src source image
     * @param[out] dst Fourier representation of the image
     * @param[in] s ID of the CUDA stream
     */
    void dft2(const float2_t* src, float2_t* dst, cudaStream_t s = 0) const;

    /** Backward two-dimensional FFT.
     *
     * @param[in] src Fourier representation of the image
     * @param[out] dst output image
     * @param[in] s ID of the CUDA stream
     * @warning Unlike inverse FFT, backward FFT comes with a scaling factor.
     *      To find the inverse, divide the result by N*N.
     */
    void idft2(const float2_t* src, float2_t* dst, cudaStream_t s = 0) const;

    bool isInitialized() const;
};
