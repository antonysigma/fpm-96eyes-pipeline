#pragma once

namespace halide_cuda {

/** Obtain the Halide generated cuda context.
 *  Halide runtime created a separate 'sandbox' execution environment on the
 *  same GPU device to execute the halide compute pipelines.
 *  However, we wish to execute cuFFT in the same context as well for easier
 *  execution flow control.
 *  Reference:
 * https://github.com/halide/Halide/blob/7373eb9593d0accab5a47c6727e7f21fcf3f0e7b/src/runtime/cuda.cpp#L282
 */
struct Context {
    Context();

    ~Context();

   public:
    static const Context& getInstance() {
        static thread_local const Context singleton;
        return singleton;
    }
};

}  // namespace halide_cuda