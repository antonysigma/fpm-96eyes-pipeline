#include "cuda-context.h"

#include <cuda.h>

#include <cassert>
#include <mutex>
#include <thread>

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Cuda {

extern CUcontext context;
}
}  // namespace Internal
}  // namespace Runtime

}  // namespace Halide

namespace halide_cuda {

static std::mutex context_mutex;

Context::Context() {
    using Halide::Runtime::Internal::Cuda::context;

    std::lock_guard<std::mutex> lock(context_mutex);
    assert(context != nullptr && "Halide cuda context is null.");

    const auto error = cuCtxPushCurrent(context);
    assert(!error && "External stage unable to obtain Halide cuda context.");
}

Context::~Context() {
    CUcontext old;
    cuCtxPopCurrent(&old);
}

}  // namespace halide_cuda