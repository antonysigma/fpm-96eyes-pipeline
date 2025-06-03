#pragma once
#include <iostream>
#include <stdexcept>

#include "cufft.h"

#ifdef NDEBUG
#define gpuErrchk(ans) ans
#define cufftErrchk(ans, msg) ans
#else
#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
#define cufftErrchk(ans, msg) cufftAssert((ans), msg, __FILE__, __LINE__)
#endif

#define gpuReschk(ans) gpuAssert((ans), __FILE__, __LINE__)

namespace cuda_assert {
constexpr auto RETURN_CODE = 1;
constexpr bool ABORT = false;

}  // namespace cuda_assert

/** Translate cufft error code */
static const char *
cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        default:
            return "<unknown>";
    }
}

inline void
cufftAssert(cufftResult rc, const char *msg, const char *file, int line) {
    if (rc == CUFFT_SUCCESS) return;

    char buffer[100];
    sprintf(buffer, "cufftAssert: %s - %s (%s:%d)\n", msg, cufftGetErrorString(rc), file, line);

    if (cuda_assert::ABORT) {
        std::cerr << buffer;
        exit(cuda_assert::RETURN_CODE);
        return;
    }

    throw std::runtime_error(buffer);
}
