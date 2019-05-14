#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template<typename T>
__global__ void _MaskedAssign(
    const int               nthreads,
    const uint8_t*          mask,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = mask[i] ? x[i] : y[i];
    }
}

/* Kernel Launchers */

#define DEFINE_ASSIGN_KERNEL_LAUNCHER(T) \
    template<> void MaskedAssign<T, CUDAContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _MaskedAssign \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, mask, x, y \
        ); \
    }

DEFINE_ASSIGN_KERNEL_LAUNCHER(bool);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int8_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(uint8_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int64_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(float16);
DEFINE_ASSIGN_KERNEL_LAUNCHER(float);
DEFINE_ASSIGN_KERNEL_LAUNCHER(double);

#undef DEFINE_ASSIGN_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA