#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Arange(
    const int               nthreads,
    const int               start,
    const int               step,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = start + i * step;
    }
}

/* Kernel Launchers */

#define DEFINE_ARANGE_KERNEL_LAUNCHER(T) \
    template <> void Arange<T, CUDAContext>( \
        const int               count, \
        const int               start, \
        const int               step, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Arange \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, start, step, y \
        ); \
    }

DEFINE_ARANGE_KERNEL_LAUNCHER(int8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(uint8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(int);
DEFINE_ARANGE_KERNEL_LAUNCHER(int64_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(float);
DEFINE_ARANGE_KERNEL_LAUNCHER(double);

/*! Arange <T = float16, Device = CUDA> */

template<> __global__ void _Arange<half>(
    const int               nthreads,
    const int               start,
    const int               step,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        y[i] = __float2half((float)(start + i * step));
#endif
    }
}

template <> void Arange<float16, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    float16*                y,
    CUDAContext*            ctx) {
    _Arange
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, start, step,
        reinterpret_cast<half*>(y)
    );
}

#undef DEFINE_ARANGE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA