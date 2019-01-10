#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! Arange <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        y[idx] = start + idx * step;
    }
}

#define DEFINE_ARANGE_KERNEL_LAUNCHER(T) \
    template <> void Arange<T, CUDAContext>( \
        const int               count, \
        const int               start, \
        const int               step, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Arange<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, start, step, y); \
    }

DEFINE_ARANGE_KERNEL_LAUNCHER(int8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(uint8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(int);
DEFINE_ARANGE_KERNEL_LAUNCHER(int64_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(float);
DEFINE_ARANGE_KERNEL_LAUNCHER(double);

/*! Arange <T = float16, Device = CUDA> */

__global__ void _ArangeHalf(
    const int               count,
    const int               start,
    const int               step,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        y[idx] = __float2half((float)(start + idx * step));
#endif
    }
}

template <> void Arange<float16, CUDAContext>(
    const int               count,
    const int               start,
    const int               step,
    float16*                y,
    CUDAContext*            ctx) {
    _ArangeHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, start, step, reinterpret_cast<half*>(y));
}

#undef DEFINE_ARANGE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA