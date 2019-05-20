#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template<typename T>
__global__ void _Where(
    const int               nthreads,
    const uint8_t*          mask,
    const T*                a,
    const T*                b,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        y[i] = mask[i] ? a[i] : b[i];
    }
}

template <typename T>
__global__ void _WhereGrad(
    const int               nthreads,
    const uint8_t*          mask,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        da[i] = mask[i] ? dy[i] : kZero;
        db[i] = mask[i] ? kZero : dy[i];
    }
}

template<> __global__ void _WhereGrad<half>(
    const int               nthreads,
    const uint8_t*          mask,
    const half*             dy,
    half*                   da,
    half*                   db) {
#if __CUDA_ARCH__ >= 530
    const half kZero = __float2half(0.f);
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        const bool dy_to_da = mask[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
#endif
}

/* Kernel Launchers */

#define DEFINE_WHERE_KERNEL_LAUNCHER(T) \
    template<> void Where<T, CUDAContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Where \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, mask, a, b, y \
        ); \
    }

#define DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(T) \
    template <> void WhereGrad<T, CUDAContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                dy, \
        T*                      da, \
        T*                      db, \
        CUDAContext*            ctx) { \
        _WhereGrad \
            <<< CUDA_BLOCKS(count), CUDA_THREADS, \
                0, ctx->cuda_stream() >>>( \
            count, mask, dy, da, db \
        ); \
    }

DEFINE_WHERE_KERNEL_LAUNCHER(bool);
DEFINE_WHERE_KERNEL_LAUNCHER(int8_t);
DEFINE_WHERE_KERNEL_LAUNCHER(uint8_t);
DEFINE_WHERE_KERNEL_LAUNCHER(int);
DEFINE_WHERE_KERNEL_LAUNCHER(int64_t);
DEFINE_WHERE_KERNEL_LAUNCHER(float16);
DEFINE_WHERE_KERNEL_LAUNCHER(float);
DEFINE_WHERE_KERNEL_LAUNCHER(double);

DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(bool);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(double);

template <> void WhereGrad<float16, CUDAContext>(
    const int               count,
    const uint8_t*          mask,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CUDAContext*            ctx) {
    _WhereGrad
        <<< CUDA_BLOCKS(count), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        count,
        mask,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(da),
        reinterpret_cast<half*>(db)
    );
}

#undef DEFINE_WHERE_KERNEL_LAUNCHER
#undef DEFINE_WHERE_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA