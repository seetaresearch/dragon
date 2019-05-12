#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! GradientTwoSum <T = ?, Device = CUDA> */

template <typename T>
__global__ void _GradientTwoSum(
    const int               nthreads,
    const T*                dy1,
    const T*                dy2,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
        dx[i] += (dy1[i] + dy2[i]);
    }
}

/*! GradientTwoSum <T = float16, Device = CUDA> */

template <> __global__ void _GradientTwoSum<half>(
    const int               nthreads,
    const half*             dy1,
    const half*             dy2,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        dx[i] = __hadd(
            dx[i],
            __hadd(dy1[i], dy2[i])
        );
#endif
    }
}

template <> __global__ void _GradientTwoSum<half2>(
    const int               nthreads,
    const half2*            dy1,
    const half2*            dy2,
    half2*                   dx) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 530
        dx[i] = __hadd2(
            dx[i],
            __hadd2(dy1[i], dy2[i])
        );
#endif
    }
}

/*! Kernel Launchers */

#define DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(T) \
    template <> void GradientTwoSum<T, CUDAContext>( \
        const int               count, \
        const T*                dy1, \
        const T*                dy2, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        _GradientTwoSum \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, dy1, dy2, dx \
        ); \
    }

DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int8_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(uint8_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int64_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(float);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(double);

template <> void GradientTwoSum<float16, CUDAContext>(
    const int               count,
    const float16*          dy1,
    const float16*          dy2,
    float16*                dx,
    CUDAContext*            ctx) {
    if ((count & 1) == 0) {
        _GradientTwoSum
            << < CUDA_BLOCKS(count >> 2), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            count >> 2,
            reinterpret_cast<const half2*>(dy1),
            reinterpret_cast<const half2*>(dy2),
            reinterpret_cast<half2*>(dx)
        );
    } else {
        _GradientTwoSum
            << < CUDA_BLOCKS(count), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            count,
            reinterpret_cast<const half*>(dy1),
            reinterpret_cast<const half*>(dy2),
            reinterpret_cast<half*>(dx)
        );
    }
}

#undef DEFINE_GRAD_SUM2_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon

#endif  // WITH_CUDA