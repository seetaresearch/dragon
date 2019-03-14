#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

/*! Transpose <T = ?, Device = CUDA> */

template <typename T>
__global__ void _Transpose(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
       int x_idx = 0, tmp = y_idx;
#pragma unroll
       for (int d = ndims - 1; d >= 0; --d) {
           int r;
#if __CUDA_ARCH__ >= 350
           FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
           x_idx += r * __ldg(x_strides + d);
#else
           FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
           x_idx += r * x_strides[d];
#endif
       }
       y[y_idx] = x[x_idx];
   }
}

/*! TransposeGrad <T = ?, Device = CUDA> */

template <typename T>
__global__ void _TransposeGrad(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int x_idx = 0, tmp = y_idx;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            x_idx += r * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            x_idx += r * x_strides[d];
#endif
        }
        dx[x_idx] = dy[y_idx];
    }
}

/*! Kernel Launchers */

#define DEFINE_TRANSPOSE_KERNEL_LAUNCHER(name, T) \
    template <> void name<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, ndims, x_strides, y_dims, x, y); \
    }

DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, bool);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, uint8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, int64_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, float16);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, float);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(Transpose, double);

DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, bool);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, uint8_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, int64_t);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, float16);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, float);
DEFINE_TRANSPOSE_KERNEL_LAUNCHER(TransposeGrad, double);

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_TRANSPOSE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA