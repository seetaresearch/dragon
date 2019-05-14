#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
    do {                                  \
        const auto n_copy = n;            \
        *q = n_copy / d;                  \
        *r = n_copy % d;                  \
    } while (0)

/* <T = ?, Device = CUDA> */

template<typename T>
__global__ void _Assign(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              y_strides,
    const int*              starts,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        int yi = 0, tmp = xi;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(x_dims + d), tmp, &tmp, &r);
            yi += (r + __ldg(starts + d)) * __ldg(y_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(x_dims[d], tmp, &tmp, &r);
            yi += (r + starts[d]) * y_strides[d];
#endif
        }
        y[yi] = x[xi];
    }
}

/* Kernel Launchers */

#define DEFINE_ASSIGN_KERNEL_LAUNCHER(T) \
    template<> void Assign<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              y_strides, \
        const int*              starts, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _Assign \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, \
            ndims, \
            x_dims, \
            y_strides, \
            starts, \
            x, y \
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

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_ASSIGN_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA