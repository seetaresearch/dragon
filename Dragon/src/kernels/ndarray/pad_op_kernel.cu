#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

/*! ConstPad <T = ?, Device = CUDA> */

template<typename T>
__global__ void _ConstPad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T                 value,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int x_idx = 0, tmp = y_idx, d;
#pragma unroll
        for (d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            r -= __ldg(l_pads + d); if (r < 0 || r >= __ldg(x_dims + d)) break;
            x_idx += r * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            r -= l_pads[d]; if (r < 0 || r >= x_dims[d]) break;
            x_idx += r * x_strides[d];
#endif
        }
        y[y_idx] = d >= 0 ? value : x[x_idx];
    }
}

/*! ReflectPad <T = ?, Device = CUDA> */

template<typename T>
__global__ void _ReflectPad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int x_idx = 0, tmp = y_idx;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            r -= __ldg(l_pads + d);
            r = max(r, -r);
            r = min(r, 2 * __ldg(x_dims + d) - r - 2);
            x_idx += r * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            r -= l_pads[d];
            r = max(r, -r);
            r = min(r, 2 * x_dims[d] - r - 2);
            x_idx += r * x_strides[d];
#endif
        }
        y[y_idx] = x[x_idx];
    }
}

/*! EdgePad <T = ?, Device = CUDA> */

template<typename T>
__global__ void _EdgePad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int x_idx = 0, tmp = y_idx;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            r = min(__ldg(x_dims + d) - 1, max(r - __ldg(l_pads + d), 0));
            x_idx += r * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            r = min(x_dims[d] - 1, max(r - l_pads[d], 0));
            x_idx += r * x_strides[d];
#endif
        }
        y[y_idx] = x[x_idx];
    }
}

/*! Kernel Launchers */

#define DEFINE_CONST_PAD_KERNEL_LAUNCHER(T) \
    template<> void ConstPad<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              l_pads, \
        const float             value, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _ConstPad<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, ndims, x_dims, x_strides, y_dims, l_pads, \
                cast::to<T>(value), x, y); \
    }

#define DEFINE_PAD_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              l_pads, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name<T> \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> > \
            (count, ndims, x_dims, x_strides, \
                y_dims, l_pads, x, y); \
    }

DEFINE_CONST_PAD_KERNEL_LAUNCHER(bool);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int8_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int64_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(float16);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(float);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(double);

DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, bool);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int8_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, uint8_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int64_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, float16);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, float);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, double);

DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, bool);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int8_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, uint8_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int64_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, float16);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, float);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, double);

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_PAD_KERNEL_LAUNCHER
#undef DEFINE_CONST_PAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA