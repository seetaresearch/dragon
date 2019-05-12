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
__global__ void _Crop(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                x,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        int xi = 0, tmp = yi;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            xi += (r + __ldg(starts + d)) * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            xi += (r + starts[d]) * x_strides[d];
#endif
        }
        y[yi] = x[xi];
    }
}

/* <T = ?, Device = CUDA> */

template<typename T>
__global__ void _CropGrad(
    const int               nthreads,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        int xi = 0, tmp = yi;
#pragma unroll
        for (int d = ndims - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), tmp, &tmp, &r);
            xi += (r + __ldg(starts + d)) * __ldg(x_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(y_dims[d], tmp, &tmp, &r);
            xi += (r + starts[d]) * x_strides[d];
#endif
        }
        dx[xi] = dy[yi];
    }
}

/* Kernel Launchers */

#define DEFINE_CROP_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CUDAContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              starts, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _##name \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, ndims, \
            x_strides, y_dims, \
            starts, x, y \
        ); \
    }

DEFINE_CROP_KERNEL_LAUNCHER(Crop, bool);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int8_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, uint8_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int64_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, float16);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, float);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, double);

DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, bool);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int8_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, uint8_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int64_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, float16);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, float);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, double);

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_CROP_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA