#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/cub_device.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ColwiseReduceSum(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const T*                    x,
    T*                          y) {
    __shared__ typename BlockReduce<T>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, rows) {
        T val = 0;
        CUDA_2D_KERNEL_LOOP2(j, cols) {
            const int xi = i * cols + j;
#if __CUDA_ARCH__ >= 350
            val += __ldg(x + xi);
#else
            val += x[xi];
#endif
        }
        val = BlockReduce<T>(storage).Sum(val);
        if (threadIdx.x == 0) y[i] = val * scale;
    }
}

template<> __global__ void _ColwiseReduceSum<half>(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const half*                 x,
    half*                       y) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, rows) {
        float val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, cols) {
            const int xi = i * cols + j;
            val += __half2float(__ldg(x + xi));
        }
        val = BlockReduce<float>(storage).Sum(val);
        if (threadIdx.x == 0) {
            y[i] = __float2half(val * scale);
        }
    }
#endif
}

template <typename T>
__global__ void _RowwiseReduceSum(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const T*                    x,
    T*                          y) {
    __shared__ typename BlockReduce<T>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, cols) {
        T val = 0;
        CUDA_2D_KERNEL_LOOP2(j, rows) {
            const int xi = j * cols + i;
#if __CUDA_ARCH__ >= 350
            val += __ldg(x + xi);
#else
            val += x[xi];
#endif
        }
        val = BlockReduce<T>(storage).Sum(val);
        if (threadIdx.x == 0) y[i] = val * scale;
    }
}

template<> __global__ void _RowwiseReduceSum<half>(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const half*                 x,
    half*                       y) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, cols) {
        float val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, rows) {
            const int xi = j * cols + i;
            val += __half2float(__ldg(x + xi));
        }
        val = BlockReduce<float>(storage).Sum(val);
        if (threadIdx.x == 0) {
            y[i] = __float2half(val * scale);
        }
    }
#endif
}

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
    do {                                  \
        const auto n_copy = n;            \
        *q = n_copy / d;                  \
        *r = n_copy % d;                  \
    } while (0)

template <typename T>
__global__ void _GenericReduceSum(
    const int                   ndims,
    const int                   outer_dim,
    const int                   inner_dim,
    const int*                  x_strides,
    const int*                  y_dims,
    const float                 scale,
    const T*                    x,
    T*                          y) {
    __shared__ typename BlockReduce<T>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        T val = 0;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            int xi = 0, yi = i * inner_dim + j;
#pragma unroll
            for (int d = ndims - 1; d >= 0; --d) {
                int r;
#if __CUDA_ARCH__ >= 350
                FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), yi, &yi, &r);
                xi += r * __ldg(x_strides + d);
#else
                FIXED_DIVISOR_DIV_MOD(y_dims[d], yi, &yi, &r);
                xi += r * x_strides[d];
#endif
            }
#if __CUDA_ARCH__ >= 350
            val += __ldg(x + xi);
#else
            val += x[xi];
#endif
        }
        val = BlockReduce<T>(storage).Sum(val);
        if (threadIdx.x == 0) y[i] = val * scale;
    }
}

template <> __global__ void _GenericReduceSum<half>(
    const int                   ndims,
    const int                   outer_dim,
    const int                   inner_dim,
    const int*                  x_strides,
    const int*                  y_dims,
    const float                 scale,
    const half*                 x,
    half*                       y) {
#if __CUDA_ARCH__ >= 530
    __shared__ typename BlockReduce<float>::TempStorage storage;
    CUDA_2D_KERNEL_LOOP1(i, outer_dim) {
        float val = 0.f;
        CUDA_2D_KERNEL_LOOP2(j, inner_dim) {
            int xi = 0, yi = i * inner_dim + j;
#pragma unroll
            for (int d = ndims - 1; d >= 0; --d) {
                int r;
                FIXED_DIVISOR_DIV_MOD(__ldg(y_dims + d), yi, &yi, &r);
                xi += r * __ldg(x_strides + d);
            }
            val += __half2float(__ldg(x + xi));
        }
        val = BlockReduce<float>(storage).Sum(val);
        if (threadIdx.x == 0) {
            y[i] = __float2half(val * scale);
        }
    }
#endif
}

template <typename T>
void _ReduceSum(
    const int               ndims,
    const int*              dims,
    const int               naxes,
    const int*              axes,
    const float             scale,
    const T*                x,
    T*                      y,
    CUDAContext*            ctx) {
    vec32_t y_dimsV(dims, dims + ndims);
    for (int i = 0; i < naxes; ++i) y_dimsV[axes[i]] = 1;
    const int* x_dims = dims; const int* y_dims = y_dimsV.data();
    const int x_size = std::accumulate(x_dims,
        x_dims + ndims, 1, std::multiplies<int>());
    const int y_size = std::accumulate(y_dims,
        y_dims + ndims, 1, std::multiplies<int>());

    int rows, cols;

    /*! Case #1: Colwise Reduce */
    if (utils::IsColwiseReduce(
        ndims, x_dims, y_dims,
            &rows, &cols)) {
        _ColwiseReduceSum
            << < CUDA_2D_BLOCKS(rows), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            rows, cols, scale, x, y
        ); return;
    }

    /*! Case #2: Rowwise Reduce */
    if (utils::IsRowwiseReduce(
            ndims, x_dims, y_dims,
                &rows, &cols)) {
        _RowwiseReduceSum
            << < CUDA_2D_BLOCKS(cols), CUDA_THREADS,
                 0, ctx->cuda_stream() >> >(
            rows, cols, scale, x, y
        ); return;
    }

    /*! Case #3: Generic Reduce */
    vec32_t axesT(ndims), stridesT(ndims), dimsT(ndims);

    utils::ComputeTransposedAxesForReduce(
        ndims, naxes, axes,
        axesT.data()
    );

    utils::ComputeTransposedStrides(
        ndims, dims,
        axesT.data(),
        stridesT.data()
    );

    int outer_dim = 1, inner_dim = 1;
    const int pivot = ndims - naxes;
    for (int i = 0; i < pivot; ++i) outer_dim *= dims[axesT[i]];
    for (int i = pivot; i < ndims; ++i) inner_dim *= dims[axesT[i]];
    for (int i = 0; i < ndims; ++i) dimsT[i] = dims[axesT[i]];
    
    const size_t dbytes = sizeof(int) * ndims;
    int* XSS = (int*)ctx->New(dbytes), *YDS = (int*)ctx->New(dbytes);
    ctx->Memcpy<CUDAContext, CPUContext>(dbytes, XSS, stridesT.data());
    ctx->Memcpy<CUDAContext, CPUContext>(dbytes, YDS, dimsT.data());

    _GenericReduceSum
        << < CUDA_2D_BLOCKS(outer_dim), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        ndims, outer_dim, inner_dim,
        XSS, YDS, scale, x, y
    );

    ctx->FinishDeviceCompution();
    ctx->Delete(XSS); ctx->Delete(YDS);

}

#define DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(T) \
    template <> void ReduceSum<T, CUDAContext>( \
        const int               num_dims, \
        const int*              dims, \
        const int               num_axes, \
        const int*              axes, \
        const float             scale, \
        const T*                x, \
        T*                      y, \
        CUDAContext*            ctx) { \
        _ReduceSum( \
            num_dims, dims, \
            num_axes, axes, \
            scale, x, y, ctx \
        ); \
    }

DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int8_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(uint8_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int64_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(float);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(double);

template <> void ReduceSum<float16, CUDAContext>(
    const int               num_dims,
    const int*              dims,
    const int               num_axes,
    const int*              axes,
    const float             scale,
    const float16*          x,
    float16*                y,
    CUDAContext*            ctx) {
    _ReduceSum(
        num_dims, dims,
        num_axes, axes,
        scale,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y), ctx
    );
}

/* <T = ?, Device = CUDA> */

template <typename T>
__global__ void _ReduceSumGrad(
    const int               nthreads,
    const int               ndim,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        int yi = 0, tmp = xi;
#pragma unroll
        for (int d = ndim - 1; d >= 0; --d) {
            int r;
#if __CUDA_ARCH__ >= 350
            FIXED_DIVISOR_DIV_MOD(__ldg(x_dims + d), tmp, &tmp, &r);
            yi += (r % __ldg(y_dims + d)) * __ldg(y_strides + d);
#else
            FIXED_DIVISOR_DIV_MOD(x_dims[d], tmp, &tmp, &r);
            yi += (r % y_dims[d]) * y_strides[d];
#endif
        }
#if __CUDA_ARCH__ >= 350
        dx[xi] = __ldg(dy + yi) * scale;
#else
        dx[xi] = dy[yi] * scale;
#endif
    }
}

template <> __global__ void _ReduceSumGrad<half>(
    const int               nthreads,
    const int               ndim,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const half*             dy,
    half*                   dx) {
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
#if __CUDA_ARCH__ >= 530
        int yi = 0, tmp = xi;
#pragma unroll
        for (int d = ndim - 1; d >= 0; --d) {
            int r;
            FIXED_DIVISOR_DIV_MOD(__ldg(x_dims + d), tmp, &tmp, &r);
            yi += r % __ldg(y_dims + d) * __ldg(y_strides + d);
        }
        dx[xi] = __float2half(
            __half2float(
                __ldg(dy + yi)
            ) * scale
        );
#endif
    }
}

/* Kernel Launchers */

#define DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(T) \
    template<> void ReduceSumGrad<T, CUDAContext>( \
        const int               count, \
        const int               ndim, \
        const int*              x_dims, \
        const int*              y_dims, \
        const int*              y_strides, \
        const float             scale, \
        const T*                dy, \
        T*                      dx, \
        CUDAContext*            ctx) { \
        _ReduceSumGrad \
            << < CUDA_BLOCKS(count), CUDA_THREADS, \
                 0, ctx->cuda_stream() >> >( \
            count, ndim, x_dims, \
            y_dims, y_strides, \
            scale, dy, dx \
        ); \
    }

DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(int);
DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(float);
DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(double);

template<> void ReduceSumGrad<float16, CUDAContext>(
    const int               count,
    const int               ndim,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const float16*          dy,
    float16*                dx,
    CUDAContext*            ctx) {
    _ReduceSumGrad
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >(
        count, ndim, x_dims,
        y_dims, y_strides,
        scale,
        reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx)
    );
}

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_REDUCE_SUM_KERNEL_LAUNCHER
#undef DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA