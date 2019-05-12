#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/eigen_utils.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
    do {                                  \
        const auto n_copy = n;            \
        *q = n_copy / d;                  \
        *r = n_copy % d;                  \
    } while (0)

/* <T = ?, Device = CPU> */

template <typename T>
void _ColwiseReduceSum(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const T*                    x,
    T*                          y) {
    for (int i = 0; i < rows; ++i) {
        T val = ConstEigenVectorArrayMap<T>(x, cols).sum();
        y[i] = val * scale;
    }
}

template <typename T>
void _RowwiseReduceSum(
    const int                   rows,
    const int                   cols,
    const float                 scale,
    const T*                    x,
    T*                          y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(cols))
#endif
    for (int i = 0; i < cols; ++i) {
        T val = 0;
        for (int j = 0; j < rows; ++j) {
            val += x[j * cols + i];
        }
        y[i] = val * scale;
    }
}

template <typename T>
void _GenericReduceSum(
    const int                   outer_dim,
    const int                   inner_dim,
    const int                   ndims,
    const int*                  x_strides,
    const int*                  y_dims,
    const float                 scale,
    const T*                    x,
    T*                          y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(outer_dim))
#endif
    for (int i = 0; i < outer_dim; ++i) {
        T val = 0;
        int xi, yi, r;
        for (int j = 0; j < inner_dim; ++j) {
            xi = 0; yi = i * inner_dim + j;
            for (int d = ndims - 1; d >= 0; --d) {
                FIXED_DIVISOR_DIV_MOD(y_dims[d], yi, &yi, &r);
                xi += r * x_strides[d];
            }
            val += x[xi];
        }
        y[i] = val;
    }
}

template <typename T>
void _GenericReduceSumLauncher(
    const int                   outer_dim,
    const int                   inner_dim,
    const int                   ndims,
    const int*                  dims,
    const int*                  axes,
    const float                 scale,
    const T*                    x,
    T*                          y) {
    vec32_t x_strides(ndims), y_dims(ndims);
    utils::ComputeTransposedStrides(
        ndims, dims, axes,
        x_strides.data()
    );
    for (int i = 0; i < ndims; ++i)
        y_dims[i] = dims[axes[i]];
    _GenericReduceSum(
        outer_dim,
        inner_dim,
        ndims,
        x_strides.data(),
        y_dims.data(),
        scale,
        x, y
    );
}

template <typename T>
void _ReduceSum(
    const int               num_dims,
    const int*              dims,
    const int               num_axes,
    const int*              axes,
    const float             scale,
    const T*                x,
    T*                      y) {
    vec32_t y_dims_V(dims, dims + num_dims);
    for (int i = 0; i < num_axes; ++i) y_dims_V[axes[i]] = 1;
    const int* x_dims = dims;
    const int* y_dims = y_dims_V.data();
    const int x_size = std::accumulate(x_dims,
        x_dims + num_dims, 1, std::multiplies<int>());
    const int y_size = std::accumulate(y_dims,
        y_dims + num_dims, 1, std::multiplies<int>());
    int rows, cols;
    // Case #1: Colwise Reduce
    if (utils::IsColwiseReduce(
            num_dims, x_dims, y_dims,
                &rows, &cols)) {
        _ColwiseReduceSum(
            rows, cols, scale, x, y
        ); return;
    }
    // Case #2: Rowwise Reduce
    if (utils::IsRowwiseReduce(
            num_dims, x_dims, y_dims, 
                &rows, &cols)) {
        _RowwiseReduceSum(
            rows, cols, scale, x, y
        ); return;
    }
    // Case #3: Generic Reduce
    vec32_t transpose_axes(num_dims);
    utils::ComputeTransposedAxesForReduce(
        num_dims, num_axes, axes,
        transpose_axes.data()
    );
    const int pivot = num_dims - num_axes;
    int outer_dim = 1, inner_dim = 1;
    for (int i = 0; i < pivot; ++i)
        outer_dim *= dims[transpose_axes[i]];
    for (int i = pivot; i < num_dims; ++i)
        inner_dim *= dims[transpose_axes[i]];
    _GenericReduceSumLauncher(
        outer_dim,
        inner_dim,
        num_dims,
        dims,
        transpose_axes.data(),
        scale,
        x, y
    );
}

/* Kernel Launchers */

#define DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(T) \
    template <> void ReduceSum<T, CPUContext>( \
        const int               num_dims, \
        const int*              dims, \
        const int               num_axes, \
        const int*              axes, \
        const float             scale, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _ReduceSum( \
            num_dims, dims, \
            num_axes, axes, \
            scale, x, y \
        ); \
    }

DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int8_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(uint8_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(int64_t);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(float);
DEFINE_REDUCE_SUM_KERNEL_LAUNCHER(double);

/*! ReduceSum <T = float16, Device = CPU> */

template <> void ReduceSum<float16, CPUContext>(
    const int               num_dims,
    const int*              dims,
    const int               num_axes,
    const int*              axes,
    const float             scale,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ReduceSumGrad <T = ?, Device = CPU> */

template <typename T>
void _ReduceSumGrad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const T*                dy,
    T*                      dx) {
    vec32_t index(ndims, 0); int yi;
    for (int xi = 0; xi < nthreads; ++xi) {
        yi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            yi += (
                index[d] % y_dims[d]
            ) * y_strides[d];
        }
        dx[xi] = dy[yi] * scale;
        utils::IncreaseIndexInDims(
            ndims, x_dims, index.data()
        );
    }
}

/*! Kernel Launchers */

#define DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER(T) \
    template<> void ReduceSumGrad<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              y_dims, \
        const int*              y_strides, \
        const float             scale, \
        const T*                dy, \
        T*                      dx, \
        CPUContext*             ctx) { \
        _ReduceSumGrad( \
            count, ndims, x_dims, \
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

template<> void ReduceSumGrad<float16, CPUContext>(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              y_dims,
    const int*              y_strides,
    const float             scale,
    const float16*          dy,
    float16*                dx,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_REDUCE_SUM_KERNEL_LAUNCHER
#undef DEFINE_REDUCE_SUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon