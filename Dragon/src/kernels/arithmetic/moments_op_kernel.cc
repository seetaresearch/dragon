#include "utils/op_kernel.h"
#include "utils/math_utils.h"
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

/* <Tx = ?, Ty = ?, Device = CPU> */

template <typename Tx, typename Ty>
void _ColwiseMoments(
    const int                   rows,
    const int                   cols,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    const Ty scale = Ty(1) / (Ty)cols;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
    for (int i = 0; i < rows; ++i) {
        Tx x_val; Ty m_val = 0, v_val = 0, mu;
        for (int j = 0; j < cols; ++j) {
            x_val = x[i * cols + j];
            m_val += x_val; v_val += x_val * x_val;
        }
        mean[i] = mu = m_val * scale;
        var[i] = v_val * scale - mu * mu;
    }
}

template <typename Tx, typename Ty>
void _RowwiseMoments(
    const int                   rows,
    const int                   cols,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    const Ty scale = Ty(1) / (Ty)rows;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(cols))
#endif
    for (int i = 0; i < cols; ++i) {
        Tx x_val; Ty m_val = 0, v_val = 0, mu;
        for (int j = 0; j < rows; ++j) {
            x_val = x[j * cols + i];
            m_val += x_val; v_val += x_val * x_val;
        }
        mean[i] = mu = m_val * scale;
        var[i] = v_val * scale - mu * mu;
    }
}

template <typename Tx, typename Ty>
void _GenericMoments(
    const int                   outer_dim,
    const int                   inner_dim,
    const int                   ndim,
    const int*                  x_strides,
    const int*                  y_dims,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    const Ty scale = Ty(1) / (Ty)inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(outer_dim))
#endif
    for (int i = 0; i < outer_dim; ++i) {
        Tx x_val;
        Ty m_val = 0, v_val = 0, mu;
        int x_idx, y_idx, r;
        for (int j = 0; j < inner_dim; ++j) {
            x_idx = 0; y_idx = i * inner_dim + j;
            for (int d = ndim - 1; d >= 0; --d) {
                FIXED_DIVISOR_DIV_MOD(y_dims[d], y_idx, &y_idx, &r);
                x_idx += r * x_strides[d];
            }
            x_val = x[x_idx];
            m_val += x_val; v_val += x_val * x_val;
        }
        mean[i] = mu = m_val * scale;
        var[i] = v_val * scale - mu * mu;
    }
}

template <typename Tx, typename Ty>
void _GenericMomentsLauncher(
    const int                   outer_dim,
    const int                   inner_dim,
    const int                   ndim,
    const int*                  dims,
    const int*                  axes,
    const Tx*                   x,
    Ty*                         mean,
    Ty*                         var) {
    vec32_t x_strides(ndim);
    vec32_t y_dims(ndim);
    utils::ComputeTransposedStrides(
        ndim, dims, axes,
        x_strides.data()
    );
    for (int i = 0; i < ndim; ++i)
        y_dims[i] = dims[axes[i]];
    _GenericMoments(
        outer_dim, inner_dim, ndim,
        x_strides.data(), y_dims.data(), 
        x, mean, var
    );
}

template <typename Tx, typename Ty>
void _Moments(
    const int               num_dims,
    const int*              dims,
    const int               num_axes,
    const int*              axes,
    const Tx*               x,
    Ty*                     mean,
    Ty*                     var,
    CPUContext*             ctx) {
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
        _ColwiseMoments(
            rows, cols, x, mean, var
        ); return;
    }
    // Case #2: Rowwise Reduce
    if (utils::IsRowwiseReduce(
            num_dims, x_dims, y_dims,
                &rows, &cols)) {
        _RowwiseMoments(
            rows, cols, x, mean, var
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
    _GenericMomentsLauncher(
        outer_dim, inner_dim,
        num_dims, dims,
        transpose_axes.data(),
        x, mean, var
    );
}

/* Kernel Launchers */

#define DEFINE_MOMENTS_KERNEL_LAUNCHER(Tx, Ty) \
    template <> void Moments<Tx, Ty, CPUContext>( \
        const int               num_dims, \
        const int*              dims, \
        const int               num_axes, \
        const int*              axes, \
        const Tx*               x, \
        Ty*                     mean, \
        Ty*                     var, \
        CPUContext*             ctx) { \
        _Moments( \
            num_dims, dims, \
            num_axes, axes, \
            x, mean, var, ctx \
        ); \
    }

DEFINE_MOMENTS_KERNEL_LAUNCHER(int8_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(int, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(int64_t, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(float, float);
DEFINE_MOMENTS_KERNEL_LAUNCHER(double, double);

template <> void Moments<float16, float, CPUContext>(
    const int               num_dims,
    const int*              dims,
    const int               num_axes,
    const int*              axes,
    const float16*          x,
    float*                  mean,
    float*                  var,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef FIXED_DIVISOR_DIV_MOD
#undef DEFINE_MOMENTS_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon