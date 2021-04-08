#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _RowwiseMoments(
    const int rows,
    const int cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(rows);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(cols))
#endif
  for (int i = 0; i < cols; ++i) {
    AccT x_val, m_val = AccT(0), v_val = AccT(0);
    for (int j = 0; j < rows; ++j) {
      x_val = convert::To<AccT>(x[j * cols + i]);
      m_val += x_val;
      v_val += x_val * x_val;
    }
    m_val *= scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void _ColwiseMoments(
    const int rows,
    const int cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(cols);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    AccT x_val, m_val = AccT(0), v_val = AccT(0);
    for (int j = 0; j < cols; ++j) {
      x_val = convert::To<AccT>(x[i * cols + j]);
      m_val += x_val;
      v_val += x_val * x_val;
    }
    m_val *= scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void _GenericMoments(
    const int rows,
    const int cols,
    const int num_dims,
    const int* x_dims,
    const int* x_strides,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(cols);
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    AccT x_val, m_val = AccT(0), v_val = AccT(0);
    int xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      x_val = convert::To<AccT>(x[xi]);
      m_val += x_val;
      v_val += x_val * x_val;
    }
    m_val *= scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void _Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CPUContext* ctx) {
  int rows, cols;
  vec32_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    _RowwiseMoments(rows, cols, x, mean, var);
    return;
  }
  if (math::utils::IsColwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    _ColwiseMoments(rows, cols, x, mean, var);
    return;
  }
  vec32_t transpose_axes(num_dims);
  vec32_t transpose_strides(num_dims);
  vec32_t transpose_dims(num_dims);
  math::utils::TransposeAxesForReduce(
      num_dims, num_axes, axes, transpose_axes.data());
  math::utils::ComputeTransposeStrides(
      num_dims, dims, transpose_axes.data(), transpose_strides.data());
  rows = cols = 1;
  const int pivot = num_dims - num_axes;
  for (int i = 0; i < pivot; ++i) {
    rows *= dims[transpose_axes[i]];
  }
  for (int i = pivot; i < num_dims; ++i) {
    cols *= dims[transpose_axes[i]];
  }
  for (int i = 0; i < num_dims; ++i) {
    transpose_dims[i] = dims[transpose_axes[i]];
  }
  _GenericMoments(
      rows,
      cols,
      num_dims,
      transpose_dims.data(),
      transpose_strides.data(),
      x,
      mean,
      var);
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                          \
  template <>                                                    \
  void Moments<T, AccT, CPUContext>(                             \
      const int num_dims,                                        \
      const int* dims,                                           \
      const int num_axes,                                        \
      const int* axes,                                           \
      const T* x,                                                \
      AccT* mean,                                                \
      AccT* var,                                                 \
      CPUContext* ctx) {                                         \
    _Moments(num_dims, dims, num_axes, axes, x, mean, var, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, double);
DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE__KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
