#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
void _RowwiseMoments(
    const int64_t rows,
    const int64_t cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(rows);
  for (int64_t i = 0; i < cols; ++i) {
    AccT m_val = AccT(0), v_val = AccT(0);
    for (int64_t j = 0; j < rows; ++j) {
      const AccT val = convert::To<AccT>(x[j * cols + i]);
      m_val += val;
      v_val += val * val;
    }
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void _ColwiseMoments(
    const int64_t rows,
    const int64_t cols,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(cols);
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t offset = i * cols;
    AccT m_val = AccT(0), v_val = AccT(0);
    for (int64_t j = 0; j < cols; ++j) {
      const AccT val = convert::To<AccT>(x[offset + j]);
      m_val += val;
      v_val += val * val;
    }
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void _GenericMoments(
    const int64_t rows,
    const int64_t cols,
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const T* x,
    AccT* mean,
    AccT* var) {
  const AccT scale = AccT(1) / AccT(cols);
  for (int64_t i = 0; i < rows; ++i) {
    const int64_t offset = i * cols;
    AccT m_val = AccT(0), v_val = AccT(0);
    for (int64_t j = 0; j < cols; ++j) {
      int64_t xi = 0, c = offset + j, r;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      const AccT val = convert::To<AccT>(x[xi]);
      m_val += val;
      v_val += val * val;
    }
    m_val = m_val * scale;
    mean[i] = m_val;
    var[i] = v_val * scale - m_val * m_val;
  }
}

template <typename T, typename AccT>
void DispatchMoments(
    const int num_dims,
    const int64_t* dims,
    const int num_axes,
    const int64_t* axes,
    const T* x,
    AccT* mean,
    AccT* var,
    CPUContext* ctx) {
  int64_t rows, cols;
  vec64_t out_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i) {
    out_dims[axes[i]] = 1;
  }
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    return _RowwiseMoments(rows, cols, x, mean, var);
  }
  if (math::utils::IsColwiseReduce(
          num_dims, dims, out_dims.data(), &rows, &cols)) {
    return _ColwiseMoments(rows, cols, x, mean, var);
  }
  // clang-format off
  vec64_t transpose_axes(num_dims);
  vec64_t transpose_dims(num_dims);
  vec64_t transpose_strides(num_dims);
  math::utils::TransposeAxesForReduce(
      num_dims, num_axes, axes, transpose_axes.data());
  math::utils::ComputeTransposeStrides(
      num_dims, dims, transpose_axes.data(), transpose_strides.data());
  rows = cols = 1;
  const int pivot = num_dims - num_axes;
  for (int i = 0; i < pivot; ++i) rows *= dims[transpose_axes[i]];
  for (int i = pivot; i < num_dims; ++i) cols *= dims[transpose_axes[i]];
  for (int i = 0; i < num_dims; ++i) transpose_dims[i] = dims[transpose_axes[i]];
  _GenericMoments( // clang-format on
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

#define DEFINE_KERNEL_LAUNCHER(T, AccT)                      \
  template <>                                                \
  void Moments<T, AccT, CPUContext>(                         \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const T* x,                                            \
      AccT* mean,                                            \
      AccT* var,                                             \
      CPUContext* ctx) {                                     \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    DispatchMoments(                                         \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        x,                                                   \
        mean,                                                \
        var,                                                 \
        ctx);                                                \
  }

DEFINE_KERNEL_LAUNCHER(float16, float);
DEFINE_KERNEL_LAUNCHER(bfloat16, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE__KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
