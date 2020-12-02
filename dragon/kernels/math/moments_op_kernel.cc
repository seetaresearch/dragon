#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename Tx, typename Ty>
void _RowwiseMoments(
    const int rows,
    const int cols,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  const Ty scale = Ty(1) / (Ty)rows;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(cols))
#endif
  for (int i = 0; i < cols; ++i) {
    Tx x_val;
    Ty m_val = 0, v_val = 0, mu;
    for (int j = 0; j < rows; ++j) {
      x_val = x[j * cols + i];
      m_val += x_val;
      v_val += x_val * x_val;
    }
    mean[i] = mu = m_val * scale;
    var[i] = v_val * scale - mu * mu;
  }
}

template <typename Tx, typename Ty>
void _ColwiseMoments(
    const int rows,
    const int cols,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  const Ty scale = Ty(1) / (Ty)cols;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    Tx x_val;
    Ty m_val = 0, v_val = 0, mu;
    for (int j = 0; j < cols; ++j) {
      x_val = x[i * cols + j];
      m_val += x_val;
      v_val += x_val * x_val;
    }
    mean[i] = mu = m_val * scale;
    var[i] = v_val * scale - mu * mu;
  }
}

template <typename Tx, typename Ty>
void _GenericMoments(
    const int rows,
    const int cols,
    const int num_dims,
    const int* x_dims,
    const int* x_strides,
    const Tx* x,
    Ty* mean,
    Ty* var) {
  const Ty scale = Ty(1) / (Ty)cols;
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    Tx x_val;
    Ty m_val = 0, v_val = 0, mu;
    int xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      x_val = x[xi];
      m_val += x_val;
      v_val += x_val * x_val;
    }
    mean[i] = mu = m_val * scale;
    var[i] = v_val * scale - mu * mu;
  }
}

template <typename Tx, typename Ty>
void _Moments(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const Tx* x,
    Ty* mean,
    Ty* var,
    CPUContext* ctx) {
  int rows, cols;
  vec32_t y_dims(dims, dims + num_dims);
  for (int i = 0; i < num_axes; ++i)
    y_dims[axes[i]] = 1;

  // Case #1: Rowwise Reduce
  if (math::utils::IsRowwiseReduce(
          num_dims, dims, y_dims.data(), &rows, &cols)) {
    _RowwiseMoments(rows, cols, x, mean, var);
    return;
  }

  // Case #2: Colwise Reduce
  if (math::utils::IsColwiseReduce(
          num_dims, dims, y_dims.data(), &rows, &cols)) {
    _ColwiseMoments(rows, cols, x, mean, var);
    return;
  }

  // Case #3: Generic Reduce
  vec32_t axesT(num_dims), stridesT(num_dims), dimsT(num_dims);
  math::utils::TransposeAxesForReduce(num_dims, num_axes, axes, axesT.data());
  math::utils::ComputeTransposeStrides(
      num_dims, dims, axesT.data(), stridesT.data());

  rows = cols = 1;
  const int pivot = num_dims - num_axes;
  for (int i = 0; i < pivot; ++i)
    rows *= dims[axesT[i]];
  for (int i = pivot; i < num_dims; ++i)
    cols *= dims[axesT[i]];
  for (int i = 0; i < num_dims; ++i)
    dimsT[i] = dims[axesT[i]];

  _GenericMoments(
      rows, cols, num_dims, dimsT.data(), stridesT.data(), x, mean, var);
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void Moments<float16, float, CPUContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float* mean,
    float* var,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(Tx, Ty)                           \
  template <>                                                    \
  void Moments<Tx, Ty, CPUContext>(                              \
      const int num_dims,                                        \
      const int* dims,                                           \
      const int num_axes,                                        \
      const int* axes,                                           \
      const Tx* x,                                               \
      Ty* mean,                                                  \
      Ty* var,                                                   \
      CPUContext* ctx) {                                         \
    _Moments(num_dims, dims, num_axes, axes, x, mean, var, ctx); \
  }

DEFINE_KERNEL_LAUNCHER(int8_t, float);
DEFINE_KERNEL_LAUNCHER(uint8_t, float);
DEFINE_KERNEL_LAUNCHER(int, float);
DEFINE_KERNEL_LAUNCHER(int64_t, float);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
#undef DEFINE__KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
