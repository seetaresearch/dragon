#include "dragon/utils/math/reduce.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/math/utils.h"
#include "dragon/utils/omp_utils.h"

namespace dragon {

namespace math {

namespace {

#define DEFINE_ROWWISE_REDUCE_FUNC(name, expr)                               \
  template <typename T>                                                      \
  void _RowwiseReduce##name(                                                 \
      const int rows, const int cols, const T* scale, const T* x, T* y) {    \
    if (scale != nullptr) {                                                  \
      EigenVectorMap<T>(y, cols) =                                           \
          ConstEigenMatrixMap<T>(x, cols, rows).rowwise().expr() * (*scale); \
    } else {                                                                 \
      EigenVectorMap<T>(y, cols) =                                           \
          ConstEigenMatrixMap<T>(x, cols, rows).rowwise().expr();            \
    }                                                                        \
  }

DEFINE_ROWWISE_REDUCE_FUNC(Max, maxCoeff);
DEFINE_ROWWISE_REDUCE_FUNC(Min, minCoeff);
DEFINE_ROWWISE_REDUCE_FUNC(Sum, sum);
#undef DEFINE_ROWWISE_REDUCE_FUNC

#define DEFINE_COLWISE_REDUCE_FUNC(name, expr)                               \
  template <typename T>                                                      \
  void _ColwiseReduce##name(                                                 \
      const int rows, const int cols, const T* scale, const T* x, T* y) {    \
    if (scale != nullptr) {                                                  \
      EigenVectorMap<T>(y, rows) =                                           \
          ConstEigenMatrixMap<T>(x, cols, rows).colwise().expr() * (*scale); \
    } else {                                                                 \
      EigenVectorMap<T>(y, rows) =                                           \
          ConstEigenMatrixMap<T>(x, cols, rows).colwise().expr();            \
    }                                                                        \
  }

DEFINE_COLWISE_REDUCE_FUNC(Max, maxCoeff);
DEFINE_COLWISE_REDUCE_FUNC(Min, minCoeff);
DEFINE_COLWISE_REDUCE_FUNC(Sum, sum);
#undef DEFINE_COLWISE_REDUCE_FUNC

template <typename T>
void _GenericReduceMax(
    const int rows,
    const int cols,
    const int num_dims,
    const int* x_dims,
    const int* x_strides,
    const T* scale,
    const T* x,
    T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    T val = std::numeric_limits<T>::lowest();
    int xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      val = std::max(x[xi], val);
    }
    y[i] = val;
  }
}

template <typename T>
void _GenericReduceMin(
    const int rows,
    const int cols,
    const int num_dims,
    const int* x_dims,
    const int* x_strides,
    const T* scale,
    const T* x,
    T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    T val = std::numeric_limits<T>::max();
    int xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      val = std::min(x[xi], val);
    }
    y[i] = val;
  }
}

template <typename T>
void _GenericReduceSum(
    const int rows,
    const int cols,
    const int num_dims,
    const int* x_dims,
    const int* x_strides,
    const T* scale,
    const T* x,
    T* y) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(rows))
#endif
  for (int i = 0; i < rows; ++i) {
    T val = 0;
    int xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      val += x[xi];
    }
    if (scale != nullptr) {
      y[i] = val * (*scale);
    } else {
      y[i] = val;
    }
  }
}

#define DEFINE_REDUCE_FUNC(name)                                           \
  template <typename T>                                                    \
  void _Reduce##name(                                                      \
      const int num_dims,                                                  \
      const int* dims,                                                     \
      const int num_axes,                                                  \
      const int* axes,                                                     \
      const T* scale,                                                      \
      const T* x,                                                          \
      T* y) {                                                              \
    int rows, cols;                                                        \
    vec32_t y_dims(dims, dims + num_dims);                                 \
    for (int i = 0; i < num_axes; ++i)                                     \
      y_dims[axes[i]] = 1;                                                 \
    /* Case #1: Rowwise Reduce */                                          \
    if (utils::math::IsRowwiseReduce(                                      \
            num_dims, dims, y_dims.data(), &rows, &cols)) {                \
      _RowwiseReduce##name(rows, cols, scale, x, y);                       \
      return;                                                              \
    }                                                                      \
    /* Case #2: Colwise Reduce */                                          \
    if (utils::math::IsColwiseReduce(                                      \
            num_dims, dims, y_dims.data(), &rows, &cols)) {                \
      _ColwiseReduce##name(rows, cols, scale, x, y);                       \
      return;                                                              \
    }                                                                      \
    /* Case #3: Generic Reduce */                                          \
    vec32_t axesT(num_dims), stridesT(num_dims), dimsT(num_dims);          \
    utils::math::TransposeAxesForReduce(                                   \
        num_dims, num_axes, axes, axesT.data());                           \
    utils::math::ComputeTransposeStrides(                                  \
        num_dims, dims, axesT.data(), stridesT.data());                    \
    rows = cols = 1;                                                       \
    const int pivot = num_dims - num_axes;                                 \
    for (int i = 0; i < pivot; ++i)                                        \
      rows *= dims[axesT[i]];                                              \
    for (int i = pivot; i < num_dims; ++i)                                 \
      cols *= dims[axesT[i]];                                              \
    for (int i = 0; i < num_dims; ++i)                                     \
      dimsT[i] = dims[axesT[i]];                                           \
    _GenericReduce##name(                                                  \
        rows, cols, num_dims, dimsT.data(), stridesT.data(), scale, x, y); \
  }

DEFINE_REDUCE_FUNC(Max);
DEFINE_REDUCE_FUNC(Min);
DEFINE_REDUCE_FUNC(Sum);
#undef DEFINE_REDUCE_FUNC

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ReduceMax<float16, CPUContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ReduceMin<float16, CPUContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ReduceSum<float16, CPUContext>(
    const int num_dims,
    const int* dims,
    const int num_axes,
    const int* axes,
    const float scale,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
DRAGON_API void Sum<float16, CPUContext>(
    const int n,
    const float alpha,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
DRAGON_API float16 Sum<float16, CPUContext>(
    const int n,
    const float alpha,
    const float16* x,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
  return float16();
}

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void Reduce##name<T, CPUContext>(                                   \
      const int num_dims,                                             \
      const int* dims,                                                \
      const int num_axes,                                             \
      const int* axes,                                                \
      const T* x,                                                     \
      T* y,                                                           \
      CPUContext* ctx) {                                              \
    _Reduce##name(num_dims, dims, num_axes, axes, (T*)nullptr, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(Max, int8_t);
DEFINE_KERNEL_LAUNCHER(Max, uint8_t);
DEFINE_KERNEL_LAUNCHER(Max, int);
DEFINE_KERNEL_LAUNCHER(Max, int64_t);
DEFINE_KERNEL_LAUNCHER(Max, float);
DEFINE_KERNEL_LAUNCHER(Max, double);
DEFINE_KERNEL_LAUNCHER(Min, int8_t);
DEFINE_KERNEL_LAUNCHER(Min, uint8_t);
DEFINE_KERNEL_LAUNCHER(Min, int);
DEFINE_KERNEL_LAUNCHER(Min, int64_t);
DEFINE_KERNEL_LAUNCHER(Min, float);
DEFINE_KERNEL_LAUNCHER(Min, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                      \
  template <>                                                \
  void Reduce##name<T, CPUContext>(                          \
      const int num_dims,                                    \
      const int* dims,                                       \
      const int num_axes,                                    \
      const int* axes,                                       \
      const float scale,                                     \
      const T* x,                                            \
      T* y,                                                  \
      CPUContext* ctx) {                                     \
    T s = static_cast<T>(scale);                             \
    _Reduce##name(num_dims, dims, num_axes, axes, &s, x, y); \
  }

DEFINE_KERNEL_LAUNCHER(Sum, int8_t);
DEFINE_KERNEL_LAUNCHER(Sum, uint8_t);
DEFINE_KERNEL_LAUNCHER(Sum, int);
DEFINE_KERNEL_LAUNCHER(Sum, int64_t);
DEFINE_KERNEL_LAUNCHER(Sum, float);
DEFINE_KERNEL_LAUNCHER(Sum, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_SUM_FUNC(T)                                                 \
  template <>                                                              \
  DRAGON_API void Sum<T, CPUContext>(                                      \
      const int n, const float scale, const T* x, T* y, CPUContext* ctx) { \
    T val = ConstEigenVectorArrayMap<T>(x, n).sum();                       \
    *y = val * scale;                                                      \
  }                                                                        \
  template <>                                                              \
  T Sum<T, CPUContext>(                                                    \
      const int n, const float scale, const T* x, CPUContext* ctx) {       \
    T val = ConstEigenVectorArrayMap<T>(x, n).sum();                       \
    return val * scale;                                                    \
  }

DEFINE_SUM_FUNC(int8_t);
DEFINE_SUM_FUNC(uint8_t);
DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon
