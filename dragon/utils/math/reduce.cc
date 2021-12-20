#include "dragon/utils/math/reduce.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

#define DEFINE_GLOBAL_REDUCE_FUNC(name, Expr)                                  \
  template <typename T>                                                        \
  void _GlobalReduce##name(const int N, const float scale, const T* x, T* y) { \
    *y = ConstEigenVectorMap<T>(x, N).Expr();                                  \
    if (scale != 1.f) y[0] *= T(scale);                                        \
  }

DEFINE_GLOBAL_REDUCE_FUNC(Max, maxCoeff);
DEFINE_GLOBAL_REDUCE_FUNC(Min, minCoeff);
DEFINE_GLOBAL_REDUCE_FUNC(Sum, sum);
DEFINE_GLOBAL_REDUCE_FUNC(SumSqr, squaredNorm);
DEFINE_GLOBAL_REDUCE_FUNC(L1, template lpNorm<1>);
#undef DEFINE_GLOBAL_REDUCE_FUNC

#define DEFINE_ROWWISE_REDUCE_FUNC(name, Expr)                               \
  template <typename T>                                                      \
  void _RowwiseReduce##name(                                                 \
      const int rows, const int cols, const float scale, const T* x, T* y) { \
    EigenVectorMap<T>(y, cols) =                                             \
        ConstEigenMatrixMap<T>(x, cols, rows).rowwise().Expr();              \
    if (scale != 1.f) EigenVectorMap<T>(y, cols) *= T(scale);                \
  }

DEFINE_ROWWISE_REDUCE_FUNC(Max, maxCoeff);
DEFINE_ROWWISE_REDUCE_FUNC(Min, minCoeff);
DEFINE_ROWWISE_REDUCE_FUNC(Sum, sum);
DEFINE_ROWWISE_REDUCE_FUNC(SumSqr, squaredNorm);
DEFINE_ROWWISE_REDUCE_FUNC(L1, template lpNorm<1>);
#undef DEFINE_ROWWISE_REDUCE_FUNC

#define DEFINE_COLWISE_REDUCE_FUNC(name, Expr)                               \
  template <typename T>                                                      \
  void _ColwiseReduce##name(                                                 \
      const int rows, const int cols, const float scale, const T* x, T* y) { \
    EigenVectorMap<T>(y, rows) =                                             \
        ConstEigenMatrixMap<T>(x, cols, rows).colwise().Expr();              \
    if (scale != 1.f) EigenVectorMap<T>(y, rows) *= T(scale);                \
  }

DEFINE_COLWISE_REDUCE_FUNC(Max, maxCoeff);
DEFINE_COLWISE_REDUCE_FUNC(Min, minCoeff);
DEFINE_COLWISE_REDUCE_FUNC(Sum, sum);
DEFINE_COLWISE_REDUCE_FUNC(SumSqr, squaredNorm);
DEFINE_COLWISE_REDUCE_FUNC(L1, template lpNorm<1>);
#undef DEFINE_COLWISE_REDUCE_FUNC

template <typename T, typename AccT, class Functor, class Reducer>
void _GenericReduce(
    const int rows,
    const int cols,
    const int num_dims,
    const int64_t* x_dims,
    const int64_t* x_strides,
    const Functor op,
    const Reducer reducer,
    const AccT init,
    const AccT scale,
    const T* x,
    T* y) {
  for (int i = 0; i < rows; ++i) {
    AccT val = init;
    int64_t xi, c, r;
    for (int j = 0; j < cols; ++j) {
      xi = 0;
      c = i * cols + j;
      for (int d = num_dims - 1; d >= 0; --d) {
        FIXED_DIVISOR_DIV_MOD(x_dims[d], c, &c, &r);
        xi += r * x_strides[d];
      }
      val = reducer(val, op(convert::To<AccT>(x[xi])));
    }
    y[i] = convert::To<T>(val * scale);
  }
}

#define DEFINE_REDUCE_DISPATCHER(name, Functor, Reducer)                  \
  template <typename T, typename AccT>                                    \
  void _Reduce##name(                                                     \
      const int num_dims,                                                 \
      const int64_t* dims,                                                \
      const int num_axes,                                                 \
      const int64_t* axes,                                                \
      const AccT init,                                                    \
      const AccT scale,                                                   \
      const T* x,                                                         \
      T* y) {                                                             \
    if (num_dims == num_axes) {                                           \
      const auto N = math::utils::Prod(num_dims, dims);                   \
      _GlobalReduce##name(N, scale, x, y);                                \
      return;                                                             \
    }                                                                     \
    int64_t rows, cols;                                                   \
    vec64_t out_dims(dims, dims + num_dims);                              \
    for (int i = 0; i < num_axes; ++i) {                                  \
      out_dims[axes[i]] = 1;                                              \
    }                                                                     \
    if (math::utils::IsRowwiseReduce(                                     \
            num_dims, dims, out_dims.data(), &rows, &cols)) {             \
      _RowwiseReduce##name(rows, cols, scale, x, y);                      \
      return;                                                             \
    }                                                                     \
    if (math::utils::IsColwiseReduce(                                     \
            num_dims, dims, out_dims.data(), &rows, &cols)) {             \
      _ColwiseReduce##name(rows, cols, scale, x, y);                      \
      return;                                                             \
    }                                                                     \
    vec64_t transpose_axes(num_dims);                                     \
    vec64_t transpose_strides(num_dims);                                  \
    vec64_t transpose_dims(num_dims);                                     \
    math::utils::TransposeAxesForReduce(                                  \
        num_dims, num_axes, axes, transpose_axes.data());                 \
    math::utils::ComputeTransposeStrides(                                 \
        num_dims, dims, transpose_axes.data(), transpose_strides.data()); \
    rows = cols = 1;                                                      \
    const int pivot = num_dims - num_axes;                                \
    for (int i = 0; i < pivot; ++i) {                                     \
      rows *= dims[transpose_axes[i]];                                    \
    }                                                                     \
    for (int i = pivot; i < num_dims; ++i) {                              \
      cols *= dims[transpose_axes[i]];                                    \
    }                                                                     \
    for (int i = 0; i < num_dims; ++i) {                                  \
      transpose_dims[i] = dims[transpose_axes[i]];                        \
    }                                                                     \
    _GenericReduce(                                                       \
        rows,                                                             \
        cols,                                                             \
        num_dims,                                                         \
        transpose_dims.data(),                                            \
        transpose_strides.data(),                                         \
        Functor<AccT>(),                                                  \
        Reducer<AccT>(),                                                  \
        init,                                                             \
        scale,                                                            \
        x,                                                                \
        y);                                                               \
  }

DEFINE_REDUCE_DISPATCHER(Max, math::IdentityFunctor, math::MaxFunctor);
DEFINE_REDUCE_DISPATCHER(Min, math::IdentityFunctor, math::MinFunctor);
DEFINE_REDUCE_DISPATCHER(Sum, math::IdentityFunctor, math::PlusFunctor);
DEFINE_REDUCE_DISPATCHER(SumSqr, math::SqrFunctor, math::PlusFunctor);
DEFINE_REDUCE_DISPATCHER(L1, math::AbsFunctor, math::PlusFunctor);
#undef DEFINE_REDUCE_DISPATCHER

} // namespace

#define DEFINE_REDUCE_FUNC(name)                     \
  template <>                                        \
  DRAGON_API void Reduce##name<float16, CPUContext>( \
      const int num_dims,                            \
      const int64_t* dims,                           \
      const int num_axes,                            \
      const int64_t* axes,                           \
      const float scale,                             \
      const float16* x,                              \
      float16* y,                                    \
      CPUContext* ctx) {                             \
    CPU_FP16_NOT_SUPPORTED;                          \
  }

DEFINE_REDUCE_FUNC(Max);
DEFINE_REDUCE_FUNC(Min);
DEFINE_REDUCE_FUNC(Sum);
DEFINE_REDUCE_FUNC(SumSqr);
DEFINE_REDUCE_FUNC(L1);
#undef DEFINE_REDUCE_FUNC

template <>
DRAGON_API void Sum<float16, CPUContext>(
    const int N,
    const float alpha,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
DRAGON_API float16 Sum<float16, CPUContext>(
    const int N,
    const float alpha,
    const float16* x,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
  return float16();
}

#define DEFINE_REDUCE_FUNC(name, T, kInit)                   \
  template <>                                                \
  DRAGON_API void Reduce##name<T, CPUContext>(               \
      const int num_dims,                                    \
      const int64_t* dims,                                   \
      const int num_axes,                                    \
      const int64_t* axes,                                   \
      const float scale,                                     \
      const T* x,                                            \
      T* y,                                                  \
      CPUContext* ctx) {                                     \
    vec64_t new_dims, new_axes;                              \
    math::utils::CollapseReduceAxes(                         \
        num_dims, dims, num_axes, axes, new_dims, new_axes); \
    _Reduce##name(                                           \
        new_dims.size(),                                     \
        new_dims.data(),                                     \
        new_axes.size(),                                     \
        new_axes.data(),                                     \
        convert::To<T>(kInit),                               \
        convert::To<T>(scale),                               \
        x,                                                   \
        y);                                                  \
  }

DEFINE_REDUCE_FUNC(Max, uint8_t, std::numeric_limits<uint8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int8_t, std::numeric_limits<int8_t>::lowest());
DEFINE_REDUCE_FUNC(Max, int, std::numeric_limits<int>::lowest());
DEFINE_REDUCE_FUNC(Max, int64_t, std::numeric_limits<int64_t>::lowest());
DEFINE_REDUCE_FUNC(Max, float, std::numeric_limits<float>::lowest());
DEFINE_REDUCE_FUNC(Max, double, std::numeric_limits<double>::lowest());
DEFINE_REDUCE_FUNC(Min, uint8_t, std::numeric_limits<uint8_t>::max());
DEFINE_REDUCE_FUNC(Min, int8_t, std::numeric_limits<int8_t>::max());
DEFINE_REDUCE_FUNC(Min, int, std::numeric_limits<int>::max());
DEFINE_REDUCE_FUNC(Min, int64_t, std::numeric_limits<int64_t>::max());
DEFINE_REDUCE_FUNC(Min, float, std::numeric_limits<float>::max());
DEFINE_REDUCE_FUNC(Min, double, std::numeric_limits<double>::max());
DEFINE_REDUCE_FUNC(Sum, int, int(0));
DEFINE_REDUCE_FUNC(Sum, int64_t, int64_t(0));
DEFINE_REDUCE_FUNC(Sum, float, 0.f);
DEFINE_REDUCE_FUNC(Sum, double, 0.);
DEFINE_REDUCE_FUNC(SumSqr, int, int(0));
DEFINE_REDUCE_FUNC(SumSqr, int64_t, int64_t(0));
DEFINE_REDUCE_FUNC(SumSqr, float, 0.f);
DEFINE_REDUCE_FUNC(SumSqr, double, 0.);
DEFINE_REDUCE_FUNC(L1, float, 0.f);
DEFINE_REDUCE_FUNC(L1, double, 0.);
#undef DEFINE_REDUCE_FUNC

#define DEFINE_SUM_FUNC(T)                                                 \
  template <>                                                              \
  DRAGON_API void Sum<T, CPUContext>(                                      \
      const int N, const float scale, const T* x, T* y, CPUContext* ctx) { \
    T val = ConstEigenVectorArrayMap<T>(x, N).sum();                       \
    *y = val * T(scale);                                                   \
  }                                                                        \
  template <>                                                              \
  DRAGON_API T Sum<T, CPUContext>(                                         \
      const int N, const float scale, const T* x, CPUContext* ctx) {       \
    T val = ConstEigenVectorArrayMap<T>(x, N).sum();                       \
    return val * T(scale);                                                 \
  }

DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon
