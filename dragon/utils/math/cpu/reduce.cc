#include "dragon/utils/math/reduce.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/transpose.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace math {

namespace {

#define DEFINE_GLOBAL_REDUCE_FUNC(name, Expr)                                  \
  template <typename T>                                                        \
  void _GlobalReduce##name(const int N, const float scale, const T* x, T* y) { \
    using EigenT = typename math::Traits<T>::eigen_type;                       \
    auto* y_alias = (EigenT*)y;                                                \
    *y_alias = ConstEigenVectorMap<EigenT>((const EigenT*)x, N).Expr();        \
    if (scale != 1.f) EigenVectorMap<EigenT>(y_alias, 1) *= EigenT(scale);     \
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
    using EigenT = typename math::Traits<T>::eigen_type;                     \
    ConstEigenMatrixMap<EigenT> X((const EigenT*)x, cols, rows);             \
    EigenVectorMap<EigenT> Y((EigenT*)y, cols);                              \
    Y = X.rowwise().Expr();                                                  \
    if (scale != 1.f) Y *= EigenT(scale);                                    \
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
    using EigenT = typename math::Traits<T>::eigen_type;                     \
    ConstEigenMatrixMap<EigenT> X((const EigenT*)x, cols, rows);             \
    EigenVectorMap<T>(y, rows) = X.colwise().Expr();                         \
    if (scale != 1.f) EigenVectorMap<T>(y, rows) *= EigenT(scale);           \
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
      return _RowwiseReduce##name(rows, cols, scale, x, y);               \
    }                                                                     \
    if (math::utils::IsColwiseReduce(                                     \
            num_dims, dims, out_dims.data(), &rows, &cols)) {             \
      return _ColwiseReduce##name(rows, cols, scale, x, y);               \
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

#define DEFINE_REDUCE_FUNC(name, T)            \
  template <>                                  \
  DRAGON_API void Reduce##name<T, CPUContext>( \
      const int num_dims,                      \
      const int64_t* dims,                     \
      const int num_axes,                      \
      const int64_t* axes,                     \
      const float scale,                       \
      const T* x,                              \
      T* y,                                    \
      CPUContext* ctx) {                       \
    CPU_UNSUPPORTED_DTYPE(T);                  \
  }

DEFINE_REDUCE_FUNC(Max, float16);
DEFINE_REDUCE_FUNC(Max, bfloat16);
DEFINE_REDUCE_FUNC(Min, float16);
DEFINE_REDUCE_FUNC(Min, bfloat16);
DEFINE_REDUCE_FUNC(Sum, float16);
DEFINE_REDUCE_FUNC(Sum, bfloat16);
DEFINE_REDUCE_FUNC(SumSqr, float16);
DEFINE_REDUCE_FUNC(SumSqr, bfloat16);
DEFINE_REDUCE_FUNC(L1, float16);
DEFINE_REDUCE_FUNC(L1, bfloat16);
#undef DEFINE_REDUCE_FUNC

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

DEFINE_REDUCE_FUNC(Max, uint8_t, math::Traits<uint8_t>::Lowest());
DEFINE_REDUCE_FUNC(Max, int8_t, math::Traits<int8_t>::Lowest());
DEFINE_REDUCE_FUNC(Max, int, math::Traits<int>::Lowest());
DEFINE_REDUCE_FUNC(Max, int64_t, math::Traits<int64_t>::Lowest());
DEFINE_REDUCE_FUNC(Max, float, math::Traits<float>::Lowest());
DEFINE_REDUCE_FUNC(Max, double, math::Traits<double>::Lowest());
DEFINE_REDUCE_FUNC(Min, uint8_t, math::Traits<uint8_t>::Max());
DEFINE_REDUCE_FUNC(Min, int8_t, math::Traits<int8_t>::Max());
DEFINE_REDUCE_FUNC(Min, int, math::Traits<int>::Max());
DEFINE_REDUCE_FUNC(Min, int64_t, math::Traits<int64_t>::Max());
DEFINE_REDUCE_FUNC(Min, float, math::Traits<float>::Max());
DEFINE_REDUCE_FUNC(Min, double, math::Traits<double>::Max());
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
    vec64_t dims = {N}, axes = {0};                                        \
    math::ReduceSum(1, dims.data(), 1, axes.data(), scale, x, y, ctx);     \
  }                                                                        \
  template <>                                                              \
  DRAGON_API T Sum<T, CPUContext>(                                         \
      const int N, const float scale, const T* x, CPUContext* ctx) {       \
    T ret;                                                                 \
    vec64_t dims = {N}, axes = {0};                                        \
    math::ReduceSum(1, dims.data(), 1, axes.data(), scale, x, &ret, ctx);  \
    return ret;                                                            \
  }

DEFINE_SUM_FUNC(int);
DEFINE_SUM_FUNC(int64_t);
DEFINE_SUM_FUNC(float16);
DEFINE_SUM_FUNC(bfloat16);
DEFINE_SUM_FUNC(float);
DEFINE_SUM_FUNC(double);
#undef DEFINE_SUM_FUNC

} // namespace math

} // namespace dragon
