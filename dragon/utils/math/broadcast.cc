#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

#define DECLARE_ROWWISE_COLWISE_BINARY_FUNC(name, TOut)                 \
  template <typename T, bool BroadcastA>                              \
  void _Rowwise##name(                                                  \
      const int rows, const int cols, const T* a, const T* b, TOut* y); \
  template <typename T, bool BroadcastA>                                \
  void _Colwise##name(                                                  \
      const int rows, const int cols, const T* a, const T* b, TOut* y);

DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Add, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Sub, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Mul, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Div, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Pow, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Minimum, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Maximum, T);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Equal, bool);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(NotEqual, bool);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Less, bool);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(LessEqual, bool);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(Greater, bool);
DECLARE_ROWWISE_COLWISE_BINARY_FUNC(GreaterEqual, bool);
#undef DECLARE_ROWWISE_COLWISE_BINARY_FUNC

#define DEFINE_BROADCAST_1ST_FUNC(name, T, expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows).colwise() expr## =              \
          ConstEigenVectorArrayMap<T>(a, cols);                       \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(b, cols, rows)                        \
              .colwise() expr ConstEigenVectorArrayMap<T>(a, cols);   \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows).rowwise() expr## =              \
          ConstEigenVectorArrayMap<T>(a, rows).transpose();           \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(b, cols, rows)                        \
              .rowwise() expr ConstEigenVectorArrayMap<T>(a, rows)    \
              .transpose();                                           \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Add, int8_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, uint8_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, int, +);
DEFINE_BROADCAST_1ST_FUNC(Add, int64_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, float, +);
DEFINE_BROADCAST_1ST_FUNC(Add, double, +);
DEFINE_BROADCAST_1ST_FUNC(Mul, int8_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, uint8_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, int, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, int64_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, float, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, double, *);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_1ST_FUNC(name, T, expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows) *= -1;                          \
      EigenArrayMap<T>(y, cols, rows).colwise() +=                    \
          ConstEigenVectorArrayMap<T>(a, cols);                       \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          (-ConstEigenArrayMap<T>(b, cols, rows)).colwise() +         \
          ConstEigenVectorArrayMap<T>(a, cols);                       \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows) *= -1;                          \
      EigenArrayMap<T>(y, cols, rows).rowwise() +=                    \
          ConstEigenVectorArrayMap<T>(a, rows).transpose();           \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          (-ConstEigenArrayMap<T>(b, cols, rows)).rowwise() +         \
          ConstEigenVectorArrayMap<T>(a, rows).transpose();           \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Sub, int8_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, uint8_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, int, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, int64_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, float, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, double, -);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_1ST_FUNC(name, T, expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(y, cols, rows).inverse();             \
      EigenArrayMap<T>(y, cols, rows).colwise() *=                    \
          ConstEigenVectorArrayMap<T>(a, cols);                       \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(b, cols, rows).inverse().colwise() *  \
          ConstEigenVectorArrayMap<T>(a, cols);                       \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (b == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(y, cols, rows).inverse();             \
      EigenArrayMap<T>(y, cols, rows).rowwise() *=                    \
          ConstEigenVectorArrayMap<T>(a, rows).transpose();           \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(b, cols, rows).inverse().rowwise() *  \
          ConstEigenVectorArrayMap<T>(a, rows).transpose();           \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Div, int8_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, uint8_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, int, /);
DEFINE_BROADCAST_1ST_FUNC(Div, int64_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, float, /);
DEFINE_BROADCAST_1ST_FUNC(Div, double, /);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_2ND_FUNC(name, T, expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (a == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows).colwise() expr## =              \
          ConstEigenVectorArrayMap<T>(b, cols);                       \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(a, cols, rows)                        \
              .colwise() expr ConstEigenVectorArrayMap<T>(b, cols);   \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    if (a == y) {                                                     \
      EigenArrayMap<T>(y, cols, rows).rowwise() expr## =              \
          ConstEigenVectorArrayMap<T>(b, rows).transpose();           \
    } else {                                                          \
      EigenArrayMap<T>(y, cols, rows) =                               \
          ConstEigenArrayMap<T>(a, cols, rows)                        \
              .rowwise() expr ConstEigenVectorArrayMap<T>(b, rows)    \
              .transpose();                                           \
    }                                                                 \
  }

DEFINE_BROADCAST_2ND_FUNC(Add, int8_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, uint8_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, int, +);
DEFINE_BROADCAST_2ND_FUNC(Add, int64_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, float, +);
DEFINE_BROADCAST_2ND_FUNC(Add, double, +);
DEFINE_BROADCAST_2ND_FUNC(Sub, int8_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, uint8_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, int, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, int64_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, float, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, double, -);
DEFINE_BROADCAST_2ND_FUNC(Mul, int8_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, uint8_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, int, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, int64_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, float, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, double, *);
DEFINE_BROADCAST_2ND_FUNC(Div, int8_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, uint8_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, int, /);
DEFINE_BROADCAST_2ND_FUNC(Div, int64_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, float, /);
DEFINE_BROADCAST_2ND_FUNC(Div, double, /);
#undef DEFINE_BROADCAST_2ND_FUNC

#define DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(name, TIn, TOut, expr)            \
  template <>                                                                \
  void _Rowwise##name<TIn, true>(                                            \
      const int rows, const int cols, const TIn* a, const TIn* b, TOut* y) { \
    EigenArrayMap<TOut>(y, cols, rows) =                                     \
        ConstEigenVectorArrayMap<TIn>(a, cols).rowwise().replicate(rows)     \
            expr ConstEigenArrayMap<TIn>(b, cols, rows);                     \
  }                                                                          \
  template <>                                                                \
  void _Colwise##name<TIn, true>(                                            \
      const int rows, const int cols, const TIn* a, const TIn* b, TOut* y) { \
    EigenArrayMap<TOut>(y, cols, rows) =                                     \
        ConstEigenVectorArrayMap2<TIn>(a, rows).colwise().replicate(cols)    \
            expr ConstEigenArrayMap<TIn>(b, cols, rows);                     \
  }                                                                          \
  template <>                                                                \
  void _Rowwise##name<TIn, false>(                                           \
      const int rows, const int cols, const TIn* a, const TIn* b, TOut* y) { \
    EigenArrayMap<TOut>(y, cols, rows) =                                     \
        ConstEigenArrayMap<TIn>(a, cols, rows)                               \
            expr ConstEigenVectorArrayMap<TIn>(b, cols)                      \
                .rowwise()                                                   \
                .replicate(rows);                                            \
  }                                                                          \
  template <>                                                                \
  void _Colwise##name<TIn, false>(                                           \
      const int rows, const int cols, const TIn* a, const TIn* b, TOut* y) { \
    EigenArrayMap<TOut>(y, cols, rows) =                                     \
        ConstEigenArrayMap<TIn>(a, cols, rows)                               \
            expr ConstEigenVectorArrayMap2<TIn>(b, rows)                     \
                .colwise()                                                   \
                .replicate(cols);                                            \
  }

DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int8_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, uint8_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int64_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, float, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, double, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int8_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, uint8_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int64_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, float, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, double, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int8_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, uint8_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int64_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, float, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, double, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int8_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, uint8_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int64_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, float, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, double, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int8_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, uint8_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int64_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, float, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, double, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int8_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, uint8_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int64_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, float, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, double, bool, >=);
#undef DEFINE_ROWWISE_COLWISE_BIANRY_FUNC

#define DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(name, T, func)                     \
  template <>                                                                 \
  void _Rowwise##name<T, true>(                                               \
      const int rows, const int cols, const T* a, const T* b, T* y) {         \
    EigenArrayMap<T>(y, cols, rows) =                                         \
        ConstEigenVectorArrayMap<T>(a, cols).rowwise().replicate(rows).func(  \
            ConstEigenArrayMap<T>(b, cols, rows));                            \
  }                                                                           \
  template <>                                                                 \
  void _Colwise##name<T, true>(                                               \
      const int rows, const int cols, const T* a, const T* b, T* y) {         \
    EigenArrayMap<T>(y, cols, rows) =                                         \
        ConstEigenVectorArrayMap2<T>(a, rows).colwise().replicate(cols).func( \
            ConstEigenArrayMap<T>(b, cols, rows));                            \
  }                                                                           \
  template <>                                                                 \
  void _Rowwise##name<T, false>(                                              \
      const int rows, const int cols, const T* a, const T* b, T* y) {         \
    EigenArrayMap<T>(y, cols, rows) =                                         \
        ConstEigenArrayMap<T>(a, cols, rows)                                  \
            .func(ConstEigenVectorArrayMap<T>(b, cols).rowwise().replicate(   \
                rows));                                                       \
  }                                                                           \
  template <>                                                                 \
  void _Colwise##name<T, false>(                                              \
      const int rows, const int cols, const T* a, const T* b, T* y) {         \
    EigenArrayMap<T>(y, cols, rows) =                                         \
        ConstEigenArrayMap<T>(a, cols, rows)                                  \
            .func(ConstEigenVectorArrayMap2<T>(b, rows).colwise().replicate(  \
                cols));                                                       \
  }

DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, float, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, double, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int8_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, uint8_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int64_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, float, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, double, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int8_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, uint8_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int64_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, float, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, double, max);
#undef DEFINE_ROWWISE_COLWISE_BIANRY_FUNC

#define DEFINE_BROADCAST_BINARY_FUNC(name, TOut, expr)                \
  template <typename T>                                               \
  void _Broadcast##name(                                              \
      const int num_dims,                                             \
      const int64_t* a_strides,                                       \
      const int64_t* b_strides,                                       \
      const int64_t* y_dims,                                          \
      const T* a,                                                     \
      const T* b,                                                     \
      TOut* y) {                                                      \
    const auto count = std::accumulate(                               \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());    \
    vec64_t idx(num_dims, 0);                                         \
    int64_t ai, bi;                                                   \
    for (int yi = 0; yi < count; ++yi) {                              \
      ai = bi = 0;                                                    \
      for (int d = num_dims - 1; d >= 0; --d) {                       \
        ai += idx[d] * a_strides[d];                                  \
        bi += idx[d] * b_strides[d];                                  \
      }                                                               \
      y[yi] = a[ai] expr b[bi];                                       \
      utils::math::IncreaseIndexInDims(num_dims, y_dims, idx.data()); \
    }                                                                 \
  }

DEFINE_BROADCAST_BINARY_FUNC(Add, T, +);
DEFINE_BROADCAST_BINARY_FUNC(Sub, T, -);
DEFINE_BROADCAST_BINARY_FUNC(Mul, T, *);
DEFINE_BROADCAST_BINARY_FUNC(Div, T, /);
DEFINE_BROADCAST_BINARY_FUNC(Equal, bool, ==);
DEFINE_BROADCAST_BINARY_FUNC(NotEqual, bool, !=);
DEFINE_BROADCAST_BINARY_FUNC(Less, bool, <);
DEFINE_BROADCAST_BINARY_FUNC(LessEqual, bool, <=);
DEFINE_BROADCAST_BINARY_FUNC(Greater, bool, >);
DEFINE_BROADCAST_BINARY_FUNC(GreaterEqual, bool, >=);
#undef DEFINE_BROADCAST_BINARY_FUNC

#define DEFINE_BROADCAST_BINARY_FUNC(name, TOut, func)                \
  template <typename T>                                               \
  void _Broadcast##name(                                              \
      const int num_dims,                                             \
      const int64_t* a_strides,                                       \
      const int64_t* b_strides,                                       \
      const int64_t* y_dims,                                          \
      const T* a,                                                     \
      const T* b,                                                     \
      TOut* y) {                                                      \
    const auto count = std::accumulate(                               \
        y_dims, y_dims + num_dims, 1, std::multiplies<int64_t>());    \
    vec64_t idx(num_dims, 0);                                         \
    int64_t ai, bi;                                                   \
    for (int yi = 0; yi < count; ++yi) {                              \
      ai = bi = 0;                                                    \
      for (int d = num_dims - 1; d >= 0; --d) {                       \
        ai += idx[d] * a_strides[d];                                  \
        bi += idx[d] * b_strides[d];                                  \
      }                                                               \
      y[yi] = func(a[ai], b[bi]);                                     \
      utils::math::IncreaseIndexInDims(num_dims, y_dims, idx.data()); \
    }                                                                 \
  }

DEFINE_BROADCAST_BINARY_FUNC(Pow, T, std::pow);
DEFINE_BROADCAST_BINARY_FUNC(Minimum, T, std::min);
DEFINE_BROADCAST_BINARY_FUNC(Maximum, T, std::max);
#undef DEFINE_BROADCAST_BINARY_FUNC

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_SET_FUNC(T)                                                   \
  template <>                                                                \
  DRAGON_API void Set<T, CPUContext>(                                        \
      const int x_ndim,                                                      \
      const int64_t* x_dims,                                                 \
      const int y_ndim,                                                      \
      const int64_t* y_dims,                                                 \
      const T* x,                                                            \
      T* y,                                                                  \
      CPUContext* ctx) {                                                     \
    int rows, cols, broadcast_1st;                                           \
    vec64_t X_dims(x_dims, x_dims + x_ndim);                                 \
    vec64_t Y_dims(y_dims, y_dims + y_ndim);                                 \
    vec64_t X_broadcast_dims, Y_broadcast_dims;                              \
    utils::math::ComputeBinaryBroadcastDims(                                 \
        X_dims, Y_dims, X_broadcast_dims, Y_broadcast_dims);                 \
    if (X_broadcast_dims == Y_broadcast_dims) {                              \
      auto count = std::accumulate(                                          \
          x_dims, x_dims + x_ndim, 1, std::multiplies<int64_t>());           \
      Copy(count, x, y, ctx);                                                \
      return;                                                                \
    }                                                                        \
    if (utils::math::IsRowwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {     \
      EigenArrayMap<T>(y, cols, rows).colwise() =                            \
          ConstEigenVectorArrayMap<T>(x, cols);                              \
      return;                                                                \
    }                                                                        \
    if (utils::math::IsColwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {     \
      EigenArrayMap<T>(y, cols, rows).rowwise() =                            \
          ConstEigenVectorArrayMap<T>(x, rows).transpose();                  \
      return;                                                                \
    }                                                                        \
    vec64_t X_broadcast_strides, _;                                          \
    utils::math::ComputeBinaryBroadcastStrides(                              \
        X_dims, Y_dims, X_broadcast_strides, _, _);                          \
    const int num_dims = Y_dims.size();                                      \
    const auto count = std::accumulate(                                      \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());        \
    vec64_t idx(num_dims, 0);                                                \
    int64_t xi;                                                              \
    for (int yi = 0; yi < count; ++yi) {                                     \
      xi = 0;                                                                \
      for (int d = num_dims - 1; d >= 0; --d) {                              \
        xi += idx[d] * X_broadcast_strides[d];                               \
      }                                                                      \
      y[yi] = x[xi];                                                         \
      utils::math::IncreaseIndexInDims(num_dims, Y_dims.data(), idx.data()); \
    }                                                                        \
  }

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float16);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_BINARY_FUNC(name, TIn, TOut)                                \
  template <>                                                              \
  DRAGON_API void name<TIn, CPUContext>(                                   \
      const int a_ndim,                                                    \
      const int64_t* a_dims,                                               \
      const int b_ndim,                                                    \
      const int64_t* b_dims,                                               \
      const TIn* a,                                                        \
      const TIn* b,                                                        \
      TOut* y,                                                             \
      CPUContext* ctx) {                                                   \
    int rows, cols, broadcast_1st;                                         \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                               \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                               \
    vec64_t A_broadcast_dims, B_broadcast_dims;                            \
    utils::math::ComputeBinaryBroadcastDims(                               \
        A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);               \
    if (A_broadcast_dims == B_broadcast_dims) {                            \
      auto count = std::accumulate(                                        \
          a_dims, a_dims + a_ndim, 1, std::multiplies<int64_t>());         \
      name(count, a, b, y, ctx);                                           \
      return;                                                              \
    }                                                                      \
    if (utils::math::IsRowwiseBroadcast(                                   \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {               \
      if (broadcast_1st > 0) {                                             \
        _Rowwise##name<TIn, true>(rows, cols, a, b, y);                    \
      } else {                                                             \
        _Rowwise##name<TIn, false>(rows, cols, a, b, y);                   \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    if (utils::math::IsColwiseBroadcast(                                   \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {               \
      if (broadcast_1st > 0) {                                             \
        _Colwise##name<TIn, true>(rows, cols, a, b, y);                    \
      } else {                                                             \
        _Colwise##name<TIn, false>(rows, cols, a, b, y);                   \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;              \
    utils::math::ComputeBinaryBroadcastStrides(                            \
        A_dims, B_dims, A_broadcast_strides, B_broadcast_strides, Y_dims); \
    _Broadcast##name(                                                      \
        Y_dims.size(),                                                     \
        A_broadcast_strides.data(),                                        \
        B_broadcast_strides.data(),                                        \
        Y_dims.data(),                                                     \
        a,                                                                 \
        b,                                                                 \
        y);                                                                \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Add, int, int);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t);
DEFINE_BINARY_FUNC(Add, float, float);
DEFINE_BINARY_FUNC(Add, double, double);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Sub, int, int);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t);
DEFINE_BINARY_FUNC(Sub, float, float);
DEFINE_BINARY_FUNC(Sub, double, double);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Mul, int, int);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t);
DEFINE_BINARY_FUNC(Mul, float, float);
DEFINE_BINARY_FUNC(Mul, double, double);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Div, int, int);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t);
DEFINE_BINARY_FUNC(Div, float, float);
DEFINE_BINARY_FUNC(Div, double, double);
DEFINE_BINARY_FUNC(Pow, float, float);
DEFINE_BINARY_FUNC(Pow, double, double);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Minimum, int, int);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Minimum, float, float);
DEFINE_BINARY_FUNC(Minimum, double, double);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Maximum, int, int);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Maximum, float, float);
DEFINE_BINARY_FUNC(Maximum, double, double);
DEFINE_BINARY_FUNC(Equal, int8_t, bool);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool);
DEFINE_BINARY_FUNC(Equal, int, bool);
DEFINE_BINARY_FUNC(Equal, int64_t, bool);
DEFINE_BINARY_FUNC(Equal, float, bool);
DEFINE_BINARY_FUNC(Equal, double, bool);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, int, bool);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool);
DEFINE_BINARY_FUNC(NotEqual, float, bool);
DEFINE_BINARY_FUNC(NotEqual, double, bool);
DEFINE_BINARY_FUNC(Less, int8_t, bool);
DEFINE_BINARY_FUNC(Less, uint8_t, bool);
DEFINE_BINARY_FUNC(Less, int, bool);
DEFINE_BINARY_FUNC(Less, int64_t, bool);
DEFINE_BINARY_FUNC(Less, float, bool);
DEFINE_BINARY_FUNC(Less, double, bool);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, int, bool);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool);
DEFINE_BINARY_FUNC(LessEqual, float, bool);
DEFINE_BINARY_FUNC(LessEqual, double, bool);
DEFINE_BINARY_FUNC(Greater, int8_t, bool);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool);
DEFINE_BINARY_FUNC(Greater, int, bool);
DEFINE_BINARY_FUNC(Greater, int64_t, bool);
DEFINE_BINARY_FUNC(Greater, float, bool);
DEFINE_BINARY_FUNC(Greater, double, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, T, dtype) \
  template <>                              \
  DRAGON_API void name<T, CPUContext>(     \
      const int a_ndim,                    \
      const int64_t* a_dims,               \
      const int b_ndim,                    \
      const int64_t* b_dims,               \
      const T* a,                          \
      const T* b,                          \
      T* y,                                \
      CPUContext* ctx) {                   \
    name(                                  \
        a_ndim,                            \
        a_dims,                            \
        b_ndim,                            \
        b_dims,                            \
        reinterpret_cast<const dtype*>(a), \
        reinterpret_cast<const dtype*>(b), \
        reinterpret_cast<dtype*>(y),       \
        ctx);                              \
  }

DEFINE_BINARY_FUNC(Add, bool, uint8_t); // Or
DEFINE_BINARY_FUNC(Sub, bool, uint8_t); // Xor
DEFINE_BINARY_FUNC(Mul, bool, uint8_t); // And
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, TOut)       \
  template <>                                \
  DRAGON_API void name<float16, CPUContext>( \
      const int a_ndim,                      \
      const int64_t* a_dims,                 \
      const int b_ndim,                      \
      const int64_t* b_dims,                 \
      const float16* a,                      \
      const float16* b,                      \
      TOut* y,                               \
      CPUContext* ctx) {                     \
    CPU_FP16_NOT_SUPPORTED;                  \
  }

DEFINE_BINARY_FUNC(Add, float16);
DEFINE_BINARY_FUNC(Sub, float16);
DEFINE_BINARY_FUNC(Mul, float16);
DEFINE_BINARY_FUNC(Div, float16);
DEFINE_BINARY_FUNC(Pow, float16);
DEFINE_BINARY_FUNC(Minimum, float16);
DEFINE_BINARY_FUNC(Maximum, float16);
DEFINE_BINARY_FUNC(Equal, bool);
DEFINE_BINARY_FUNC(NotEqual, bool);
DEFINE_BINARY_FUNC(Less, bool);
DEFINE_BINARY_FUNC(LessEqual, bool);
DEFINE_BINARY_FUNC(Greater, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bool);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T)                                                 \
  template <>                                                                \
  DRAGON_API void Where<T, CPUContext>(                                      \
      const int a_ndim,                                                      \
      const int64_t* a_dims,                                                 \
      const int b_ndim,                                                      \
      const int64_t* b_dims,                                                 \
      const int c_ndim,                                                      \
      const int64_t* c_dims,                                                 \
      const T* a,                                                            \
      const T* b,                                                            \
      const bool* c,                                                         \
      T* y,                                                                  \
      CPUContext* ctx) {                                                     \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                                 \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                                 \
    vec64_t C_dims(c_dims, c_dims + c_ndim);                                 \
    vec64_t A_broadcast_dims, B_broadcast_dims, C_broadcast_dims;            \
    vec64_t A_broadcast_strides, B_broadcast_strides, C_broadcast_strides;   \
    vec64_t Y_dims, _, __;                                                   \
    utils::math::ComputeBinaryBroadcastStrides(A_dims, B_dims, _, _, __);    \
    utils::math::ComputeBinaryBroadcastStrides(C_dims, __, _, _, Y_dims);    \
    utils::math::ComputeBinaryBroadcastDims(                                 \
        A_dims, Y_dims, A_broadcast_dims, _);                                \
    utils::math::ComputeBinaryBroadcastDims(                                 \
        B_dims, Y_dims, B_broadcast_dims, _);                                \
    utils::math::ComputeBinaryBroadcastDims(                                 \
        C_dims, Y_dims, C_broadcast_dims, _);                                \
    if (A_broadcast_dims == B_broadcast_dims &&                              \
        B_broadcast_dims == C_broadcast_dims) {                              \
      auto count = std::accumulate(                                          \
          a_dims, a_dims + a_ndim, 1, std::multiplies<int64_t>());           \
      Where(count, a, b, c, y, ctx);                                         \
      return;                                                                \
    }                                                                        \
    utils::math::ComputeBinaryBroadcastStrides(                              \
        A_dims, Y_dims, A_broadcast_strides, _, _);                          \
    utils::math::ComputeBinaryBroadcastStrides(                              \
        B_dims, Y_dims, B_broadcast_strides, _, _);                          \
    utils::math::ComputeBinaryBroadcastStrides(                              \
        C_dims, Y_dims, C_broadcast_strides, _, _);                          \
    const int num_dims = Y_dims.size();                                      \
    const auto count = std::accumulate(                                      \
        Y_dims.begin(), Y_dims.end(), 1, std::multiplies<int64_t>());        \
    vec64_t idx(num_dims, 0);                                                \
    int64_t ai, bi, ci;                                                      \
    for (int yi = 0; yi < count; ++yi) {                                     \
      ai = bi = ci = 0;                                                      \
      for (int d = num_dims - 1; d >= 0; --d) {                              \
        ai += idx[d] * A_broadcast_strides[d];                               \
        bi += idx[d] * B_broadcast_strides[d];                               \
        ci += idx[d] * C_broadcast_strides[d];                               \
      }                                                                      \
      y[yi] = c[ci] ? a[ai] : b[bi];                                         \
      utils::math::IncreaseIndexInDims(num_dims, Y_dims.data(), idx.data()); \
    }                                                                        \
  }

DEFINE_WHERE_FUNC(bool);
DEFINE_WHERE_FUNC(int8_t);
DEFINE_WHERE_FUNC(uint8_t);
DEFINE_WHERE_FUNC(int);
DEFINE_WHERE_FUNC(int64_t);
DEFINE_WHERE_FUNC(float16);
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
