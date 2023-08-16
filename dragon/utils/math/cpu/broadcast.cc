#include "dragon/utils/math/broadcast.h"
#include "dragon/utils/math/blas.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT, class Functor, bool BroadcastA>
void _RowwiseBinaryFunc(
    const int rows,
    const int cols,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int yi = i * cols + j;
      const int ai = BroadcastA ? j : yi;
      const int bi = BroadcastA ? yi : j;
      y[yi] = op(a[ai], b[bi]);
    }
  }
}

template <typename InputT, typename OutputT, class Functor, bool BroadcastA>
void _ColwiseBinaryFunc(
    const int rows,
    const int cols,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int yi = i * cols + j;
      const int ai = BroadcastA ? i : yi;
      const int bi = BroadcastA ? yi : i;
      y[yi] = op(a[ai], b[bi]);
    }
  }
}

template <typename InputT, typename OutputT, class Functor>
void _BroadcastBinaryFunc(
    const int num_dims,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* y_dims,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  const auto N = math::utils::Prod(num_dims, y_dims);
  vec64_t index(num_dims, 0);
  int64_t ai, bi;
  for (int yi = 0; yi < N; ++yi) {
    ai = bi = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      ai += index[d] * a_strides[d];
      bi += index[d] * b_strides[d];
    }
    y[yi] = op(a[ai], b[bi]);
    math::utils::IncreaseIndexInDims(num_dims, y_dims, index.data());
  }
}

#define DECLARE_ROWWISE_COLWISE_BINARY_FUNC(name, OutputT)                 \
  template <typename T, bool BroadcastA>                                   \
  void _Rowwise##name(const int, const int, const T*, const T*, OutputT*); \
  template <typename T, bool BroadcastA>                                   \
  void _Colwise##name(const int, const int, const T*, const T*, OutputT*);

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

#define DEFINE_BROADCAST_1ST_FUNC(name, T, Expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, cols);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y.colwise() Expr## = A;                                         \
    } else {                                                          \
      Y = B.colwise() Expr A;                                         \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, rows);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y.rowwise() Expr## = A.transpose();                             \
    } else {                                                          \
      Y = B.rowwise() Expr A.transpose();                             \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Add, int8_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, uint8_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, int, +);
DEFINE_BROADCAST_1ST_FUNC(Add, int64_t, +);
DEFINE_BROADCAST_1ST_FUNC(Add, float16, +);
DEFINE_BROADCAST_1ST_FUNC(Add, bfloat16, +);
DEFINE_BROADCAST_1ST_FUNC(Add, float, +);
DEFINE_BROADCAST_1ST_FUNC(Add, double, +);
DEFINE_BROADCAST_1ST_FUNC(Mul, int8_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, uint8_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, int, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, int64_t, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, float16, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, bfloat16, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, float, *);
DEFINE_BROADCAST_1ST_FUNC(Mul, double, *);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_1ST_FUNC(name, T, Expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, cols);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y *= EigenT(-1);                                                \
      Y.colwise() += A;                                               \
    } else {                                                          \
      Y = (-B).colwise() + A;                                         \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, rows);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y *= EigenT(-1);                                                \
      Y.rowwise() += A.transpose();                                   \
    } else {                                                          \
      Y = (-B).rowwise() + A.transpose();                             \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Sub, int8_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, uint8_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, int, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, int64_t, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, float16, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, bfloat16, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, float, -);
DEFINE_BROADCAST_1ST_FUNC(Sub, double, -);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_1ST_FUNC(name, T, Expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, cols);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y = Y.inverse();                                                \
      Y.colwise() *= A;                                               \
    } else {                                                          \
      Y = B.inverse().colwise() * A;                                  \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, rows);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    if (b == y) {                                                     \
      Y = Y.inverse();                                                \
      Y.rowwise() *= A.transpose();                                   \
    } else {                                                          \
      Y = B.inverse().rowwise() * A.transpose();                      \
    }                                                                 \
  }

DEFINE_BROADCAST_1ST_FUNC(Div, int8_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, uint8_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, int, /);
DEFINE_BROADCAST_1ST_FUNC(Div, int64_t, /);
DEFINE_BROADCAST_1ST_FUNC(Div, float16, /);
DEFINE_BROADCAST_1ST_FUNC(Div, bfloat16, /);
DEFINE_BROADCAST_1ST_FUNC(Div, float, /);
DEFINE_BROADCAST_1ST_FUNC(Div, double, /);
#undef DEFINE_BROADCAST_1ST_FUNC

#define DEFINE_BROADCAST_2ND_FUNC(name, T, Expr)                      \
  template <>                                                         \
  void _Rowwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenArrayMap<EigenT> A((const EigenT*)a, cols, rows);       \
    ConstEigenVectorArrayMap<EigenT> B((const EigenT*)b, cols);       \
    if (a == y) {                                                     \
      Y.colwise() Expr## = B;                                         \
    } else {                                                          \
      Y = A.colwise() Expr B;                                         \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenArrayMap<EigenT> A((const EigenT*)a, cols, rows);       \
    ConstEigenVectorArrayMap2<EigenT> B((const EigenT*)b, rows);      \
    if (a == y) {                                                     \
      Y.rowwise() Expr## = B;                                         \
    } else {                                                          \
      Y = A.rowwise() Expr B;                                         \
    }                                                                 \
  }

DEFINE_BROADCAST_2ND_FUNC(Add, int8_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, uint8_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, int, +);
DEFINE_BROADCAST_2ND_FUNC(Add, int64_t, +);
DEFINE_BROADCAST_2ND_FUNC(Add, float16, +);
DEFINE_BROADCAST_2ND_FUNC(Add, bfloat16, +);
DEFINE_BROADCAST_2ND_FUNC(Add, float, +);
DEFINE_BROADCAST_2ND_FUNC(Add, double, +);
DEFINE_BROADCAST_2ND_FUNC(Sub, int8_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, uint8_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, int, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, int64_t, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, float16, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, bfloat16, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, float, -);
DEFINE_BROADCAST_2ND_FUNC(Sub, double, -);
DEFINE_BROADCAST_2ND_FUNC(Mul, int8_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, uint8_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, int, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, int64_t, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, float16, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, bfloat16, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, float, *);
DEFINE_BROADCAST_2ND_FUNC(Mul, double, *);
DEFINE_BROADCAST_2ND_FUNC(Div, int8_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, uint8_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, int, /);
DEFINE_BROADCAST_2ND_FUNC(Div, int64_t, /);
DEFINE_BROADCAST_2ND_FUNC(Div, float16, /);
DEFINE_BROADCAST_2ND_FUNC(Div, bfloat16, /);
DEFINE_BROADCAST_2ND_FUNC(Div, float, /);
DEFINE_BROADCAST_2ND_FUNC(Div, double, /);
#undef DEFINE_BROADCAST_2ND_FUNC

#define DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(name, InputT, OutputT, Expr)    \
  template <>                                                              \
  void _Rowwise##name<InputT, true>(                                       \
      const int rows,                                                      \
      const int cols,                                                      \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y) {                                                        \
    using EigenInputT = math::Traits<InputT>::eigen_type;                  \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;                \
    EigenArrayMap<EigenOutputT> Y((EigenOutputT*)y, cols, rows);           \
    ConstEigenVectorArrayMap<EigenInputT> A((const EigenInputT*)a, cols);  \
    ConstEigenArrayMap<EigenInputT> B((const EigenInputT*)b, cols, rows);  \
    Y = A.rowwise().replicate(rows) Expr B;                                \
  }                                                                        \
  template <>                                                              \
  void _Colwise##name<InputT, true>(                                       \
      const int rows,                                                      \
      const int cols,                                                      \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y) {                                                        \
    using EigenInputT = math::Traits<InputT>::eigen_type;                  \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;                \
    EigenArrayMap<EigenOutputT> Y((EigenOutputT*)y, cols, rows);           \
    ConstEigenVectorArrayMap2<EigenInputT> A((const EigenInputT*)a, rows); \
    ConstEigenArrayMap<EigenInputT> B((const EigenInputT*)b, cols, rows);  \
    Y = A.colwise().replicate(cols) Expr B;                                \
  }                                                                        \
  template <>                                                              \
  void _Rowwise##name<InputT, false>(                                      \
      const int rows,                                                      \
      const int cols,                                                      \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y) {                                                        \
    using EigenInputT = math::Traits<InputT>::eigen_type;                  \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;                \
    EigenArrayMap<EigenOutputT> Y((EigenOutputT*)y, cols, rows);           \
    ConstEigenArrayMap<EigenInputT> A((const EigenInputT*)a, cols, rows);  \
    ConstEigenVectorArrayMap<EigenInputT> B((const EigenInputT*)b, cols);  \
    Y = A Expr B.rowwise().replicate(rows);                                \
  }                                                                        \
  template <>                                                              \
  void _Colwise##name<InputT, false>(                                      \
      const int rows,                                                      \
      const int cols,                                                      \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y) {                                                        \
    using EigenInputT = math::Traits<InputT>::eigen_type;                  \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;                \
    EigenArrayMap<EigenOutputT> Y((EigenOutputT*)y, cols, rows);           \
    ConstEigenArrayMap<EigenInputT> A((const EigenInputT*)a, cols, rows);  \
    ConstEigenVectorArrayMap2<EigenInputT> B((const EigenInputT*)b, rows); \
    Y = A Expr B.colwise().replicate(cols);                                \
  }

DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, bool, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, uint8_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int8_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, int64_t, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, float16, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, bfloat16, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, float, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Equal, double, bool, ==);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, bool, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, uint8_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int8_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, int64_t, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, float16, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, bfloat16, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, float, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(NotEqual, double, bool, !=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, bool, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, uint8_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int8_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, int64_t, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, float16, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, bfloat16, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, float, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Less, double, bool, <);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, bool, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, uint8_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int8_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, int64_t, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, float16, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, bfloat16, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, float, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(LessEqual, double, bool, <=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, bool, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, uint8_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int8_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, int64_t, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, float16, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, bfloat16, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, float, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Greater, double, bool, >);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, bool, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, uint8_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int8_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, int64_t, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, float16, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, bfloat16, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, float, bool, >=);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(GreaterEqual, double, bool, >=);
#undef DEFINE_ROWWISE_COLWISE_BIANRY_FUNC

#define DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(name, T, Func)             \
  template <>                                                         \
  void _Rowwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap<EigenT> A((const EigenT*)a, cols);       \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    Y = A.rowwise().replicate(rows).Func(B);                          \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, true>(                                       \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenVectorArrayMap2<EigenT> A((const EigenT*)a, rows);      \
    ConstEigenArrayMap<EigenT> B((const EigenT*)b, cols, rows);       \
    Y = A.colwise().replicate(cols).Func(B);                          \
  }                                                                   \
  template <>                                                         \
  void _Rowwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenArrayMap<EigenT> A((const EigenT*)a, cols, rows);       \
    ConstEigenVectorArrayMap<EigenT> B((const EigenT*)b, cols);       \
    Y = A.Func(B.rowwise().replicate(rows));                          \
  }                                                                   \
  template <>                                                         \
  void _Colwise##name<T, false>(                                      \
      const int rows, const int cols, const T* a, const T* b, T* y) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenArrayMap<EigenT> Y((EigenT*)y, cols, rows);                  \
    ConstEigenArrayMap<EigenT> A((const EigenT*)a, cols, rows);       \
    ConstEigenVectorArrayMap2<EigenT> B((const EigenT*)b, rows);      \
    Y = A.Func(B.colwise().replicate(cols));                          \
  }

DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, float16, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, bfloat16, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, float, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Pow, double, pow);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int8_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, uint8_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, int64_t, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, float16, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, bfloat16, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, float, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Minimum, double, min);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int8_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, uint8_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, int64_t, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, float16, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, bfloat16, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, float, max);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Maximum, double, max);
#undef DEFINE_ROWWISE_COLWISE_BIANRY_FUNC

#define DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(name, OutputT, Functor)          \
  template <typename T, bool BroadcastA>                                    \
  void _Rowwise##name(                                                      \
      const int rows, const int cols, const T* a, const T* b, OutputT* y) { \
    _RowwiseBinaryFunc<T, OutputT, Functor<T>, BroadcastA>(                 \
        rows, cols, Functor<T>(), a, b, y);                                 \
  }                                                                         \
  template <typename T, bool BroadcastA>                                    \
  void _Colwise##name(                                                      \
      const int rows, const int cols, const T* a, const T* b, OutputT* y) { \
    _ColwiseBinaryFunc<T, OutputT, Functor<T>, BroadcastA>(                 \
        rows, cols, Functor<T>(), a, b, y);                                 \
  }

DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Atan2, T, math::Atan2Functor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(BitwiseAnd, T, math::BitAndFunctor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(BitwiseOr, T, math::BitOrFunctor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(BitwiseXor, T, math::BitXorFunctor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(And, bool, math::AndFunctor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Or, bool, math::OrFunctor);
DEFINE_ROWWISE_COLWISE_BIANRY_FUNC(Xor, bool, math::XorFunctor);
#undef DEFINE_ROWWISE_COLWISE_BIANRY_FUNC

#define DEFINE_BROADCAST_BINARY_FUNC(name, OutputT, Functor)            \
  template <typename T>                                                 \
  void _Broadcast##name(                                                \
      const int num_dims,                                               \
      const int64_t* a_strides,                                         \
      const int64_t* b_strides,                                         \
      const int64_t* y_dims,                                            \
      const T* a,                                                       \
      const T* b,                                                       \
      OutputT* y) {                                                     \
    _BroadcastBinaryFunc(                                               \
        num_dims, a_strides, b_strides, y_dims, Functor<T>(), a, b, y); \
  }

DEFINE_BROADCAST_BINARY_FUNC(Add, T, math::PlusFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Sub, T, math::MinusFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Mul, T, math::MultipliesFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Div, T, math::DividesFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Pow, T, math::PowFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Minimum, T, math::MinFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Maximum, T, math::MaxFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Atan2, T, math::Atan2Functor);
DEFINE_BROADCAST_BINARY_FUNC(BitwiseAnd, T, math::BitAndFunctor);
DEFINE_BROADCAST_BINARY_FUNC(BitwiseOr, T, math::BitOrFunctor);
DEFINE_BROADCAST_BINARY_FUNC(BitwiseXor, T, math::BitXorFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Equal, bool, math::EqualFunctor);
DEFINE_BROADCAST_BINARY_FUNC(NotEqual, bool, math::NotEqualFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Less, bool, math::LessFunctor);
DEFINE_BROADCAST_BINARY_FUNC(LessEqual, bool, math::LessEqualFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Greater, bool, math::GreaterFunctor);
DEFINE_BROADCAST_BINARY_FUNC(GreaterEqual, bool, math::GreaterEqualFunctor);
DEFINE_BROADCAST_BINARY_FUNC(And, bool, math::AndFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Or, bool, math::OrFunctor);
DEFINE_BROADCAST_BINARY_FUNC(Xor, bool, math::XorFunctor);
#undef DEFINE_BROADCAST_BINARY_FUNC

} // namespace

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
    int64_t rows, cols, broadcast_1st;                                       \
    vec64_t X_dims(x_dims, x_dims + x_ndim);                                 \
    vec64_t Y_dims(y_dims, y_dims + y_ndim);                                 \
    vec64_t X_broadcast_dims, Y_broadcast_dims;                              \
    math::utils::ComputeBroadcastDims(                                       \
        X_dims, Y_dims, X_broadcast_dims, Y_broadcast_dims);                 \
    if (X_broadcast_dims == Y_broadcast_dims) {                              \
      const auto N = math::utils::Prod(x_ndim, x_dims);                      \
      Copy(N, x, y, ctx);                                                    \
      return;                                                                \
    }                                                                        \
    if (math::utils::IsRowwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {     \
      EigenArrayMap<T>(y, cols, rows).colwise() =                            \
          ConstEigenVectorArrayMap<T>(x, cols);                              \
      return;                                                                \
    }                                                                        \
    if (math::utils::IsColwiseBroadcast(X_dims, Y_dims, &rows, &cols)) {     \
      EigenArrayMap<T>(y, cols, rows).rowwise() =                            \
          ConstEigenVectorArrayMap<T>(x, rows).transpose();                  \
      return;                                                                \
    }                                                                        \
    vec64_t X_broadcast_strides, _;                                          \
    math::utils::ComputeBroadcastStrides(                                    \
        X_dims, Y_dims, X_broadcast_strides, _, _);                          \
    const int num_dims = Y_dims.size();                                      \
    const auto N = math::utils::Prod(Y_dims);                                \
    vec64_t idx(num_dims, 0);                                                \
    int64_t xi;                                                              \
    for (int yi = 0; yi < N; ++yi) {                                         \
      xi = 0;                                                                \
      for (int d = num_dims - 1; d >= 0; --d) {                              \
        xi += idx[d] * X_broadcast_strides[d];                               \
      }                                                                      \
      y[yi] = x[xi];                                                         \
      math::utils::IncreaseIndexInDims(num_dims, Y_dims.data(), idx.data()); \
    }                                                                        \
  }

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float16);
DEFINE_SET_FUNC(bfloat16);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT)                          \
  template <>                                                              \
  DRAGON_API void name<InputT, CPUContext>(                                \
      const int a_ndim,                                                    \
      const int64_t* a_dims,                                               \
      const int b_ndim,                                                    \
      const int64_t* b_dims,                                               \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y,                                                          \
      CPUContext* ctx) {                                                   \
    int64_t rows, cols, broadcast_1st;                                     \
    vec64_t A_dims(a_dims, a_dims + a_ndim);                               \
    vec64_t B_dims(b_dims, b_dims + b_ndim);                               \
    vec64_t A_broadcast_dims, B_broadcast_dims;                            \
    math::utils::ComputeBroadcastDims(                                     \
        A_dims, B_dims, A_broadcast_dims, B_broadcast_dims);               \
    if (A_broadcast_dims == B_broadcast_dims) {                            \
      const auto N = math::utils::Prod(a_ndim, a_dims);                    \
      name(N, a, b, y, ctx);                                               \
      return;                                                              \
    }                                                                      \
    if (math::utils::IsRowwiseBroadcast(                                   \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {               \
      if (broadcast_1st > 0) {                                             \
        _Rowwise##name<InputT, true>(rows, cols, a, b, y);                 \
      } else {                                                             \
        _Rowwise##name<InputT, false>(rows, cols, a, b, y);                \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    if (math::utils::IsColwiseBroadcast(                                   \
            A_dims, B_dims, &rows, &cols, &broadcast_1st)) {               \
      if (broadcast_1st > 0) {                                             \
        _Colwise##name<InputT, true>(rows, cols, a, b, y);                 \
      } else {                                                             \
        _Colwise##name<InputT, false>(rows, cols, a, b, y);                \
      }                                                                    \
      return;                                                              \
    }                                                                      \
    vec64_t A_broadcast_strides, B_broadcast_strides, Y_dims;              \
    math::utils::ComputeBroadcastStrides(                                  \
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

DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Add, int8_t, int8_t);
DEFINE_BINARY_FUNC(Add, int, int);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t);
DEFINE_BINARY_FUNC(Add, float16, float16);
DEFINE_BINARY_FUNC(Add, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Add, float, float);
DEFINE_BINARY_FUNC(Add, double, double);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t);
DEFINE_BINARY_FUNC(Sub, int, int);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t);
DEFINE_BINARY_FUNC(Sub, float16, float16);
DEFINE_BINARY_FUNC(Sub, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Sub, float, float);
DEFINE_BINARY_FUNC(Sub, double, double);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t);
DEFINE_BINARY_FUNC(Mul, int, int);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t);
DEFINE_BINARY_FUNC(Mul, float16, float16);
DEFINE_BINARY_FUNC(Mul, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Mul, float, float);
DEFINE_BINARY_FUNC(Mul, double, double);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t);
DEFINE_BINARY_FUNC(Div, int, int);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t);
DEFINE_BINARY_FUNC(Div, float16, float16);
DEFINE_BINARY_FUNC(Div, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Div, float, float);
DEFINE_BINARY_FUNC(Div, double, double);
DEFINE_BINARY_FUNC(Pow, float16, float16);
DEFINE_BINARY_FUNC(Pow, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Pow, float, float);
DEFINE_BINARY_FUNC(Pow, double, double);
DEFINE_BINARY_FUNC(Atan2, float16, float16);
DEFINE_BINARY_FUNC(Atan2, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Atan2, float, float);
DEFINE_BINARY_FUNC(Atan2, double, double);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Minimum, int, int);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Minimum, float16, float16);
DEFINE_BINARY_FUNC(Minimum, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Minimum, float, float);
DEFINE_BINARY_FUNC(Minimum, double, double);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t);
DEFINE_BINARY_FUNC(Maximum, int, int);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t);
DEFINE_BINARY_FUNC(Maximum, float16, float16);
DEFINE_BINARY_FUNC(Maximum, bfloat16, bfloat16);
DEFINE_BINARY_FUNC(Maximum, float, float);
DEFINE_BINARY_FUNC(Maximum, double, double);
DEFINE_BINARY_FUNC(BitwiseAnd, bool, bool);
DEFINE_BINARY_FUNC(BitwiseAnd, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseAnd, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseAnd, int, int);
DEFINE_BINARY_FUNC(BitwiseAnd, int64_t, int64_t);
DEFINE_BINARY_FUNC(BitwiseOr, bool, bool);
DEFINE_BINARY_FUNC(BitwiseOr, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseOr, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseOr, int, int);
DEFINE_BINARY_FUNC(BitwiseOr, int64_t, int64_t);
DEFINE_BINARY_FUNC(BitwiseXor, bool, bool);
DEFINE_BINARY_FUNC(BitwiseXor, uint8_t, uint8_t);
DEFINE_BINARY_FUNC(BitwiseXor, int8_t, int8_t);
DEFINE_BINARY_FUNC(BitwiseXor, int, int);
DEFINE_BINARY_FUNC(BitwiseXor, int64_t, int64_t);
DEFINE_BINARY_FUNC(And, bool, bool);
DEFINE_BINARY_FUNC(And, uint8_t, bool);
DEFINE_BINARY_FUNC(And, int8_t, bool);
DEFINE_BINARY_FUNC(And, int, bool);
DEFINE_BINARY_FUNC(And, int64_t, bool);
DEFINE_BINARY_FUNC(And, float16, bool);
DEFINE_BINARY_FUNC(And, bfloat16, bool);
DEFINE_BINARY_FUNC(And, float, bool);
DEFINE_BINARY_FUNC(And, double, bool);
DEFINE_BINARY_FUNC(Or, bool, bool);
DEFINE_BINARY_FUNC(Or, uint8_t, bool);
DEFINE_BINARY_FUNC(Or, int8_t, bool);
DEFINE_BINARY_FUNC(Or, int, bool);
DEFINE_BINARY_FUNC(Or, int64_t, bool);
DEFINE_BINARY_FUNC(Or, float16, bool);
DEFINE_BINARY_FUNC(Or, bfloat16, bool);
DEFINE_BINARY_FUNC(Or, float, bool);
DEFINE_BINARY_FUNC(Or, double, bool);
DEFINE_BINARY_FUNC(Xor, bool, bool);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool);
DEFINE_BINARY_FUNC(Xor, int8_t, bool);
DEFINE_BINARY_FUNC(Xor, int, bool);
DEFINE_BINARY_FUNC(Xor, int64_t, bool);
DEFINE_BINARY_FUNC(Xor, float16, bool);
DEFINE_BINARY_FUNC(Xor, bfloat16, bool);
DEFINE_BINARY_FUNC(Xor, float, bool);
DEFINE_BINARY_FUNC(Xor, double, bool);
DEFINE_BINARY_FUNC(Equal, bool, bool);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool);
DEFINE_BINARY_FUNC(Equal, int8_t, bool);
DEFINE_BINARY_FUNC(Equal, int, bool);
DEFINE_BINARY_FUNC(Equal, int64_t, bool);
DEFINE_BINARY_FUNC(Equal, float16, bool);
DEFINE_BINARY_FUNC(Equal, bfloat16, bool);
DEFINE_BINARY_FUNC(Equal, float, bool);
DEFINE_BINARY_FUNC(Equal, double, bool);
DEFINE_BINARY_FUNC(NotEqual, bool, bool);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool);
DEFINE_BINARY_FUNC(NotEqual, int, bool);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool);
DEFINE_BINARY_FUNC(NotEqual, float16, bool);
DEFINE_BINARY_FUNC(NotEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(NotEqual, float, bool);
DEFINE_BINARY_FUNC(NotEqual, double, bool);
DEFINE_BINARY_FUNC(Less, bool, bool);
DEFINE_BINARY_FUNC(Less, uint8_t, bool);
DEFINE_BINARY_FUNC(Less, int8_t, bool);
DEFINE_BINARY_FUNC(Less, int, bool);
DEFINE_BINARY_FUNC(Less, int64_t, bool);
DEFINE_BINARY_FUNC(Less, float16, bool);
DEFINE_BINARY_FUNC(Less, bfloat16, bool);
DEFINE_BINARY_FUNC(Less, float, bool);
DEFINE_BINARY_FUNC(Less, double, bool);
DEFINE_BINARY_FUNC(LessEqual, bool, bool);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool);
DEFINE_BINARY_FUNC(LessEqual, int, bool);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool);
DEFINE_BINARY_FUNC(LessEqual, float16, bool);
DEFINE_BINARY_FUNC(LessEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(LessEqual, float, bool);
DEFINE_BINARY_FUNC(LessEqual, double, bool);
DEFINE_BINARY_FUNC(Greater, bool, bool);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool);
DEFINE_BINARY_FUNC(Greater, int8_t, bool);
DEFINE_BINARY_FUNC(Greater, int, bool);
DEFINE_BINARY_FUNC(Greater, int64_t, bool);
DEFINE_BINARY_FUNC(Greater, float16, bool);
DEFINE_BINARY_FUNC(Greater, bfloat16, bool);
DEFINE_BINARY_FUNC(Greater, float, bool);
DEFINE_BINARY_FUNC(Greater, double, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bfloat16, bool);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool);
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
    math::utils::ComputeBroadcastStrides(A_dims, B_dims, _, _, __);          \
    math::utils::ComputeBroadcastStrides(C_dims, __, _, _, Y_dims);          \
    math::utils::ComputeBroadcastDims(A_dims, Y_dims, A_broadcast_dims, _);  \
    math::utils::ComputeBroadcastDims(B_dims, Y_dims, B_broadcast_dims, _);  \
    math::utils::ComputeBroadcastDims(C_dims, Y_dims, C_broadcast_dims, _);  \
    if (A_broadcast_dims == B_broadcast_dims &&                              \
        B_broadcast_dims == C_broadcast_dims) {                              \
      const auto N = math::utils::Prod(a_ndim, a_dims);                      \
      Where(N, a, b, c, y, ctx);                                             \
      return;                                                                \
    }                                                                        \
    math::utils::ComputeBroadcastStrides(                                    \
        A_dims, Y_dims, A_broadcast_strides, _, _);                          \
    math::utils::ComputeBroadcastStrides(                                    \
        B_dims, Y_dims, B_broadcast_strides, _, _);                          \
    math::utils::ComputeBroadcastStrides(                                    \
        C_dims, Y_dims, C_broadcast_strides, _, _);                          \
    const int num_dims = Y_dims.size();                                      \
    const auto N = math::utils::Prod(Y_dims);                                \
    vec64_t idx(num_dims, 0);                                                \
    int64_t ai, bi, ci;                                                      \
    for (int yi = 0; yi < N; ++yi) {                                         \
      ai = bi = ci = 0;                                                      \
      for (int d = num_dims - 1; d >= 0; --d) {                              \
        ai += idx[d] * A_broadcast_strides[d];                               \
        bi += idx[d] * B_broadcast_strides[d];                               \
        ci += idx[d] * C_broadcast_strides[d];                               \
      }                                                                      \
      y[yi] = c[ci] ? a[ai] : b[bi];                                         \
      math::utils::IncreaseIndexInDims(num_dims, Y_dims.data(), idx.data()); \
    }                                                                        \
  }

DEFINE_WHERE_FUNC(bool);
DEFINE_WHERE_FUNC(int8_t);
DEFINE_WHERE_FUNC(uint8_t);
DEFINE_WHERE_FUNC(int);
DEFINE_WHERE_FUNC(int64_t);
DEFINE_WHERE_FUNC(float16);
DEFINE_WHERE_FUNC(bfloat16);
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
