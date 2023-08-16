#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace math {

namespace {

template <typename InputT, typename OutputT, class Functor>
void _SimpleUnaryFunc(
    const int N,
    const Functor op,
    const InputT* x,
    OutputT* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = op(x[i]);
  }
}

template <typename InputT, typename OutputT, class Functor>
void _SimpleBinaryFunc(
    const int N,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = op(a[i], b[i]);
  }
}

} // namespace

#define DEFINE_UNARY_FUNC(name, T, Expr)                              \
  template <>                                                         \
  DRAGON_API void name<T, CPUContext>(                                \
      const int N, const T* x, T* y, CPUContext* ctx) {               \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                      \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N).Expr(); \
  }

DEFINE_UNARY_FUNC(Abs, int8_t, abs);
DEFINE_UNARY_FUNC(Abs, int, abs);
DEFINE_UNARY_FUNC(Abs, int64_t, abs);
DEFINE_UNARY_FUNC(Abs, float16, abs);
DEFINE_UNARY_FUNC(Abs, bfloat16, abs);
DEFINE_UNARY_FUNC(Abs, float, abs);
DEFINE_UNARY_FUNC(Abs, double, abs);
DEFINE_UNARY_FUNC(Square, uint8_t, square);
DEFINE_UNARY_FUNC(Square, int8_t, square);
DEFINE_UNARY_FUNC(Square, int, square);
DEFINE_UNARY_FUNC(Square, int64_t, square);
DEFINE_UNARY_FUNC(Square, float16, square);
DEFINE_UNARY_FUNC(Square, bfloat16, square);
DEFINE_UNARY_FUNC(Square, float, square);
DEFINE_UNARY_FUNC(Square, double, square);
DEFINE_UNARY_FUNC(Ceil, float16, ceil);
DEFINE_UNARY_FUNC(Ceil, bfloat16, ceil);
DEFINE_UNARY_FUNC(Ceil, float, ceil);
DEFINE_UNARY_FUNC(Ceil, double, ceil);
DEFINE_UNARY_FUNC(Floor, float16, floor);
DEFINE_UNARY_FUNC(Floor, bfloat16, floor);
DEFINE_UNARY_FUNC(Floor, float, floor);
DEFINE_UNARY_FUNC(Floor, double, floor);
DEFINE_UNARY_FUNC(Round, float16, round);
DEFINE_UNARY_FUNC(Round, bfloat16, round);
DEFINE_UNARY_FUNC(Round, float, round);
DEFINE_UNARY_FUNC(Round, double, round);
DEFINE_UNARY_FUNC(Exp, float16, exp);
DEFINE_UNARY_FUNC(Exp, bfloat16, exp);
DEFINE_UNARY_FUNC(Exp, float, exp);
DEFINE_UNARY_FUNC(Exp, double, exp);
DEFINE_UNARY_FUNC(Log, float16, log);
DEFINE_UNARY_FUNC(Log, bfloat16, log);
DEFINE_UNARY_FUNC(Log, float, log);
DEFINE_UNARY_FUNC(Log, double, log);
DEFINE_UNARY_FUNC(Inv, float16, inverse);
DEFINE_UNARY_FUNC(Inv, bfloat16, inverse);
DEFINE_UNARY_FUNC(Inv, float, inverse);
DEFINE_UNARY_FUNC(Inv, double, inverse);
DEFINE_UNARY_FUNC(Sqrt, float16, sqrt);
DEFINE_UNARY_FUNC(Sqrt, bfloat16, sqrt);
DEFINE_UNARY_FUNC(Sqrt, float, sqrt);
DEFINE_UNARY_FUNC(Sqrt, double, sqrt);
DEFINE_UNARY_FUNC(Rsqrt, float16, rsqrt);
DEFINE_UNARY_FUNC(Rsqrt, bfloat16, rsqrt);
DEFINE_UNARY_FUNC(Rsqrt, float, rsqrt);
DEFINE_UNARY_FUNC(Rsqrt, double, rsqrt);
DEFINE_UNARY_FUNC(Sin, float16, sin);
DEFINE_UNARY_FUNC(Sin, bfloat16, sin);
DEFINE_UNARY_FUNC(Sin, float, sin);
DEFINE_UNARY_FUNC(Sin, double, sin);
DEFINE_UNARY_FUNC(Cos, float16, cos);
DEFINE_UNARY_FUNC(Cos, bfloat16, cos);
DEFINE_UNARY_FUNC(Cos, float, cos);
DEFINE_UNARY_FUNC(Cos, double, cos);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, InputT, OutputT, Functor)          \
  template <>                                                      \
  DRAGON_API void name<InputT, CPUContext>(                        \
      const int N, const InputT* x, OutputT* y, CPUContext* ctx) { \
    _SimpleUnaryFunc(N, Functor<InputT>(), x, y);                  \
  }

DEFINE_UNARY_FUNC(Sign, uint8_t, uint8_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int8_t, int8_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int, int, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int64_t, int64_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, float16, float16, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, bfloat16, bfloat16, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, float, float, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, double, double, math::SignFunctor);
DEFINE_UNARY_FUNC(BitwiseNot, bool, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(BitwiseNot, uint8_t, uint8_t, math::BitNotFunctor);
DEFINE_UNARY_FUNC(BitwiseNot, int8_t, int8_t, math::BitNotFunctor);
DEFINE_UNARY_FUNC(BitwiseNot, int, int, math::BitNotFunctor);
DEFINE_UNARY_FUNC(BitwiseNot, int64_t, int64_t, math::BitNotFunctor);
DEFINE_UNARY_FUNC(Not, bool, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, uint8_t, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, int8_t, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, int, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, int64_t, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, float16, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, bfloat16, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, float, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, double, bool, math::NotFunctor);
#undef DEFINE_UNARY_FUNC

#define DEFINE_NEG_FUNC(T)                                      \
  template <>                                                   \
  DRAGON_API void Neg<T, CPUContext>(                           \
      const int N, const T* x, T* y, CPUContext* ctx) {         \
    using EigenT = math::Traits<T>::eigen_type;                 \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                \
        -ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N); \
  }

DEFINE_NEG_FUNC(int8_t);
DEFINE_NEG_FUNC(int);
DEFINE_NEG_FUNC(int64_t);
DEFINE_NEG_FUNC(float16);
DEFINE_NEG_FUNC(bfloat16);
DEFINE_NEG_FUNC(float);
DEFINE_NEG_FUNC(double);
#undef DEFINE_NEG_FUNC

#define DEFINE_SET_FUNC(T)                                 \
  template <>                                              \
  DRAGON_API void Set<T, CPUContext>(                      \
      const int N, const T value, T* y, CPUContext* ctx) { \
    if (N == 0) return;                                    \
    if (value == T(0)) {                                   \
      memset(y, 0, sizeof(T) * N);                         \
    } else {                                               \
      EigenVectorMap<T>(y, N).setConstant(value);          \
    }                                                      \
  }

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_SET_FUNC(T)                                 \
  template <>                                              \
  DRAGON_API void Set<T, CPUContext>(                      \
      const int N, const T value, T* y, CPUContext* ctx) { \
    if (N == 0) return;                                    \
    if (value.x == (unsigned short)0) {                    \
      memset(y, 0, sizeof(T) * N);                         \
    } else {                                               \
      EigenVectorMap<T>(y, N).setConstant(value);          \
    }                                                      \
  }

DEFINE_SET_FUNC(float16);
DEFINE_SET_FUNC(bfloat16);
#undef DEFINE_SET_FUNC

#define DEFINE_INVSTD_FUNC(T)                                                 \
  template <>                                                                 \
  DRAGON_API void InvStd<T, CPUContext>(                                      \
      const int N, const float eps, const T* x, T* y, CPUContext* ctx) {      \
    using EigenT = math::Traits<T>::eigen_type;                               \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                              \
        (ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N) + EigenT(eps)) \
            .rsqrt();                                                         \
  }

DEFINE_INVSTD_FUNC(float16);
DEFINE_INVSTD_FUNC(bfloat16);
DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

#define DEFINE_POWX_FUNC(T)                                                   \
  template <>                                                                 \
  DRAGON_API void Powx<T, CPUContext>(                                        \
      const int N, const float exponent, const T* x, T* y, CPUContext* ctx) { \
    using EigenT = math::Traits<T>::eigen_type;                               \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                              \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N)                 \
            .pow(EigenT(exponent));                                           \
  }

DEFINE_POWX_FUNC(float16);
DEFINE_POWX_FUNC(bfloat16);
DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

#define DEFINE_NOT_ZERO_FUNC(T)                                             \
  template <>                                                               \
  DRAGON_API void NotZero<T, CPUContext>(                                   \
      const int N, const T* x, bool* y, CPUContext* ctx) {                  \
    using EigenT = math::Traits<T>::eigen_type;                             \
    EigenVectorArrayMap<bool>(y, N) =                                       \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N) != EigenT(0); \
  }

DEFINE_NOT_ZERO_FUNC(bool);
DEFINE_NOT_ZERO_FUNC(uint8_t);
DEFINE_NOT_ZERO_FUNC(int8_t);
DEFINE_NOT_ZERO_FUNC(int);
DEFINE_NOT_ZERO_FUNC(int64_t);
DEFINE_NOT_ZERO_FUNC(float16);
DEFINE_NOT_ZERO_FUNC(bfloat16);
DEFINE_NOT_ZERO_FUNC(float);
DEFINE_NOT_ZERO_FUNC(double);
#undef DEFINE_NOT_ZERO_FUNC

#define DEFINE_IS_FUNC(name, T, Expr)                                 \
  template <>                                                         \
  DRAGON_API void name<T, CPUContext>(                                \
      const int N, const T* x, bool* y, CPUContext* ctx) {            \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenVectorArrayMap<bool>(y, N) =                                 \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N).Expr(); \
  }

DEFINE_IS_FUNC(IsInf, float16, isInf);
DEFINE_IS_FUNC(IsInf, bfloat16, isInf);
DEFINE_IS_FUNC(IsInf, float, isInf);
DEFINE_IS_FUNC(IsInf, double, isInf);
DEFINE_IS_FUNC(IsNaN, float16, isNaN);
DEFINE_IS_FUNC(IsNaN, bfloat16, isNaN);
DEFINE_IS_FUNC(IsNaN, float, isNaN);
DEFINE_IS_FUNC(IsNaN, double, isNaN);
DEFINE_IS_FUNC(IsFinite, float16, isFinite);
DEFINE_IS_FUNC(IsFinite, bfloat16, isFinite);
DEFINE_IS_FUNC(IsFinite, float, isFinite);
DEFINE_IS_FUNC(IsFinite, double, isFinite);
#undef DEFINE_IS_FUNC

#define DEFINE_NAN_TO_NUM_FUNC(T)                                        \
  template <>                                                            \
  DRAGON_API void NaNToNum<T, CPUContext>(                               \
      const int N, const float nan, const T* x, T* y, CPUContext* ctx) { \
    using EigenT = math::Traits<T>::eigen_type;                          \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);             \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);                        \
    Y = X.isNaN().select(EigenT(nan), X);                                \
  }

DEFINE_NAN_TO_NUM_FUNC(float16);
DEFINE_NAN_TO_NUM_FUNC(bfloat16);
DEFINE_NAN_TO_NUM_FUNC(float);
DEFINE_NAN_TO_NUM_FUNC(double);
#undef DEFINE_NAN_TO_NUM_FUNC

#define DEFINE_NAN_TO_NUM_FUNC(T)                                              \
  template <>                                                                  \
  DRAGON_API void NaNToNum<T, CPUContext>(                                     \
      const int N,                                                             \
      const float nan,                                                         \
      const float pos_inf,                                                     \
      const float neg_inf,                                                     \
      const T* x,                                                              \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    using EigenT = math::Traits<T>::eigen_type;                                \
    auto posinf = EigenT(std::min(pos_inf, float(math::Traits<T>::Max())));    \
    auto neginf = EigenT(std::max(neg_inf, float(math::Traits<T>::Lowest()))); \
    auto to = [&](EigenT v) { return v > EigenT(0) ? posinf : neginf; };       \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);                   \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);                              \
    Y = X.isNaN().select(EigenT(nan), X.isInf().select(X.unaryExpr(to), X));   \
  }

DEFINE_NAN_TO_NUM_FUNC(float16);
DEFINE_NAN_TO_NUM_FUNC(bfloat16);
DEFINE_NAN_TO_NUM_FUNC(float);
DEFINE_NAN_TO_NUM_FUNC(double);
#undef DEFINE_NAN_TO_NUM_FUNC

#define DEFINE_BIAS_FUNC(T)                                                   \
  template <>                                                                 \
  DRAGON_API void Bias<T, CPUContext>(                                        \
      const int N, const float beta, const T* x, T* y, CPUContext* ctx) {     \
    if (beta == 0.f) return;                                                  \
    using EigenT = math::Traits<T>::eigen_type;                               \
    EigenVectorArrayMap<EigenT>((EigenT*)y, N) =                              \
        ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N) + EigenT(beta); \
  }

DEFINE_BIAS_FUNC(uint8_t);
DEFINE_BIAS_FUNC(int8_t);
DEFINE_BIAS_FUNC(int);
DEFINE_BIAS_FUNC(int64_t);
DEFINE_BIAS_FUNC(float16);
DEFINE_BIAS_FUNC(bfloat16);
DEFINE_BIAS_FUNC(float);
DEFINE_BIAS_FUNC(double);
#undef DEFINE_BIAS_FUNC

#define DEFINE_APPLY_MASK_FUNC(T)                                             \
  template <>                                                                 \
  DRAGON_API void ApplyMask<T, CPUContext>(                                   \
      const int N,                                                            \
      const float alpha,                                                      \
      const uint8_t* mask,                                                    \
      const T* x,                                                             \
      T* y,                                                                   \
      CPUContext* ctx) {                                                      \
    using AccT = math::Traits<T>::accumulator_type;                           \
    const AccT scale = AccT(alpha);                                           \
    for (int i = 0; i < N; ++i) {                                             \
      y[i] = convert::To<T>(convert::To<AccT>(x[i]) * AccT(mask[i]) * scale); \
    }                                                                         \
  }

DEFINE_APPLY_MASK_FUNC(uint8_t);
DEFINE_APPLY_MASK_FUNC(int8_t);
DEFINE_APPLY_MASK_FUNC(int);
DEFINE_APPLY_MASK_FUNC(int64_t);
DEFINE_APPLY_MASK_FUNC(float16);
DEFINE_APPLY_MASK_FUNC(bfloat16);
DEFINE_APPLY_MASK_FUNC(float);
DEFINE_APPLY_MASK_FUNC(double);
#undef DEFINE_APPLY_MASK_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Expr)                 \
  template <>                                                           \
  DRAGON_API void name<InputT, CPUContext>(                             \
      const int N,                                                      \
      const InputT* a,                                                  \
      const InputT* b,                                                  \
      OutputT* y,                                                       \
      CPUContext* ctx) {                                                \
    using EigenInputT = math::Traits<InputT>::eigen_type;               \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;             \
    EigenVectorArrayMap<EigenOutputT>((EigenOutputT*)y, N) =            \
        ConstEigenVectorArrayMap<EigenInputT>((const EigenInputT*)a, N) \
            Expr ConstEigenVectorArrayMap<EigenInputT>(                 \
                (const EigenInputT*)b, N);                              \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, +);
DEFINE_BINARY_FUNC(Add, int8_t, int8_t, +);
DEFINE_BINARY_FUNC(Add, int, int, +);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, +);
DEFINE_BINARY_FUNC(Add, float16, float16, +);
DEFINE_BINARY_FUNC(Add, bfloat16, bfloat16, +);
DEFINE_BINARY_FUNC(Add, float, float, +);
DEFINE_BINARY_FUNC(Add, double, double, +);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, -);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, -);
DEFINE_BINARY_FUNC(Sub, int, int, -);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, -);
DEFINE_BINARY_FUNC(Sub, float16, float16, -);
DEFINE_BINARY_FUNC(Sub, bfloat16, bfloat16, -);
DEFINE_BINARY_FUNC(Sub, float, float, -);
DEFINE_BINARY_FUNC(Sub, double, double, -);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, *);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, *);
DEFINE_BINARY_FUNC(Mul, int, int, *);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, *);
DEFINE_BINARY_FUNC(Mul, float16, float16, *);
DEFINE_BINARY_FUNC(Mul, bfloat16, bfloat16, *);
DEFINE_BINARY_FUNC(Mul, float, float, *);
DEFINE_BINARY_FUNC(Mul, double, double, *);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, /);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, /);
DEFINE_BINARY_FUNC(Div, int, int, /);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, /);
DEFINE_BINARY_FUNC(Div, float16, float16, /);
DEFINE_BINARY_FUNC(Div, bfloat16, bfloat16, /);
DEFINE_BINARY_FUNC(Div, float, float, /);
DEFINE_BINARY_FUNC(Div, double, double, /);
DEFINE_BINARY_FUNC(Equal, bool, bool, ==);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, int, bool, ==);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, float16, bool, ==);
DEFINE_BINARY_FUNC(Equal, bfloat16, bool, ==);
DEFINE_BINARY_FUNC(Equal, float, bool, ==);
DEFINE_BINARY_FUNC(Equal, double, bool, ==);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, float16, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, bfloat16, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, float, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, double, bool, !=);
DEFINE_BINARY_FUNC(Less, bool, bool, <);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, <);
DEFINE_BINARY_FUNC(Less, int8_t, bool, <);
DEFINE_BINARY_FUNC(Less, int, bool, <);
DEFINE_BINARY_FUNC(Less, int64_t, bool, <);
DEFINE_BINARY_FUNC(Less, float16, bool, <);
DEFINE_BINARY_FUNC(Less, bfloat16, bool, <);
DEFINE_BINARY_FUNC(Less, float, bool, <);
DEFINE_BINARY_FUNC(Less, double, bool, <);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, float16, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, bfloat16, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, float, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, double, bool, <=);
DEFINE_BINARY_FUNC(Greater, bool, bool, >);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, int, bool, >);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, >);
DEFINE_BINARY_FUNC(Greater, float16, bool, >);
DEFINE_BINARY_FUNC(Greater, bfloat16, bool, >);
DEFINE_BINARY_FUNC(Greater, float, bool, >);
DEFINE_BINARY_FUNC(Greater, double, bool, >);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, bfloat16, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, >=);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Func)                 \
  template <>                                                           \
  DRAGON_API void name<InputT, CPUContext>(                             \
      const int N,                                                      \
      const InputT* a,                                                  \
      const InputT* b,                                                  \
      OutputT* y,                                                       \
      CPUContext* ctx) {                                                \
    using EigenInputT = math::Traits<InputT>::eigen_type;               \
    using EigenOutputT = math::Traits<OutputT>::eigen_type;             \
    EigenVectorArrayMap<EigenOutputT>((EigenOutputT*)y, N) =            \
        ConstEigenVectorArrayMap<EigenInputT>((const EigenInputT*)a, N) \
            .Func(ConstEigenVectorArrayMap<EigenInputT>(                \
                (const EigenInputT*)b, N));                             \
  }

DEFINE_BINARY_FUNC(Pow, float16, float16, pow);
DEFINE_BINARY_FUNC(Pow, bfloat16, bfloat16, pow);
DEFINE_BINARY_FUNC(Pow, float, float, pow);
DEFINE_BINARY_FUNC(Pow, double, double, pow);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, min);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, min);
DEFINE_BINARY_FUNC(Minimum, int, int, min);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, min);
DEFINE_BINARY_FUNC(Minimum, float16, float16, min);
DEFINE_BINARY_FUNC(Minimum, bfloat16, bfloat16, min);
DEFINE_BINARY_FUNC(Minimum, float, float, min);
DEFINE_BINARY_FUNC(Minimum, double, double, min);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, max);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, max);
DEFINE_BINARY_FUNC(Maximum, int, int, max);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, max);
DEFINE_BINARY_FUNC(Maximum, float16, float16, max);
DEFINE_BINARY_FUNC(Maximum, bfloat16, bfloat16, max);
DEFINE_BINARY_FUNC(Maximum, float, float, max);
DEFINE_BINARY_FUNC(Maximum, double, double, max);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Functor) \
  template <>                                              \
  DRAGON_API void name<InputT, CPUContext>(                \
      const int N,                                         \
      const InputT* a,                                     \
      const InputT* b,                                     \
      OutputT* y,                                          \
      CPUContext* ctx) {                                   \
    _SimpleBinaryFunc(N, Functor<InputT>(), a, b, y);      \
  }

DEFINE_BINARY_FUNC(Atan2, float16, float16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, bfloat16, bfloat16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, float, float, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, double, double, math::Atan2Functor);
DEFINE_BINARY_FUNC(BitwiseAnd, bool, bool, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, uint8_t, uint8_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int8_t, int8_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int, int, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseAnd, int64_t, int64_t, math::BitAndFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, bool, bool, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, uint8_t, uint8_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int8_t, int8_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int, int, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseOr, int64_t, int64_t, math::BitOrFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, bool, bool, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, uint8_t, uint8_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int8_t, int8_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int, int, math::BitXorFunctor);
DEFINE_BINARY_FUNC(BitwiseXor, int64_t, int64_t, math::BitXorFunctor);
DEFINE_BINARY_FUNC(And, bool, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, uint8_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int8_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, int64_t, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, float16, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, bfloat16, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, float, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, double, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(Or, bool, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, uint8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int64_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float16, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, bfloat16, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, double, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Xor, bool, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int64_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float16, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, bfloat16, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, double, bool, math::XorFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T)                                                   \
  template <>                                                                  \
  DRAGON_API void Where<T, CPUContext>(                                        \
      const int N,                                                             \
      const T* a,                                                              \
      const T* b,                                                              \
      const bool* c,                                                           \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    ConstEigenVectorArrayMap<bool> C(c, N);                                    \
    EigenVectorArrayMap<T>(y, N) = (C).select(                                 \
        ConstEigenVectorArrayMap<T>(a, N), ConstEigenVectorArrayMap<T>(b, N)); \
  }

DEFINE_WHERE_FUNC(bool);
DEFINE_WHERE_FUNC(uint8_t);
DEFINE_WHERE_FUNC(int8_t);
DEFINE_WHERE_FUNC(int);
DEFINE_WHERE_FUNC(int64_t);
DEFINE_WHERE_FUNC(float16);
DEFINE_WHERE_FUNC(bfloat16);
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
