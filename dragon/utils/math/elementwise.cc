#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"

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

#define DEFINE_UNARY_FUNC(name, T, Expr)                                     \
  template <>                                                                \
  DRAGON_API void name<T, CPUContext>(                                       \
      const int N, const T* x, T* y, CPUContext* ctx) {                      \
    EigenVectorArrayMap<T>(y, N) = ConstEigenVectorArrayMap<T>(x, N).Expr(); \
  }

DEFINE_UNARY_FUNC(Abs, uint8_t, abs);
DEFINE_UNARY_FUNC(Abs, int8_t, abs);
DEFINE_UNARY_FUNC(Abs, int, abs);
DEFINE_UNARY_FUNC(Abs, int64_t, abs);
DEFINE_UNARY_FUNC(Abs, float, abs);
DEFINE_UNARY_FUNC(Abs, double, abs);
DEFINE_UNARY_FUNC(Ceil, float, ceil);
DEFINE_UNARY_FUNC(Ceil, double, ceil);
DEFINE_UNARY_FUNC(Cos, float, cos);
DEFINE_UNARY_FUNC(Cos, double, cos);
DEFINE_UNARY_FUNC(Exp, float, exp);
DEFINE_UNARY_FUNC(Exp, double, exp);
DEFINE_UNARY_FUNC(Floor, float, floor);
DEFINE_UNARY_FUNC(Floor, double, floor);
DEFINE_UNARY_FUNC(Inv, float, inverse);
DEFINE_UNARY_FUNC(Inv, double, inverse);
DEFINE_UNARY_FUNC(Log, float, log);
DEFINE_UNARY_FUNC(Log, double, log);
DEFINE_UNARY_FUNC(Round, float, round);
DEFINE_UNARY_FUNC(Round, double, round);
DEFINE_UNARY_FUNC(Rsqrt, float, rsqrt);
DEFINE_UNARY_FUNC(Rsqrt, double, rsqrt);
DEFINE_UNARY_FUNC(Sin, float, sin);
DEFINE_UNARY_FUNC(Sin, double, sin);
DEFINE_UNARY_FUNC(Sqrt, float, sqrt);
DEFINE_UNARY_FUNC(Sqrt, double, sqrt);
DEFINE_UNARY_FUNC(Square, uint8_t, square);
DEFINE_UNARY_FUNC(Square, int8_t, square);
DEFINE_UNARY_FUNC(Square, int, square);
DEFINE_UNARY_FUNC(Square, int64_t, square);
DEFINE_UNARY_FUNC(Square, float, square);
DEFINE_UNARY_FUNC(Square, double, square);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, OutputT)                            \
  template <>                                                       \
  DRAGON_API void name<float16, CPUContext>(                        \
      const int N, const float16* x, OutputT* y, CPUContext* ctx) { \
    CPU_FP16_NOT_SUPPORTED;                                         \
  }

DEFINE_UNARY_FUNC(Abs, float16);
DEFINE_UNARY_FUNC(Ceil, float16);
DEFINE_UNARY_FUNC(Cos, float16);
DEFINE_UNARY_FUNC(Exp, float16);
DEFINE_UNARY_FUNC(Floor, float16);
DEFINE_UNARY_FUNC(Inv, float16);
DEFINE_UNARY_FUNC(Log, float16);
DEFINE_UNARY_FUNC(Round, float16);
DEFINE_UNARY_FUNC(Rsqrt, float16);
DEFINE_UNARY_FUNC(Sin, float16);
DEFINE_UNARY_FUNC(Sign, float16);
DEFINE_UNARY_FUNC(Sqrt, float16);
DEFINE_UNARY_FUNC(Square, float16);
DEFINE_UNARY_FUNC(Not, bool);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, T, expr)                   \
  template <>                                              \
  DRAGON_API void name<T, CPUContext>(                     \
      const int N, const T* x, T* y, CPUContext* ctx) {    \
    EigenVectorArrayMap<T>(y, N) =                         \
        ConstEigenVectorArrayMap<T>(x, N).unaryExpr(expr); \
  }

DEFINE_UNARY_FUNC(Sign, uint8_t, [](uint8_t x) {
  return (uint8_t)(x > uint8_t(0));
});
DEFINE_UNARY_FUNC(Sign, int8_t, [](int8_t x) {
  return (int8_t)((x > int8_t(0)) - (x < int8_t(0)));
});
DEFINE_UNARY_FUNC(Sign, int, [](int x) {
  return (int)((x > int(0)) - (x < int(0)));
});
DEFINE_UNARY_FUNC(Sign, int64_t, [](int64_t x) {
  return (int64_t)((x > int64_t(0)) - (x < int64_t(0)));
});
DEFINE_UNARY_FUNC(Sign, float, [](float x) {
  return (float)((x > 0.f) - (x < 0.f));
});
DEFINE_UNARY_FUNC(Sign, double, [](double x) {
  return (double)((x > 0.) - (x < 0.));
});
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, InputT, OutputT, Functor)          \
  template <>                                                      \
  DRAGON_API void name<InputT, CPUContext>(                        \
      const int N, const InputT* x, OutputT* y, CPUContext* ctx) { \
    _SimpleUnaryFunc(N, Functor<InputT>(), x, y);                  \
  }

DEFINE_UNARY_FUNC(BitwiseNot, bool, bool, std::bit_not);
DEFINE_UNARY_FUNC(BitwiseNot, uint8_t, uint8_t, std::bit_not);
DEFINE_UNARY_FUNC(BitwiseNot, int8_t, int8_t, std::bit_not);
DEFINE_UNARY_FUNC(BitwiseNot, int, int, std::bit_not);
DEFINE_UNARY_FUNC(BitwiseNot, int64_t, int64_t, std::bit_not);
DEFINE_UNARY_FUNC(Not, bool, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, uint8_t, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, int8_t, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, int, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, int64_t, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, float, bool, std::logical_not);
DEFINE_UNARY_FUNC(Not, double, bool, std::logical_not);
#undef DEFINE_UNARY_FUNC

template <>
#define DEFINE_NEG_FUNC(T)                                             \
  template <>                                                          \
  DRAGON_API void Neg<T, CPUContext>(                                  \
      const int N, const T* x, T* y, CPUContext* ctx) {                \
    EigenVectorArrayMap<T>(y, N) = -ConstEigenVectorArrayMap<T>(x, N); \
  }

DRAGON_API void Neg<float16, CPUContext>(
    const int N,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_NEG_FUNC(int8_t);
DEFINE_NEG_FUNC(int);
DEFINE_NEG_FUNC(int64_t);
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

template <>
DRAGON_API void Set<float16, CPUContext>(
    const int N,
    const float16 alpha,
    float16* y,
    CPUContext* ctx) {
  if (alpha.x == (unsigned short)0) {
    memset(y, 0, sizeof(float16) * N);
  } else {
    EigenVectorMap<float16>(y, N).setConstant(alpha);
  }
}

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_INVSTD_FUNC(T)                                            \
  template <>                                                            \
  DRAGON_API void InvStd<T, CPUContext>(                                 \
      const int N, const float eps, const T* x, T* y, CPUContext* ctx) { \
    EigenVectorArrayMap<T>(y, N) =                                       \
        (ConstEigenVectorArrayMap<T>(x, N) + (T)eps).rsqrt();            \
  }

template <>
DRAGON_API void InvStd<float16, CPUContext>(
    const int N,
    const float eps,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

#define DEFINE_POWX_FUNC(T)                                                   \
  template <>                                                                 \
  DRAGON_API void Powx<T, CPUContext>(                                        \
      const int N, const float exponent, const T* x, T* y, CPUContext* ctx) { \
    EigenVectorArrayMap<T>(y, N) =                                            \
        ConstEigenVectorArrayMap<T>(x, N).pow((T)exponent);                   \
  }

template <>
DRAGON_API void Powx<float16, CPUContext>(
    int N,
    const float alpha,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

#define DEFINE_NOT_ZERO_FUNC(T)                            \
  template <>                                              \
  DRAGON_API void NotZero<T, CPUContext>(                  \
      const int N, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, N) =                      \
        ConstEigenVectorArrayMap<T>(x, N) != T(0);         \
  }

template <>
DRAGON_API void NotZero<float16, CPUContext>(
    const int N,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_NOT_ZERO_FUNC(bool);
DEFINE_NOT_ZERO_FUNC(uint8_t);
DEFINE_NOT_ZERO_FUNC(int8_t);
DEFINE_NOT_ZERO_FUNC(int);
DEFINE_NOT_ZERO_FUNC(int64_t);
DEFINE_NOT_ZERO_FUNC(float);
DEFINE_NOT_ZERO_FUNC(double);
#undef DEFINE_NOT_ZERO_FUNC

#define DEFINE_IS_INF_FUNC(T)                              \
  template <>                                              \
  DRAGON_API void IsInf<T, CPUContext>(                    \
      const int N, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, N) =                      \
        ConstEigenVectorArrayMap<T>(x, N).isInf();         \
  }

template <>
DRAGON_API void IsInf<float16, CPUContext>(
    const int N,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    y[i] = math::utils::IsInf(x[i]);
  }
}

DEFINE_IS_INF_FUNC(float);
DEFINE_IS_INF_FUNC(double);
#undef DEFINE_IS_INF_FUNC

#define DEFINE_IS_NAN_FUNC(T)                              \
  template <>                                              \
  DRAGON_API void IsNaN<T, CPUContext>(                    \
      const int N, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, N) =                      \
        ConstEigenVectorArrayMap<T>(x, N).isNaN();         \
  }

template <>
DRAGON_API void IsNaN<float16, CPUContext>(
    const int N,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    y[i] = math::utils::IsNaN(x[i]);
  }
}

DEFINE_IS_NAN_FUNC(float);
DEFINE_IS_NAN_FUNC(double);
#undef DEFINE_IS_NAN_FUNC

#define DEFINE_REPLACE_NAN_FUNC(T)                                     \
  template <>                                                          \
  DRAGON_API void ReplaceNaN<T, CPUContext>(                           \
      const int N, const T value, const T* x, T* y, CPUContext* ctx) { \
    ConstEigenVectorArrayMap<T> X(x, N);                               \
    EigenVectorArrayMap<T>(y, N) = (X.isNaN()).select(value, X);       \
  }

template <>
DRAGON_API void ReplaceNaN<float16, CPUContext>(
    const int N,
    const float16 value,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  EigenVectorArrayMap<float16>(y, N) =
      ConstEigenVectorArrayMap<float16>(x, N).unaryExpr(
          [&](float16 x) { return math::utils::IsNaN(x) ? value : x; });
}

DEFINE_REPLACE_NAN_FUNC(float);
DEFINE_REPLACE_NAN_FUNC(double);
#undef DEFINE_REPLACE_NAN_FUNC

#define DEFINE_BIAS_FUNC(T)                                               \
  template <>                                                             \
  DRAGON_API void Bias<T, CPUContext>(                                    \
      const int N, const float beta, const T* x, T* y, CPUContext* ctx) { \
    if (beta == 0.f) return;                                              \
    EigenVectorArrayMap<T>(y, N) =                                        \
        ConstEigenVectorArrayMap<T>(x, N) + T(beta);                      \
  }

template <>
DRAGON_API void Bias<float16, CPUContext>(
    const int N,
    const float beta,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_BIAS_FUNC(uint8_t);
DEFINE_BIAS_FUNC(int8_t);
DEFINE_BIAS_FUNC(int);
DEFINE_BIAS_FUNC(int64_t);
DEFINE_BIAS_FUNC(float);
DEFINE_BIAS_FUNC(double);
#undef DEFINE_BIAS_FUNC

#define DEFINE_APPLY_MASK_FUNC(T)           \
  template <>                               \
  DRAGON_API void ApplyMask<T, CPUContext>( \
      const int N,                          \
      const float alpha,                    \
      const uint8_t* mask,                  \
      const T* x,                           \
      T* y,                                 \
      CPUContext* ctx) {                    \
    const T scale = T(alpha);               \
    for (int i = 0; i < N; ++i) {           \
      y[i] = x[i] * T(mask[i]) * scale;     \
    }                                       \
  }

template <>
DRAGON_API void ApplyMask<float16, CPUContext>(
    const int N,
    const float alpha,
    const uint8_t* mask,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    y[i] =
        convert::To<float16>(convert::To<float>(x[i]) * float(mask[i]) * alpha);
  }
}

DEFINE_APPLY_MASK_FUNC(uint8_t);
DEFINE_APPLY_MASK_FUNC(int8_t);
DEFINE_APPLY_MASK_FUNC(int);
DEFINE_APPLY_MASK_FUNC(int64_t);
DEFINE_APPLY_MASK_FUNC(float);
DEFINE_APPLY_MASK_FUNC(double);
#undef DEFINE_APPLY_MASK_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Expr)                    \
  template <>                                                              \
  DRAGON_API void name<InputT, CPUContext>(                                \
      const int N,                                                         \
      const InputT* a,                                                     \
      const InputT* b,                                                     \
      OutputT* y,                                                          \
      CPUContext* ctx) {                                                   \
    EigenVectorArrayMap<OutputT>(y, N) = ConstEigenVectorArrayMap<InputT>( \
        a, N) Expr ConstEigenVectorArrayMap<InputT>(b, N);                 \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, +);
DEFINE_BINARY_FUNC(Add, int8_t, int8_t, +);
DEFINE_BINARY_FUNC(Add, int, int, +);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, +);
DEFINE_BINARY_FUNC(Add, float, float, +);
DEFINE_BINARY_FUNC(Add, double, double, +);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, -);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, -);
DEFINE_BINARY_FUNC(Sub, int, int, -);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, -);
DEFINE_BINARY_FUNC(Sub, float, float, -);
DEFINE_BINARY_FUNC(Sub, double, double, -);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, *);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, *);
DEFINE_BINARY_FUNC(Mul, int, int, *);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, *);
DEFINE_BINARY_FUNC(Mul, float, float, *);
DEFINE_BINARY_FUNC(Mul, double, double, *);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, /);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, /);
DEFINE_BINARY_FUNC(Div, int, int, /);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, /);
DEFINE_BINARY_FUNC(Div, float, float, /);
DEFINE_BINARY_FUNC(Div, double, double, /);
DEFINE_BINARY_FUNC(Equal, bool, bool, ==);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, int, bool, ==);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, float, bool, ==);
DEFINE_BINARY_FUNC(Equal, double, bool, ==);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, float, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, double, bool, !=);
DEFINE_BINARY_FUNC(Less, bool, bool, <);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, <);
DEFINE_BINARY_FUNC(Less, int8_t, bool, <);
DEFINE_BINARY_FUNC(Less, int, bool, <);
DEFINE_BINARY_FUNC(Less, int64_t, bool, <);
DEFINE_BINARY_FUNC(Less, float, bool, <);
DEFINE_BINARY_FUNC(Less, double, bool, <);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, float, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, double, bool, <=);
DEFINE_BINARY_FUNC(Greater, bool, bool, >);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, int, bool, >);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, >);
DEFINE_BINARY_FUNC(Greater, float, bool, >);
DEFINE_BINARY_FUNC(Greater, double, bool, >);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, >=);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Func) \
  template <>                                           \
  DRAGON_API void name<InputT, CPUContext>(             \
      const int N,                                      \
      const InputT* a,                                  \
      const InputT* b,                                  \
      OutputT* y,                                       \
      CPUContext* ctx) {                                \
    EigenVectorArrayMap<OutputT>(y, N) =                \
        ConstEigenVectorArrayMap<InputT>(a, N).Func(    \
            ConstEigenVectorArrayMap<InputT>(b, N));    \
  }

DEFINE_BINARY_FUNC(Pow, float, float, pow);
DEFINE_BINARY_FUNC(Pow, double, double, pow);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, min);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, min);
DEFINE_BINARY_FUNC(Minimum, int, int, min);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, min);
DEFINE_BINARY_FUNC(Minimum, float, float, min);
DEFINE_BINARY_FUNC(Minimum, double, double, min);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, max);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, max);
DEFINE_BINARY_FUNC(Maximum, int, int, max);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, max);
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

DEFINE_BINARY_FUNC(BitwiseAnd, bool, bool, std::bit_and);
DEFINE_BINARY_FUNC(BitwiseAnd, uint8_t, uint8_t, std::bit_and);
DEFINE_BINARY_FUNC(BitwiseAnd, int8_t, int8_t, std::bit_and);
DEFINE_BINARY_FUNC(BitwiseAnd, int, int, std::bit_and);
DEFINE_BINARY_FUNC(BitwiseAnd, int64_t, int64_t, std::bit_and);
DEFINE_BINARY_FUNC(BitwiseOr, bool, bool, std::bit_or);
DEFINE_BINARY_FUNC(BitwiseOr, uint8_t, uint8_t, std::bit_or);
DEFINE_BINARY_FUNC(BitwiseOr, int8_t, int8_t, std::bit_or);
DEFINE_BINARY_FUNC(BitwiseOr, int, int, std::bit_or);
DEFINE_BINARY_FUNC(BitwiseOr, int64_t, int64_t, std::bit_or);
DEFINE_BINARY_FUNC(BitwiseXor, bool, bool, std::bit_xor);
DEFINE_BINARY_FUNC(BitwiseXor, uint8_t, uint8_t, std::bit_xor);
DEFINE_BINARY_FUNC(BitwiseXor, int8_t, int8_t, std::bit_xor);
DEFINE_BINARY_FUNC(BitwiseXor, int, int, std::bit_xor);
DEFINE_BINARY_FUNC(BitwiseXor, int64_t, int64_t, std::bit_xor);
DEFINE_BINARY_FUNC(And, bool, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, uint8_t, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, int8_t, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, int, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, int64_t, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, float, bool, std::logical_and);
DEFINE_BINARY_FUNC(And, double, bool, std::logical_and);
DEFINE_BINARY_FUNC(Or, bool, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, uint8_t, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, int8_t, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, int, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, int64_t, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, float, bool, std::logical_or);
DEFINE_BINARY_FUNC(Or, double, bool, std::logical_or);
DEFINE_BINARY_FUNC(Xor, bool, bool, std::bit_xor);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int64_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, double, bool, math::XorFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, OutputT)    \
  template <>                                \
  DRAGON_API void name<float16, CPUContext>( \
      const int N,                           \
      const float16* a,                      \
      const float16* b,                      \
      OutputT* y,                            \
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
DEFINE_BINARY_FUNC(And, bool);
DEFINE_BINARY_FUNC(Or, bool);
DEFINE_BINARY_FUNC(Xor, bool);
DEFINE_BINARY_FUNC(Equal, bool);
DEFINE_BINARY_FUNC(NotEqual, bool);
DEFINE_BINARY_FUNC(Less, bool);
DEFINE_BINARY_FUNC(LessEqual, bool);
DEFINE_BINARY_FUNC(Greater, bool);
DEFINE_BINARY_FUNC(GreaterEqual, bool);
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
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
