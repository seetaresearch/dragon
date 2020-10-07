#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace math {

#define DEFINE_UNARY_FUNC(name, T, expr)                                     \
  template <>                                                                \
  DRAGON_API void name<T, CPUContext>(                                       \
      const int n, const T* x, T* y, CPUContext* ctx) {                      \
    EigenVectorArrayMap<T>(y, n) = ConstEigenVectorArrayMap<T>(x, n).expr(); \
  }

DEFINE_UNARY_FUNC(Abs, int8_t, abs);
DEFINE_UNARY_FUNC(Abs, uint8_t, abs);
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
DEFINE_UNARY_FUNC(Square, int8_t, square);
DEFINE_UNARY_FUNC(Square, uint8_t, square);
DEFINE_UNARY_FUNC(Square, int, square);
DEFINE_UNARY_FUNC(Square, int64_t, square);
DEFINE_UNARY_FUNC(Square, float, square);
DEFINE_UNARY_FUNC(Square, double, square);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name)                                     \
  template <>                                                       \
  DRAGON_API void name<float16, CPUContext>(                        \
      const int n, const float16* x, float16* y, CPUContext* ctx) { \
    CPU_FP16_NOT_SUPPORTED;                                         \
  }

DEFINE_UNARY_FUNC(Abs);
DEFINE_UNARY_FUNC(Ceil);
DEFINE_UNARY_FUNC(Cos);
DEFINE_UNARY_FUNC(Exp);
DEFINE_UNARY_FUNC(Floor);
DEFINE_UNARY_FUNC(Inv);
DEFINE_UNARY_FUNC(Log);
DEFINE_UNARY_FUNC(Round);
DEFINE_UNARY_FUNC(Rsqrt);
DEFINE_UNARY_FUNC(Sin);
DEFINE_UNARY_FUNC(Sign);
DEFINE_UNARY_FUNC(Sqrt);
DEFINE_UNARY_FUNC(Square);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, T, expr)                   \
  template <>                                              \
  DRAGON_API void name<T, CPUContext>(                     \
      const int n, const T* x, T* y, CPUContext* ctx) {    \
    EigenVectorArrayMap<T>(y, n) =                         \
        ConstEigenVectorArrayMap<T>(x, n).unaryExpr(expr); \
  }

DEFINE_UNARY_FUNC(Invert, bool, [](bool x) { return !x; });
DEFINE_UNARY_FUNC(Invert, int8_t, [](int8_t x) { return (int8_t)~x; });
DEFINE_UNARY_FUNC(Invert, uint8_t, [](uint8_t x) { return (uint8_t)~x; });
DEFINE_UNARY_FUNC(Invert, int, [](int x) { return (int)~x; });
DEFINE_UNARY_FUNC(Invert, int64_t, [](int64_t x) { return (int64_t)~x; });
DEFINE_UNARY_FUNC(Sign, int8_t, [](int8_t x) {
  return (int8_t)((x > int8_t(0)) - (x < int8_t(0)));
});
DEFINE_UNARY_FUNC(Sign, uint8_t, [](uint8_t x) {
  return (uint8_t)(x > uint8_t(0));
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

template <>
#define DEFINE_NEG_FUNC(T)                                             \
  template <>                                                          \
  DRAGON_API void Neg<T, CPUContext>(                                  \
      const int n, const T* x, T* y, CPUContext* ctx) {                \
    EigenVectorArrayMap<T>(y, n) = -ConstEigenVectorArrayMap<T>(x, n); \
  }

DRAGON_API void Neg<float16, CPUContext>(
    const int n,
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

/* y = value */

#define DEFINE_SET_FUNC(T)                                 \
  template <>                                              \
  DRAGON_API void Set<T, CPUContext>(                      \
      const int n, const T value, T* y, CPUContext* ctx) { \
    if (n == 0) return;                                    \
    if (value == T(0)) {                                   \
      memset(y, 0, sizeof(T) * n);                         \
    } else {                                               \
      EigenVectorMap<T>(y, n).setConstant(value);          \
    }                                                      \
  }

template <>
DRAGON_API void Set<float16, CPUContext>(
    const int n,
    const float16 alpha,
    float16* y,
    CPUContext* ctx) {
  if (alpha.x == (unsigned short)0) {
    memset(y, 0, sizeof(float16) * n);
  } else {
    EigenVectorMap<float16>(y, n).setConstant(alpha);
  }
}

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

/* y = 1 / sqrt(x + eps) */

#define DEFINE_INVSTD_FUNC(T)                                            \
  template <>                                                            \
  DRAGON_API void InvStd<T, CPUContext>(                                 \
      const int n, const float eps, const T* x, T* y, CPUContext* ctx) { \
    EigenVectorArrayMap<T>(y, n) =                                       \
        (ConstEigenVectorArrayMap<T>(x, n) + (T)eps).rsqrt();            \
  }

template <>
DRAGON_API void InvStd<float16, CPUContext>(
    const int n,
    const float eps,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

/* y = x^e */

#define DEFINE_POWX_FUNC(T)                                                   \
  template <>                                                                 \
  DRAGON_API void Powx<T, CPUContext>(                                        \
      const int n, const float exponent, const T* x, T* y, CPUContext* ctx) { \
    EigenVectorArrayMap<T>(y, n) =                                            \
        ConstEigenVectorArrayMap<T>(x, n).pow((T)exponent);                   \
  }

template <>
DRAGON_API void Powx<float16, CPUContext>(
    int n,
    const float alpha,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

/* y = notzero(x) */

#define DEFINE_NOT_ZERO_FUNC(T)                                \
  template <>                                                  \
  DRAGON_API void NotZero<T, CPUContext>(                      \
      const int count, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, count) =                      \
        ConstEigenVectorArrayMap<T>(x, count) != T(0);         \
  }

template <>
DRAGON_API void NotZero<float16, CPUContext>(
    const int count,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_NOT_ZERO_FUNC(bool);
DEFINE_NOT_ZERO_FUNC(int8_t);
DEFINE_NOT_ZERO_FUNC(uint8_t);
DEFINE_NOT_ZERO_FUNC(int);
DEFINE_NOT_ZERO_FUNC(int64_t);
DEFINE_NOT_ZERO_FUNC(float);
DEFINE_NOT_ZERO_FUNC(double);
#undef DEFINE_NOT_ZERO_FUNC

/* y = isinf(x) */

#define DEFINE_IS_INF_FUNC(T)                              \
  template <>                                              \
  DRAGON_API void IsInf<T, CPUContext>(                    \
      const int n, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, n) =                      \
        ConstEigenVectorArrayMap<T>(x, n).isInf();         \
  }

template <>
DRAGON_API void IsInf<float16, CPUContext>(
    const int n,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  for (int i = 0; i < n; ++i) {
    y[i] = utils::math::IsInf(x[i]);
  }
}

DEFINE_IS_INF_FUNC(float);
DEFINE_IS_INF_FUNC(double);
#undef DEFINE_IS_INF_FUNC

/*    y = isnan(x)         */

#define DEFINE_IS_NAN_FUNC(T)                              \
  template <>                                              \
  DRAGON_API void IsNaN<T, CPUContext>(                    \
      const int n, const T* x, bool* y, CPUContext* ctx) { \
    EigenVectorArrayMap<bool>(y, n) =                      \
        ConstEigenVectorArrayMap<T>(x, n).isNaN();         \
  }

template <>
DRAGON_API void IsNaN<float16, CPUContext>(
    const int n,
    const float16* x,
    bool* y,
    CPUContext* ctx) {
  for (int i = 0; i < n; ++i) {
    y[i] = utils::math::IsNaN(x[i]);
  }
}

DEFINE_IS_NAN_FUNC(float);
DEFINE_IS_NAN_FUNC(double);
#undef DEFINE_IS_NAN_FUNC

/*    y = isnan(x) ? value : x         */

#define DEFINE_REPLACE_NAN_FUNC(T)                                     \
  template <>                                                          \
  DRAGON_API void ReplaceNaN<T, CPUContext>(                           \
      const int n, const T value, const T* x, T* y, CPUContext* ctx) { \
    ConstEigenVectorArrayMap<T> X(x, n);                               \
    EigenVectorArrayMap<T>(y, n) = (X.isNaN()).select(value, X);       \
  }

template <>
DRAGON_API void ReplaceNaN<float16, CPUContext>(
    const int n,
    const float16 value,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  EigenVectorArrayMap<float16>(y, n) =
      ConstEigenVectorArrayMap<float16>(x, n).unaryExpr(
          [&](float16 x) { return utils::math::IsNaN(x) ? value : x; });
}

DEFINE_REPLACE_NAN_FUNC(float);
DEFINE_REPLACE_NAN_FUNC(double);
#undef DEFINE_REPLACE_NAN_FUNC

/*     y = x + beta    */

#define DEFINE_BIAS_FUNC(T)                                               \
  template <>                                                             \
  DRAGON_API void Bias<T, CPUContext>(                                    \
      const int n, const float beta, const T* x, T* y, CPUContext* ctx) { \
    if (beta == 0.f) return;                                              \
    EigenVectorArrayMap<T>(y, n) =                                        \
        ConstEigenVectorArrayMap<T>(x, n) + T(beta);                      \
  }

template <>
DRAGON_API void Bias<float16, CPUContext>(
    const int n,
    const float beta,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_BIAS_FUNC(int8_t);
DEFINE_BIAS_FUNC(uint8_t);
DEFINE_BIAS_FUNC(int);
DEFINE_BIAS_FUNC(int64_t);
DEFINE_BIAS_FUNC(float);
DEFINE_BIAS_FUNC(double);
#undef DEFINE_BIAS_FUNC

#define DEFINE_BINARY_FUNC(name, TIn, TOut, expr)                          \
  template <>                                                              \
  DRAGON_API void name<TIn, CPUContext>(                                   \
      const int n, const TIn* a, const TIn* b, TOut* y, CPUContext* ctx) { \
    EigenVectorArrayMap<TOut>(y, n) = ConstEigenVectorArrayMap<TIn>(a, n)  \
        expr ConstEigenVectorArrayMap<TIn>(b, n);                          \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t, +);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, +);
DEFINE_BINARY_FUNC(Add, int, int, +);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, +);
DEFINE_BINARY_FUNC(Add, float, float, +);
DEFINE_BINARY_FUNC(Add, double, double, +);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, -);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, -);
DEFINE_BINARY_FUNC(Sub, int, int, -);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, -);
DEFINE_BINARY_FUNC(Sub, float, float, -);
DEFINE_BINARY_FUNC(Sub, double, double, -);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, *);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, *);
DEFINE_BINARY_FUNC(Mul, int, int, *);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, *);
DEFINE_BINARY_FUNC(Mul, float, float, *);
DEFINE_BINARY_FUNC(Mul, double, double, *);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, /);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, /);
DEFINE_BINARY_FUNC(Div, int, int, /);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, /);
DEFINE_BINARY_FUNC(Div, float, float, /);
DEFINE_BINARY_FUNC(Div, double, double, /);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, int, bool, ==);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, ==);
DEFINE_BINARY_FUNC(Equal, float, bool, ==);
DEFINE_BINARY_FUNC(Equal, double, bool, ==);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, float, bool, !=);
DEFINE_BINARY_FUNC(NotEqual, double, bool, !=);
DEFINE_BINARY_FUNC(Less, int8_t, bool, <);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, <);
DEFINE_BINARY_FUNC(Less, int, bool, <);
DEFINE_BINARY_FUNC(Less, int64_t, bool, <);
DEFINE_BINARY_FUNC(Less, float, bool, <);
DEFINE_BINARY_FUNC(Less, double, bool, <);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, float, bool, <=);
DEFINE_BINARY_FUNC(LessEqual, double, bool, <=);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, >);
DEFINE_BINARY_FUNC(Greater, int, bool, >);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, >);
DEFINE_BINARY_FUNC(Greater, float, bool, >);
DEFINE_BINARY_FUNC(Greater, double, bool, >);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, >=);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, >=);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, TIn, TOut, func)                          \
  template <>                                                              \
  DRAGON_API void name<TIn, CPUContext>(                                   \
      const int n, const TIn* a, const TIn* b, TOut* y, CPUContext* ctx) { \
    EigenVectorArrayMap<TOut>(y, n) =                                      \
        ConstEigenVectorArrayMap<TIn>(a, n).func(                          \
            ConstEigenVectorArrayMap<TIn>(b, n));                          \
  }

DEFINE_BINARY_FUNC(Pow, float, float, pow);
DEFINE_BINARY_FUNC(Pow, double, double, pow);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, min);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, min);
DEFINE_BINARY_FUNC(Minimum, int, int, min);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, min);
DEFINE_BINARY_FUNC(Minimum, float, float, min);
DEFINE_BINARY_FUNC(Minimum, double, double, min);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, max);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, max);
DEFINE_BINARY_FUNC(Maximum, int, int, max);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, max);
DEFINE_BINARY_FUNC(Maximum, float, float, max);
DEFINE_BINARY_FUNC(Maximum, double, double, max);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, T, dtype)                          \
  template <>                                                       \
  DRAGON_API void name<T, CPUContext>(                              \
      const int n, const T* a, const T* b, T* y, CPUContext* ctx) { \
    name(                                                           \
        n,                                                          \
        reinterpret_cast<const dtype*>(a),                          \
        reinterpret_cast<const dtype*>(b),                          \
        reinterpret_cast<dtype*>(y),                                \
        ctx);                                                       \
  }

DEFINE_BINARY_FUNC(Add, bool, uint8_t); // Or
DEFINE_BINARY_FUNC(Sub, bool, uint8_t); // Xor
DEFINE_BINARY_FUNC(Mul, bool, uint8_t); // And
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, TOut)       \
  template <>                                \
  DRAGON_API void name<float16, CPUContext>( \
      const int n,                           \
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

#define DEFINE_WHERE_FUNC(T)                                                   \
  template <>                                                                  \
  DRAGON_API void Where<T, CPUContext>(                                        \
      const int n,                                                             \
      const T* a,                                                              \
      const T* b,                                                              \
      const bool* c,                                                           \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    ConstEigenVectorArrayMap<bool> C(c, n);                                    \
    EigenVectorArrayMap<T>(y, n) = (C).select(                                 \
        ConstEigenVectorArrayMap<T>(a, n), ConstEigenVectorArrayMap<T>(b, n)); \
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
