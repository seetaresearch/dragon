#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

/*!
 * Unary Functors
 */

#define DEFINE_UNARY_FUNCTOR(name, func)               \
  template <typename T>                                \
  struct name##Functor {                               \
    inline __device__ T operator()(const T& x) const { \
      return func(x);                                  \
    }                                                  \
  }

DEFINE_UNARY_FUNCTOR(Neg, -);
DEFINE_UNARY_FUNCTOR(Ceil, ceil);
DEFINE_UNARY_FUNCTOR(Floor, floor);
DEFINE_UNARY_FUNCTOR(Exp, exp);
DEFINE_UNARY_FUNCTOR(Log, log);
DEFINE_UNARY_FUNCTOR(Round, round);
DEFINE_UNARY_FUNCTOR(Sqrt, sqrt);
DEFINE_UNARY_FUNCTOR(Rsqrt, rsqrt);
DEFINE_UNARY_FUNCTOR(Sin, sin);
DEFINE_UNARY_FUNCTOR(Cos, cos);
#if __CUDA_ARCH__ >= 530
DEFINE_UNARY_FUNCTOR(NegHalf, __hneg);
DEFINE_UNARY_FUNCTOR(NegHalf2, __hneg2);
DEFINE_UNARY_FUNCTOR(CeilHalf, hceil);
DEFINE_UNARY_FUNCTOR(CeilHalf2, h2ceil);
DEFINE_UNARY_FUNCTOR(FloorHalf, hfloor);
DEFINE_UNARY_FUNCTOR(FloorHalf2, h2floor);
DEFINE_UNARY_FUNCTOR(RoundHalf, hrint);
DEFINE_UNARY_FUNCTOR(RoundHalf2, h2rint);
DEFINE_UNARY_FUNCTOR(ExpHalf, hexp);
DEFINE_UNARY_FUNCTOR(ExpHalf2, h2exp);
DEFINE_UNARY_FUNCTOR(LogHalf, hlog);
DEFINE_UNARY_FUNCTOR(LogHalf2, h2log);
DEFINE_UNARY_FUNCTOR(InvHalf, hrcp);
DEFINE_UNARY_FUNCTOR(InvHalf2, h2rcp);
DEFINE_UNARY_FUNCTOR(SqrtHalf, hsqrt);
DEFINE_UNARY_FUNCTOR(SqrtHalf2, h2sqrt);
DEFINE_UNARY_FUNCTOR(RsqrtHalf, hrsqrt);
DEFINE_UNARY_FUNCTOR(RsqrtHalf2, h2rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf, hsin);
DEFINE_UNARY_FUNCTOR(SinHalf2, h2sin);
DEFINE_UNARY_FUNCTOR(CosHalf, hcos);
DEFINE_UNARY_FUNCTOR(CosHalf2, h2cos);
#endif
#undef DEFINE_UNARY_FUNCTOR

#define DEFINE_UNARY_FUNCTOR(name, func)               \
  template <typename T>                                \
  struct name##Functor {                               \
    inline __device__ T operator()(const T& x) const { \
      return __float2half(func(__half2float(x)));      \
    }                                                  \
  }

#if __CUDA_ARCH__ < 530
DEFINE_UNARY_FUNCTOR(NegHalf, -);
DEFINE_UNARY_FUNCTOR(CeilHalf, ceil);
DEFINE_UNARY_FUNCTOR(FloorHalf, floor);
DEFINE_UNARY_FUNCTOR(RoundHalf, round);
DEFINE_UNARY_FUNCTOR(ExpHalf, exp);
DEFINE_UNARY_FUNCTOR(LogHalf, log);
DEFINE_UNARY_FUNCTOR(InvHalf, __frcp_rn);
DEFINE_UNARY_FUNCTOR(SqrtHalf, sqrt);
DEFINE_UNARY_FUNCTOR(RsqrtHalf, rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf, sin);
DEFINE_UNARY_FUNCTOR(CosHalf, cos);
#endif
#undef DEFINE_UNARY_FUNCTOR

#define DEFINE_UNARY_FUNCTOR(name, func)               \
  template <typename T>                                \
  struct name##Functor {                               \
    inline __device__ T operator()(const T& x) const { \
      const float2 v = __half22float2(x);              \
      return __floats2half2_rn(func(v.x), func(v.y));  \
    }                                                  \
  }

#if __CUDA_ARCH__ < 530
DEFINE_UNARY_FUNCTOR(NegHalf2, -);
DEFINE_UNARY_FUNCTOR(CeilHalf2, ceil);
DEFINE_UNARY_FUNCTOR(FloorHalf2, floor);
DEFINE_UNARY_FUNCTOR(RoundHalf2, round);
DEFINE_UNARY_FUNCTOR(ExpHalf2, exp);
DEFINE_UNARY_FUNCTOR(LogHalf2, log);
DEFINE_UNARY_FUNCTOR(InvHalf2, __frcp_rn);
DEFINE_UNARY_FUNCTOR(SqrtHalf2, sqrt);
DEFINE_UNARY_FUNCTOR(RsqrtHalf2, rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf2, sin);
DEFINE_UNARY_FUNCTOR(CosHalf2, cos);
#endif
#undef DEFINE_UNARY_FUNCTOR

/*!
 * Unary Function Kernels
 */

template <typename InputT, typename OutputT, class Functor>
__global__ void
_SimpleUnaryFunc(const int N, const Functor op, const InputT* x, OutputT* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = op(x[i]);
  }
}

template <typename T>
__global__ void _Abs(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T val = x[i];
    y[i] = val > 0 ? val : -val;
  }
}

template <>
__global__ void _Abs<half>(const int N, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float val = __half2float(x[i]);
    y[i] = __float2half(val > 0 ? val : -val);
  }
}

template <>
__global__ void _Abs<half2>(const int N, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 v = __half22float2(x[i]);
    y[i] = __floats2half2_rn(v.x > 0.f ? v.x : -v.x, v.y > 0.f ? v.y : -v.y);
  }
}

__global__ void _Inv(const int N, const float* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __frcp_rn(x[i]);
  }
}

__global__ void _Inv(const int N, const double* x, double* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __drcp_rn(x[i]);
  }
}

template <typename T>
__global__ void _InvStd(const int N, const T eps, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = rsqrt(x[i] + eps);
  }
}

__global__ void _InvStd(const int N, const float eps, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(rsqrt(__half2float(x[i]) + eps));
  }
}

__global__ void
_InvStd(const int N, const float eps, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 v = __half22float2(x[i]);
    y[i] = __floats2half2_rn(rsqrt(v.x + eps), rsqrt(v.y + eps));
  }
}

template <typename T>
__global__ void _Powx(const int N, const T exponent, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = pow(x[i], exponent);
  }
}

__global__ void
_Powx(const int N, const float exponent, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(pow(__half2float(x[i]), exponent));
  }
}

__global__ void
_Powx(const int N, const float exponent, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 v = __half22float2(x[i]);
    y[i] = __floats2half2_rn(pow(v.x, exponent), pow(v.y, exponent));
  }
}

template <typename T>
__global__ void _Set(const int N, const T alpha, T* x) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    x[i] = alpha;
  }
}

template <typename T>
__global__ void _Sign(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::Sign(x[i]);
  }
}

template <>
__global__ void _Sign<uint8_t>(const int N, const uint8_t* x, uint8_t* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] > 0 ? uint8_t(1) : uint8_t(0);
  }
}

template <>
__global__ void _Sign<half>(const int N, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float v = __half2float(x[i]);
    y[i] = __float2half(math::utils::Sign(v));
  }
}

template <>
__global__ void _Sign<half2>(const int N, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float2 v = __half22float2(x[i]);
    y[i] = __floats2half2_rn(math::utils::Sign(v.x), math::utils::Sign(v.y));
  }
}

template <typename T>
__global__ void _Square(const int N, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::Square(x[i]);
  }
}

template <typename T>
__global__ void _NotZero(const int N, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] != T(0) ? true : false;
  }
}

template <>
__global__ void _NotZero<half>(const int N, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __half2float(x[i]) != 0.f ? true : false;
  }
}

template <typename T>
__global__ void _IsInf(const int N, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::IsInf(x[i]);
  }
}

template <typename T>
__global__ void _IsNaN(const int N, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::IsNaN(x[i]);
  }
}

template <typename T>
__global__ void _IsFinite(const int N, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::IsFinite(x[i]);
  }
}

template <typename T>
__global__ void _ReplaceNaN(const int N, const T value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::IsNaN(__ldg(x + i)) ? value : __ldg(x + i);
  }
}

template <typename T, class Functor>
__global__ void
_Bias(const int N, const T beta, const Functor op, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = op(x[i], beta);
  }
}

template <typename T>
__global__ void
_ApplyMask(const int N, const T alpha, const uint8_t* mask, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] * T(mask[i]) * alpha;
  }
}

__global__ void _ApplyMask(
    const int N,
    const float alpha,
    const uint8_t* mask,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = __float2half(__half2float(x[i]) * (alpha * float(mask[i])));
  }
}

/*!
 * Binary Function Kernels
 */

template <typename InputT, typename OutputT, class Functor>
__global__ void _SimpleBinaryFunc(
    const int N,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = op(a[i], b[i]);
  }
}

template <typename T>
__global__ void
_Where(const int N, const T* a, const T* b, const bool* c, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = c[i] ? a[i] : b[i];
  }
}

} // namespace

#define DEFINE_UNARY_FUNC(name, InputT, OutputT, Functor)                      \
  template <>                                                                  \
  DRAGON_API void name<InputT, CUDAContext>(                                   \
      const int N, const InputT* x, OutputT* y, CUDAContext* ctx) {            \
    _SimpleUnaryFunc<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                                     \
        Functor<math::ScalarType<InputT>::type>(),                             \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x),            \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));                \
  }

DEFINE_UNARY_FUNC(BitwiseNot, bool, bool, math::BitNotFunctor);
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
DEFINE_UNARY_FUNC(Not, float, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Not, double, bool, math::NotFunctor);
DEFINE_UNARY_FUNC(Neg, int8_t, int8_t, NegFunctor);
DEFINE_UNARY_FUNC(Neg, int, int, NegFunctor);
DEFINE_UNARY_FUNC(Neg, int64_t, int64_t, NegFunctor);
DEFINE_UNARY_FUNC(Neg, float, float, NegFunctor);
DEFINE_UNARY_FUNC(Neg, double, double, NegFunctor);
DEFINE_UNARY_FUNC(Ceil, float, float, CeilFunctor);
DEFINE_UNARY_FUNC(Ceil, double, double, CeilFunctor);
DEFINE_UNARY_FUNC(Floor, float, float, FloorFunctor);
DEFINE_UNARY_FUNC(Floor, double, double, FloorFunctor);
DEFINE_UNARY_FUNC(Round, float, float, RoundFunctor);
DEFINE_UNARY_FUNC(Round, double, double, RoundFunctor);
DEFINE_UNARY_FUNC(Exp, float, float, ExpFunctor);
DEFINE_UNARY_FUNC(Exp, double, double, ExpFunctor);
DEFINE_UNARY_FUNC(Log, float, float, LogFunctor);
DEFINE_UNARY_FUNC(Log, double, double, LogFunctor);
DEFINE_UNARY_FUNC(Sqrt, float, float, SqrtFunctor);
DEFINE_UNARY_FUNC(Sqrt, double, double, SqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, float, float, RsqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, double, double, RsqrtFunctor);
DEFINE_UNARY_FUNC(Sin, float, float, SinFunctor);
DEFINE_UNARY_FUNC(Sin, double, double, SinFunctor);
DEFINE_UNARY_FUNC(Cos, float, float, CosFunctor);
DEFINE_UNARY_FUNC(Cos, double, double, CosFunctor);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, T)                                             \
  template <>                                                                  \
  DRAGON_API void name<T, CUDAContext>(                                        \
      const int N, const T* x, T* y, CUDAContext* ctx) {                       \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(N, x, y); \
  }

DEFINE_UNARY_FUNC(Inv, float);
DEFINE_UNARY_FUNC(Inv, double);
DEFINE_UNARY_FUNC(Abs, uint8_t);
DEFINE_UNARY_FUNC(Abs, int8_t);
DEFINE_UNARY_FUNC(Abs, int);
DEFINE_UNARY_FUNC(Abs, int64_t);
DEFINE_UNARY_FUNC(Abs, float);
DEFINE_UNARY_FUNC(Abs, double);
DEFINE_UNARY_FUNC(Square, uint8_t);
DEFINE_UNARY_FUNC(Square, int8_t);
DEFINE_UNARY_FUNC(Square, int);
DEFINE_UNARY_FUNC(Square, int64_t);
DEFINE_UNARY_FUNC(Square, float);
DEFINE_UNARY_FUNC(Square, double);
DEFINE_UNARY_FUNC(Sign, uint8_t);
DEFINE_UNARY_FUNC(Sign, int8_t);
DEFINE_UNARY_FUNC(Sign, int);
DEFINE_UNARY_FUNC(Sign, int64_t);
DEFINE_UNARY_FUNC(Sign, float);
DEFINE_UNARY_FUNC(Sign, double);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, HalfFunctor, Half2Functor)           \
  template <>                                                        \
  DRAGON_API void name<float16, CUDAContext>(                        \
      const int N, const float16* x, float16* y, CUDAContext* ctx) { \
    if ((N & 1) == 0) {                                              \
      _SimpleUnaryFunc<<<                                            \
          CUDA_BLOCKS(N >> 1),                                       \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N >> 1,                                                    \
          Half2Functor<half2>(),                                     \
          reinterpret_cast<const half2*>(x),                         \
          reinterpret_cast<half2*>(y));                              \
    } else {                                                         \
      _SimpleUnaryFunc<<<                                            \
          CUDA_BLOCKS(N),                                            \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N,                                                         \
          HalfFunctor<half>(),                                       \
          reinterpret_cast<const half*>(x),                          \
          reinterpret_cast<half*>(y));                               \
    }                                                                \
  }

DEFINE_UNARY_FUNC(Neg, NegHalfFunctor, NegHalf2Functor);
DEFINE_UNARY_FUNC(Ceil, CeilHalfFunctor, CeilHalf2Functor);
DEFINE_UNARY_FUNC(Floor, FloorHalfFunctor, FloorHalf2Functor);
DEFINE_UNARY_FUNC(Round, RoundHalfFunctor, RoundHalf2Functor);
DEFINE_UNARY_FUNC(Exp, ExpHalfFunctor, ExpHalf2Functor);
DEFINE_UNARY_FUNC(Log, LogHalfFunctor, LogHalf2Functor);
DEFINE_UNARY_FUNC(Inv, InvHalfFunctor, InvHalf2Functor);
DEFINE_UNARY_FUNC(Sqrt, SqrtHalfFunctor, SqrtHalf2Functor);
DEFINE_UNARY_FUNC(Rsqrt, RsqrtHalfFunctor, RsqrtHalf2Functor);
DEFINE_UNARY_FUNC(Sin, SinHalfFunctor, SinHalf2Functor);
DEFINE_UNARY_FUNC(Cos, CosHalfFunctor, CosHalf2Functor);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name)                                              \
  template <>                                                                \
  DRAGON_API void name<float16, CUDAContext>(                                \
      const int N, const float16* x, float16* y, CUDAContext* ctx) {         \
    if ((N & 1) == 0) {                                                      \
      _##name<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N >> 1,                                                            \
          reinterpret_cast<const half2*>(x),                                 \
          reinterpret_cast<half2*>(y));                                      \
    } else {                                                                 \
      _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
          N, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));  \
    }                                                                        \
  }

DEFINE_UNARY_FUNC(Abs);
DEFINE_UNARY_FUNC(Square);
DEFINE_UNARY_FUNC(Sign);
#undef DEFINE_UNARY_FUNC

#define DEFINE_SET_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Set<T, CUDAContext>(                                      \
      const int N, const T value, T* y, CUDAContext* ctx) {                 \
    if (value == T(0)) {                                                    \
      CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(T) * N, ctx->cuda_stream())); \
    } else {                                                                \
      _Set<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
          N, value, y);                                                     \
    }                                                                       \
  }

template <>
DRAGON_API void Set<float16, CUDAContext>(
    const int N,
    const float16 value,
    float16* y,
    CUDAContext* ctx) {
  if (value.x == (unsigned short)0) {
    CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(float16) * N, ctx->cuda_stream()));
    return;
  }
  if ((N & 1) == 0) {
    _Set<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1, convert::To<half2>(value), reinterpret_cast<half2*>(y));
  } else {
    _Set<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(N, value, y);
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

#define DEFINE_INVSTD_FUNC(T)                                             \
  template <>                                                             \
  DRAGON_API void InvStd<T, CUDAContext>(                                 \
      const int N, const float eps, const T* x, T* y, CUDAContext* ctx) { \
    _InvStd<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
        N, (T)eps, x, y);                                                 \
  }

template <>
DRAGON_API void InvStd<float16, CUDAContext>(
    const int N,
    const float eps,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _InvStd<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        eps,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _InvStd<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N, eps, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
}

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

#define DEFINE_POWX_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void Powx<T, CUDAContext>(                                        \
      const int N, const float exponent, const T* x, T* y, CUDAContext* ctx) { \
    _Powx<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(            \
        N, (T)exponent, x, y);                                                 \
  }

template <>
DRAGON_API void Powx<float16, CUDAContext>(
    const int N,
    const float exponent,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((N & 1) == 0) {
    _Powx<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        exponent,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Powx<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        exponent,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

#define DEFINE_NOT_ZERO_FUNC(T)                                        \
  template <>                                                          \
  DRAGON_API void NotZero<T, CUDAContext>(                             \
      const int N, const T* x, bool* y, CUDAContext* ctx) {            \
    _NotZero<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, x, y);                                                      \
  }

template <>
DRAGON_API void NotZero<float16, CUDAContext>(
    const int N,
    const float16* x,
    bool* y,
    CUDAContext* ctx) {
  _NotZero<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      N, reinterpret_cast<const half*>(x), y);
}

DEFINE_NOT_ZERO_FUNC(bool);
DEFINE_NOT_ZERO_FUNC(uint8_t);
DEFINE_NOT_ZERO_FUNC(int8_t);
DEFINE_NOT_ZERO_FUNC(int);
DEFINE_NOT_ZERO_FUNC(int64_t);
DEFINE_NOT_ZERO_FUNC(float);
DEFINE_NOT_ZERO_FUNC(double);
#undef DEFINE_NOT_ZERO_FUNC

#define DEFINE_IS_FUNC(name, T)                                       \
  template <>                                                         \
  DRAGON_API void name<T, CUDAContext>(                               \
      const int N, const T* x, bool* y, CUDAContext* ctx) {           \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, reinterpret_cast<const math::ScalarType<T>::type*>(x), y); \
  }

DEFINE_IS_FUNC(IsInf, float16);
DEFINE_IS_FUNC(IsInf, float);
DEFINE_IS_FUNC(IsInf, double);
DEFINE_IS_FUNC(IsNaN, float16);
DEFINE_IS_FUNC(IsNaN, float);
DEFINE_IS_FUNC(IsNaN, double);
DEFINE_IS_FUNC(IsFinite, float16);
DEFINE_IS_FUNC(IsFinite, float);
DEFINE_IS_FUNC(IsFinite, double);
#undef DEFINE_IS_FUNC

#define DEFINE_REPLACE_NAN_FUNC(T)                                          \
  template <>                                                               \
  DRAGON_API void ReplaceNaN<T, CUDAContext>(                               \
      const int N, const float value, const T* x, T* y, CUDAContext* ctx) { \
    _ReplaceNaN<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        N,                                                                  \
        convert::To<math::ScalarType<T>::type>(value),                      \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),              \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                   \
  }

DEFINE_REPLACE_NAN_FUNC(float16);
DEFINE_REPLACE_NAN_FUNC(float);
DEFINE_REPLACE_NAN_FUNC(double);
#undef DEFINE_REPLACE_NAN_FUNC

#define DEFINE_BIAS_FUNC(T)                                                \
  template <>                                                              \
  DRAGON_API void Bias<T, CUDAContext>(                                    \
      const int N, const float beta, const T* x, T* y, CUDAContext* ctx) { \
    if (beta == 0.f) return;                                               \
    _Bias<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
        N, (T)beta, math::PlusFunctor<T>(), x, y);                         \
  }

template <>
DRAGON_API void Bias<float16, CUDAContext>(
    const int N,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if (beta == 0.f) return;
  if ((N & 1) == 0) {
    _Bias<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N >> 1,
        convert::To<half2>(beta),
        math::PlusFunctor<half2>(),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Bias<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        N,
        convert::To<half>(beta),
        math::PlusFunctor<half>(),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

DEFINE_BIAS_FUNC(uint8_t);
DEFINE_BIAS_FUNC(int8_t);
DEFINE_BIAS_FUNC(int);
DEFINE_BIAS_FUNC(int64_t);
DEFINE_BIAS_FUNC(float);
DEFINE_BIAS_FUNC(double);
#undef DEFINE_BIAS_FUNC

#define DEFINE_APPLY_MASK_FUNC(T, AccT)                                  \
  template <>                                                            \
  DRAGON_API void ApplyMask<T, CUDAContext>(                             \
      const int N,                                                       \
      const float alpha,                                                 \
      const uint8_t* mask,                                               \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    _ApplyMask<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                               \
        convert::To<AccT>(alpha),                                        \
        mask,                                                            \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),           \
        reinterpret_cast<math::ScalarType<T>::type*>(y));                \
  }

DEFINE_APPLY_MASK_FUNC(uint8_t, uint8_t);
DEFINE_APPLY_MASK_FUNC(int8_t, int8_t);
DEFINE_APPLY_MASK_FUNC(int, int);
DEFINE_APPLY_MASK_FUNC(int64_t, int64_t);
DEFINE_APPLY_MASK_FUNC(float16, float);
DEFINE_APPLY_MASK_FUNC(float, float);
DEFINE_APPLY_MASK_FUNC(double, double);
#undef DEFINE_APPLY_MASK_FUNC

#define DEFINE_BINARY_FUNC(name, T, Functor)                         \
  template <>                                                        \
  DRAGON_API void name<T, CUDAContext>(                              \
      const int N, const T* a, const T* b, T* y, CUDAContext* ctx) { \
    using ScalarT = typename math::ScalarType<T>::type;              \
    using ScalarT2 = typename math::ScalarType<T>::type2;            \
    if ((N & 1) == 0 && sizeof(ScalarT) != sizeof(ScalarT2)) {       \
      _SimpleBinaryFunc<<<                                           \
          CUDA_BLOCKS(N >> 1),                                       \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N >> 1,                                                    \
          Functor<ScalarT2>(),                                       \
          reinterpret_cast<const ScalarT2*>(a),                      \
          reinterpret_cast<const ScalarT2*>(b),                      \
          reinterpret_cast<ScalarT2*>(y));                           \
    } else {                                                         \
      _SimpleBinaryFunc<<<                                           \
          CUDA_BLOCKS(N),                                            \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N,                                                         \
          Functor<ScalarT>(),                                        \
          reinterpret_cast<const ScalarT*>(a),                       \
          reinterpret_cast<const ScalarT*>(b),                       \
          reinterpret_cast<ScalarT*>(y));                            \
    }                                                                \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int64_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float16, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, double, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, uint8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int64_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float16, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, double, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, uint8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int64_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float16, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, double, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, uint8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int64_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float16, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, double, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float16, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, float, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, double, math::PowFunctor);
DEFINE_BINARY_FUNC(Atan2, float16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, float, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, double, math::Atan2Functor);
DEFINE_BINARY_FUNC(Minimum, uint8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int64_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float16, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, double, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, uint8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int64_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float16, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, double, math::MaxFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Functor)          \
  template <>                                                       \
  DRAGON_API void name<InputT, CUDAContext>(                        \
      const int N,                                                  \
      const InputT* a,                                              \
      const InputT* b,                                              \
      OutputT* y,                                                   \
      CUDAContext* ctx) {                                           \
    _SimpleBinaryFunc<<<                                            \
        CUDA_BLOCKS(N),                                             \
        CUDA_THREADS,                                               \
        0,                                                          \
        ctx->cuda_stream()>>>(                                      \
        N,                                                          \
        Functor<math::ScalarType<InputT>::type>(),                  \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(a), \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(b), \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));     \
  }

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
DEFINE_BINARY_FUNC(And, float, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(And, double, bool, math::AndFunctor);
DEFINE_BINARY_FUNC(Or, bool, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, uint8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int8_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, int64_t, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float16, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, float, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Or, double, bool, math::OrFunctor);
DEFINE_BINARY_FUNC(Xor, bool, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, uint8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int8_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, int64_t, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float16, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, float, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Xor, double, bool, math::XorFunctor);
DEFINE_BINARY_FUNC(Equal, bool, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float16, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, double, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float16, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, double, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, bool, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int64_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float16, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, double, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float16, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, double, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, bool, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float16, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, double, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T)                                         \
  template <>                                                        \
  DRAGON_API void Where<T, CUDAContext>(                             \
      const int N,                                                   \
      const T* a,                                                    \
      const T* b,                                                    \
      const bool* c,                                                 \
      T* y,                                                          \
      CUDAContext* ctx) {                                            \
    _Where<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, a, b, c, y);                                              \
  }

DEFINE_WHERE_FUNC(uint8_t);
DEFINE_WHERE_FUNC(bool);
DEFINE_WHERE_FUNC(int8_t);
DEFINE_WHERE_FUNC(int);
DEFINE_WHERE_FUNC(int64_t);
DEFINE_WHERE_FUNC(float16);
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
