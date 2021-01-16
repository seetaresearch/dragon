#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
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
DEFINE_UNARY_FUNCTOR(Cos, cos);
DEFINE_UNARY_FUNCTOR(Exp, exp);
DEFINE_UNARY_FUNCTOR(Floor, floor);
DEFINE_UNARY_FUNCTOR(Log, log);
DEFINE_UNARY_FUNCTOR(Round, round);
DEFINE_UNARY_FUNCTOR(Rsqrt, rsqrt);
DEFINE_UNARY_FUNCTOR(Sin, sin);
DEFINE_UNARY_FUNCTOR(Sqrt, sqrt);
#if __CUDA_ARCH__ >= 530
DEFINE_UNARY_FUNCTOR(NegHalf, __hneg);
DEFINE_UNARY_FUNCTOR(NegHalf2, __hneg2);
DEFINE_UNARY_FUNCTOR(CeilHalf, hceil);
DEFINE_UNARY_FUNCTOR(CeilHalf2, h2ceil);
DEFINE_UNARY_FUNCTOR(CosHalf, hcos);
DEFINE_UNARY_FUNCTOR(CosHalf2, h2cos);
DEFINE_UNARY_FUNCTOR(ExpHalf, hexp);
DEFINE_UNARY_FUNCTOR(ExpHalf2, h2exp);
DEFINE_UNARY_FUNCTOR(FloorHalf, hfloor);
DEFINE_UNARY_FUNCTOR(FloorHalf2, h2floor);
DEFINE_UNARY_FUNCTOR(InvHalf, hrcp);
DEFINE_UNARY_FUNCTOR(InvHalf2, h2rcp);
DEFINE_UNARY_FUNCTOR(LogHalf, hlog);
DEFINE_UNARY_FUNCTOR(LogHalf2, h2log);
DEFINE_UNARY_FUNCTOR(RoundHalf, hrint);
DEFINE_UNARY_FUNCTOR(RoundHalf2, h2rint);
DEFINE_UNARY_FUNCTOR(RsqrtHalf, hrsqrt);
DEFINE_UNARY_FUNCTOR(RsqrtHalf2, h2rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf, hsin);
DEFINE_UNARY_FUNCTOR(SinHalf2, h2sin);
DEFINE_UNARY_FUNCTOR(SqrtHalf, hsqrt);
DEFINE_UNARY_FUNCTOR(SqrtHalf2, h2sqrt);
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
DEFINE_UNARY_FUNCTOR(CosHalf, cos);
DEFINE_UNARY_FUNCTOR(ExpHalf, exp);
DEFINE_UNARY_FUNCTOR(FloorHalf, floor);
DEFINE_UNARY_FUNCTOR(InvHalf, __frcp_rn);
DEFINE_UNARY_FUNCTOR(LogHalf, log);
DEFINE_UNARY_FUNCTOR(RoundHalf, round);
DEFINE_UNARY_FUNCTOR(RsqrtHalf, rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf, sin);
DEFINE_UNARY_FUNCTOR(SqrtHalf, sqrt);
#endif
#undef DEFINE_UNARY_FUNCTOR

#define DEFINE_UNARY_FUNCTOR(name, func)                  \
  template <typename T>                                   \
  struct name##Functor {                                  \
    inline __device__ T operator()(const T& x) const {    \
      const float2 val = __half22float2(x);               \
      return __floats2half2_rn(func(val.x), func(val.y)); \
    }                                                     \
  }

#if __CUDA_ARCH__ < 530
DEFINE_UNARY_FUNCTOR(NegHalf2, -);
DEFINE_UNARY_FUNCTOR(CeilHalf2, ceil);
DEFINE_UNARY_FUNCTOR(CosHalf2, cos);
DEFINE_UNARY_FUNCTOR(ExpHalf2, exp);
DEFINE_UNARY_FUNCTOR(FloorHalf2, floor);
DEFINE_UNARY_FUNCTOR(InvHalf2, __frcp_rn);
DEFINE_UNARY_FUNCTOR(LogHalf2, log);
DEFINE_UNARY_FUNCTOR(RoundHalf2, round);
DEFINE_UNARY_FUNCTOR(RsqrtHalf2, rsqrt);
DEFINE_UNARY_FUNCTOR(SinHalf2, sin);
DEFINE_UNARY_FUNCTOR(SqrtHalf2, sqrt);
#endif
#undef DEFINE_UNARY_FUNCTOR

/*!
 * Unary Function Kernels
 */

template <typename T, class Functor>
__global__ void
_SimpleUnaryFunc(const int nthreads, const Functor op, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = op(x[i]);
  }
}

template <typename T>
__global__ void _Abs(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T val = x[i];
    y[i] = val > 0 ? val : -val;
  }
}

template <>
__global__ void _Abs<half>(const int n, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float val = __half2float(x[i]);
    y[i] = __float2half(val > 0 ? val : -val);
  }
}

template <>
__global__ void _Abs<half2>(const int n, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(
        val.x > 0.f ? val.x : -val.x, val.y > 0.f ? val.y : -val.y);
  }
}

__global__ void _Inv(const int n, const float* x, float* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __frcp_rn(x[i]);
  }
}

__global__ void _Inv(const int n, const double* x, double* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __drcp_rn(x[i]);
  }
}

template <typename T>
__global__ void _InvStd(const int n, const T eps, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = rsqrt(x[i] + eps);
  }
}

__global__ void _InvStd(const int n, const float eps, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __float2half(rsqrt(__half2float(x[i]) + eps));
  }
}

__global__ void
_InvStd(const int n, const float eps, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(rsqrt(val.x + eps), rsqrt(val.y + eps));
  }
}

template <typename T>
__global__ void _Invert(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = ~x[i];
  }
}

template <>
__global__ void _Invert<bool>(const int n, const bool* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = !x[i];
  }
}

template <typename T>
__global__ void _Powx(const int n, const T exponent, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = pow(x[i], exponent);
  }
}

__global__ void
_Powx(const int n, const float exponent, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = __float2half(pow(__half2float(x[i]), exponent));
  }
}

__global__ void
_Powx(const int n, const float exponent, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(pow(val.x, exponent), pow(val.y, exponent));
  }
}

template <typename T>
__global__ void _Set(const int n, const T alpha, T* x) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    x[i] = alpha;
  }
}

template <typename T>
__global__ void _Sign(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::Sign(x[i]);
  }
}

template <>
__global__ void _Sign<uint8_t>(const int n, const uint8_t* x, uint8_t* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = x[i] > 0 ? uint8_t(1) : uint8_t(0);
  }
}

template <>
__global__ void _Sign<half>(const int n, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float val = __half2float(x[i]);
    y[i] = __float2half(math::utils::Sign(val));
  }
}

template <>
__global__ void _Sign<half2>(const int n, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float2 val = __half22float2(x[i]);
    y[i] =
        __floats2half2_rn(math::utils::Sign(val.x), math::utils::Sign(val.y));
  }
}

template <typename T>
__global__ void _Square(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::Square(x[i]);
  }
}

template <typename T>
__global__ void _NotZero(const int nthreads, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[i] != T(0) ? true : false;
  }
}

template <>
__global__ void _NotZero<half>(const int nthreads, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __half2float(x[i]) != 0.f ? true : false;
  }
}

template <typename T>
__global__ void _IsInf(const int n, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::IsInf(x[i]);
  }
}

template <>
__global__ void _IsInf<half>(const int n, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::IsInf(x[i]);
  }
}

template <typename T>
__global__ void _IsNaN(const int n, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::IsNaN(x[i]);
  }
}

template <>
__global__ void _IsNaN<half>(const int n, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = math::utils::IsNaN(x[i]);
  }
}

template <typename T>
__global__ void _ReplaceNaN(const int n, const T value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = math::utils::IsNaN(__ldg(x + i)) ? value : __ldg(x + i);
#else
    y[i] = math::utils::IsNaN(x[i]) ? value : x[i];
#endif
  }
}

template <>
__global__ void
_ReplaceNaN<half>(const int n, const half value, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = math::utils::IsNaN(__ldg(x + i)) ? value : __ldg(x + i);
#else
    y[i] = math::utils::IsNaN(x[i]) ? value : x[i];
#endif
  }
}

template <typename T, class Functor>
__global__ void
_Bias(const int n, const T beta, const Functor op, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = op(x[i], beta);
  }
}

/*!
 * Binary Function Kernels
 */

template <typename InputT, typename OutputT, class Functor>
__global__ void _SimpleBinaryFunc(
    const int n,
    const Functor op,
    const InputT* a,
    const InputT* b,
    OutputT* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = op(a[i], b[i]);
  }
}

template <typename T>
__global__ void
_Where(const int n, const T* a, const T* b, const bool* c, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = c[i] ? a[i] : b[i];
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_UNARY_FUNC(name, T, Functor)                                    \
  template <>                                                                  \
  DRAGON_API void name<T, CUDAContext>(                                        \
      const int n, const T* x, T* y, CUDAContext* ctx) {                       \
    _SimpleUnaryFunc<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, Functor<T>(), x, y);                                                \
  }

DEFINE_UNARY_FUNC(Neg, int8_t, NegFunctor);
DEFINE_UNARY_FUNC(Neg, int, NegFunctor);
DEFINE_UNARY_FUNC(Neg, int64_t, NegFunctor);
DEFINE_UNARY_FUNC(Neg, float, NegFunctor);
DEFINE_UNARY_FUNC(Neg, double, NegFunctor);
DEFINE_UNARY_FUNC(Ceil, float, CeilFunctor);
DEFINE_UNARY_FUNC(Ceil, double, CeilFunctor);
DEFINE_UNARY_FUNC(Cos, float, CosFunctor);
DEFINE_UNARY_FUNC(Cos, double, CosFunctor);
DEFINE_UNARY_FUNC(Exp, float, ExpFunctor);
DEFINE_UNARY_FUNC(Exp, double, ExpFunctor);
DEFINE_UNARY_FUNC(Floor, float, FloorFunctor);
DEFINE_UNARY_FUNC(Floor, double, FloorFunctor);
DEFINE_UNARY_FUNC(Log, float, LogFunctor);
DEFINE_UNARY_FUNC(Log, double, LogFunctor);
DEFINE_UNARY_FUNC(Round, float, RoundFunctor);
DEFINE_UNARY_FUNC(Round, double, RoundFunctor);
DEFINE_UNARY_FUNC(Rsqrt, float, RsqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, double, RsqrtFunctor);
DEFINE_UNARY_FUNC(Sin, float, SinFunctor);
DEFINE_UNARY_FUNC(Sin, double, SinFunctor);
DEFINE_UNARY_FUNC(Sqrt, float, SqrtFunctor);
DEFINE_UNARY_FUNC(Sqrt, double, SqrtFunctor);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, T)                                             \
  template <>                                                                  \
  DRAGON_API void name<T, CUDAContext>(                                        \
      const int n, const T* x, T* y, CUDAContext* ctx) {                       \
    _##name<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(n, x, y); \
  }

DEFINE_UNARY_FUNC(Abs, int8_t);
DEFINE_UNARY_FUNC(Abs, uint8_t);
DEFINE_UNARY_FUNC(Abs, int);
DEFINE_UNARY_FUNC(Abs, int64_t);
DEFINE_UNARY_FUNC(Abs, float);
DEFINE_UNARY_FUNC(Abs, double);
DEFINE_UNARY_FUNC(Inv, float);
DEFINE_UNARY_FUNC(Inv, double);
DEFINE_UNARY_FUNC(Invert, bool);
DEFINE_UNARY_FUNC(Invert, int8_t);
DEFINE_UNARY_FUNC(Invert, uint8_t);
DEFINE_UNARY_FUNC(Invert, int);
DEFINE_UNARY_FUNC(Invert, int64_t);
DEFINE_UNARY_FUNC(Sign, int8_t);
DEFINE_UNARY_FUNC(Sign, uint8_t);
DEFINE_UNARY_FUNC(Sign, int);
DEFINE_UNARY_FUNC(Sign, int64_t);
DEFINE_UNARY_FUNC(Sign, float);
DEFINE_UNARY_FUNC(Sign, double);
DEFINE_UNARY_FUNC(Square, int8_t);
DEFINE_UNARY_FUNC(Square, uint8_t);
DEFINE_UNARY_FUNC(Square, int);
DEFINE_UNARY_FUNC(Square, int64_t);
DEFINE_UNARY_FUNC(Square, float);
DEFINE_UNARY_FUNC(Square, double);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name, HalfFunctor, Half2Functor)           \
  template <>                                                        \
  DRAGON_API void name<float16, CUDAContext>(                        \
      const int n, const float16* x, float16* y, CUDAContext* ctx) { \
    if ((n & 1) == 0) {                                              \
      _SimpleUnaryFunc<<<                                            \
          CUDA_BLOCKS(n >> 1),                                       \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          n >> 1,                                                    \
          Half2Functor<half2>(),                                     \
          reinterpret_cast<const half2*>(x),                         \
          reinterpret_cast<half2*>(y));                              \
    } else {                                                         \
      _SimpleUnaryFunc<<<                                            \
          CUDA_BLOCKS(n),                                            \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          n,                                                         \
          HalfFunctor<half>(),                                       \
          reinterpret_cast<const half*>(x),                          \
          reinterpret_cast<half*>(y));                               \
    }                                                                \
  }

DEFINE_UNARY_FUNC(Neg, NegHalfFunctor, NegHalf2Functor);
DEFINE_UNARY_FUNC(Ceil, CeilHalfFunctor, CeilHalf2Functor);
DEFINE_UNARY_FUNC(Cos, CosHalfFunctor, CosHalf2Functor);
DEFINE_UNARY_FUNC(Exp, ExpHalfFunctor, ExpHalf2Functor);
DEFINE_UNARY_FUNC(Floor, FloorHalfFunctor, FloorHalf2Functor);
DEFINE_UNARY_FUNC(Log, LogHalfFunctor, LogHalf2Functor);
DEFINE_UNARY_FUNC(Inv, InvHalfFunctor, InvHalf2Functor);
DEFINE_UNARY_FUNC(Round, RoundHalfFunctor, RoundHalf2Functor);
DEFINE_UNARY_FUNC(Rsqrt, RsqrtHalfFunctor, RsqrtHalf2Functor);
DEFINE_UNARY_FUNC(Sin, SinHalfFunctor, SinHalf2Functor);
DEFINE_UNARY_FUNC(Sqrt, SqrtHalfFunctor, SqrtHalf2Functor);
#undef DEFINE_UNARY_FUNC

#define DEFINE_UNARY_FUNC(name)                                              \
  template <>                                                                \
  DRAGON_API void name<float16, CUDAContext>(                                \
      const int n, const float16* x, float16* y, CUDAContext* ctx) {         \
    if ((n & 1) == 0) {                                                      \
      _##name<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          n >> 1,                                                            \
          reinterpret_cast<const half2*>(x),                                 \
          reinterpret_cast<half2*>(y));                                      \
    } else {                                                                 \
      _##name<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
          n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));  \
    }                                                                        \
  }

DEFINE_UNARY_FUNC(Abs);
DEFINE_UNARY_FUNC(Sign);
DEFINE_UNARY_FUNC(Square);
#undef DEFINE_UNARY_FUNC

/* y = value */

#define DEFINE_SET_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Set<T, CUDAContext>(                                      \
      const int n, const T value, T* y, CUDAContext* ctx) {                 \
    if (value == T(0)) {                                                    \
      CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(T) * n, ctx->cuda_stream())); \
    } else {                                                                \
      _Set<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
          n, value, y);                                                     \
    }                                                                       \
  }

template <>
DRAGON_API void Set<float16, CUDAContext>(
    const int n,
    const float16 value,
    float16* y,
    CUDAContext* ctx) {
  if (value.x == (unsigned short)0) {
    CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(float16) * n, ctx->cuda_stream()));
    return;
  }
  if ((n & 1) == 0) {
    _Set<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n >> 1, convert::To<half2>(value), reinterpret_cast<half2*>(y));
  } else {
    _Set<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(n, value, y);
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

#define DEFINE_INVSTD_FUNC(T)                                             \
  template <>                                                             \
  DRAGON_API void InvStd<T, CUDAContext>(                                 \
      const int n, const float eps, const T* x, T* y, CUDAContext* ctx) { \
    _InvStd<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
        n, (T)eps, x, y);                                                 \
  }

template <>
DRAGON_API void InvStd<float16, CUDAContext>(
    const int n,
    const float eps,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((n & 1) == 0) {
    _InvStd<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n >> 1,
        eps,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _InvStd<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n, eps, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
}

DEFINE_INVSTD_FUNC(float);
DEFINE_INVSTD_FUNC(double);
#undef DEFINE_INVSTD_FUNC

/* y = x^e */

#define DEFINE_POWX_FUNC(T)                                                    \
  template <>                                                                  \
  DRAGON_API void Powx<T, CUDAContext>(                                        \
      const int n, const float exponent, const T* x, T* y, CUDAContext* ctx) { \
    _Powx<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(            \
        n, (T)exponent, x, y);                                                 \
  }

template <>
DRAGON_API void Powx<float16, CUDAContext>(
    const int n,
    const float exponent,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if ((n & 1) == 0) {
    _Powx<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n >> 1,
        exponent,
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Powx<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n,
        exponent,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

DEFINE_POWX_FUNC(float);
DEFINE_POWX_FUNC(double);
#undef DEFINE_POWX_FUNC

/* y = notzero(x) */

#define DEFINE_NOT_ZERO_FUNC(T)                                            \
  template <>                                                              \
  void NotZero<T, CUDAContext>(                                            \
      const int count, const T* x, bool* y, CUDAContext* ctx) {            \
    _NotZero<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, x, y);                                                      \
  }

template <>
void NotZero<float16, CUDAContext>(
    const int count,
    const float16* x,
    bool* y,
    CUDAContext* ctx) {
  _NotZero<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      count, reinterpret_cast<const half*>(x), y);
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

#define DEFINE_IS_INF_FUNC(T)                                                 \
  template <>                                                                 \
  DRAGON_API void IsInf<T, CUDAContext>(                                      \
      const int n, const T* x, bool* y, CUDAContext* ctx) {                   \
    _IsInf<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(n, x, y); \
  }

template <>
DRAGON_API void IsInf<float16, CUDAContext>(
    const int n,
    const float16* x,
    bool* y,
    CUDAContext* ctx) {
  _IsInf<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), y);
}

DEFINE_IS_INF_FUNC(float);
DEFINE_IS_INF_FUNC(double);
#undef DEFINE_IS_INF_FUNC

/* y = isnan(x) */

#define DEFINE_IS_NAN_FUNC(T)                                                 \
  template <>                                                                 \
  DRAGON_API void IsNaN<T, CUDAContext>(                                      \
      const int n, const T* x, bool* y, CUDAContext* ctx) {                   \
    _IsNaN<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(n, x, y); \
  }

template <>
DRAGON_API void IsNaN<float16, CUDAContext>(
    const int n,
    const float16* x,
    bool* y,
    CUDAContext* ctx) {
  _IsNaN<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), y);
}

DEFINE_IS_NAN_FUNC(float);
DEFINE_IS_NAN_FUNC(double);
#undef DEFINE_IS_NAN_FUNC

/* y = isnan(x) ? value : x */

#define DEFINE_REPLACE_NAN_FUNC(T)                                        \
  template <>                                                             \
  DRAGON_API void ReplaceNaN<T, CUDAContext>(                             \
      const int n, const T value, const T* x, T* y, CUDAContext* ctx) {   \
    _ReplaceNaN<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, value, x, y);                                                  \
  }

template <>
DRAGON_API void ReplaceNaN<float16, CUDAContext>(
    const int n,
    const float16 value,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  _ReplaceNaN<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      n,
      convert::To<half>(value),
      reinterpret_cast<const half*>(x),
      reinterpret_cast<half*>(y));
}

DEFINE_REPLACE_NAN_FUNC(float);
DEFINE_REPLACE_NAN_FUNC(double);
#undef DEFINE_REPLACE_NAN_FUNC

/* y = x + beta */

#define DEFINE_BIAS_FUNC(T)                                                \
  template <>                                                              \
  DRAGON_API void Bias<T, CUDAContext>(                                    \
      const int n, const float beta, const T* x, T* y, CUDAContext* ctx) { \
    if (beta == 0.f) return;                                               \
    _Bias<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
        n, (T)beta, math::PlusFunctor<T>(), x, y);                         \
  }

template <>
DRAGON_API void Bias<float16, CUDAContext>(
    const int n,
    const float beta,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  if (beta == 0.f) return;
  if ((n & 1) == 0) {
    _Bias<<<CUDA_BLOCKS(n >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n >> 1,
        convert::To<half2>(beta),
        math::PlusFunctor<half2>(),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Bias<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n,
        convert::To<half>(beta),
        math::PlusFunctor<half>(),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  }
}

DEFINE_BIAS_FUNC(int8_t);
DEFINE_BIAS_FUNC(uint8_t);
DEFINE_BIAS_FUNC(int);
DEFINE_BIAS_FUNC(int64_t);
DEFINE_BIAS_FUNC(float);
DEFINE_BIAS_FUNC(double);
#undef DEFINE_BIAS_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Op)    \
  template <>                                            \
  DRAGON_API void name<InputT, CUDAContext>(             \
      const int n,                                       \
      const InputT* a,                                   \
      const InputT* b,                                   \
      OutputT* y,                                        \
      CUDAContext* ctx) {                                \
    _SimpleBinaryFunc<<<                                 \
        CUDA_BLOCKS(n),                                  \
        CUDA_THREADS,                                    \
        0,                                               \
        ctx->cuda_stream()>>>(n, Op<InputT>(), a, b, y); \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int, int, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float, float, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, double, double, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int, int, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float, float, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, double, double, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int, int, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float, float, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, double, double, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int, int, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float, float, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, double, double, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float, float, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, double, double, math::PowFunctor);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int, int, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float, float, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, double, double, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int, int, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float, float, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, double, double, math::MaxFunctor);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, double, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, double, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, int8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int64_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, double, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, double, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, double, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, T, dtype)                           \
  template <>                                                        \
  DRAGON_API void name<T, CUDAContext>(                              \
      const int n, const T* a, const T* b, T* y, CUDAContext* ctx) { \
    name(                                                            \
        n,                                                           \
        reinterpret_cast<const dtype*>(a),                           \
        reinterpret_cast<const dtype*>(b),                           \
        reinterpret_cast<dtype*>(y),                                 \
        ctx);                                                        \
  }

DEFINE_BINARY_FUNC(Add, bool, uint8_t); // Or
DEFINE_BINARY_FUNC(Sub, bool, uint8_t); // Xor
DEFINE_BINARY_FUNC(Mul, bool, uint8_t); // And
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, Functor)     \
  template <>                                 \
  DRAGON_API void name<float16, CUDAContext>( \
      const int n,                            \
      const float16* a,                       \
      const float16* b,                       \
      float16* y,                             \
      CUDAContext* ctx) {                     \
    if ((n & 1) == 0) {                       \
      _SimpleBinaryFunc<<<                    \
          CUDA_BLOCKS(n >> 1),                \
          CUDA_THREADS,                       \
          0,                                  \
          ctx->cuda_stream()>>>(              \
          n >> 1,                             \
          Functor<half2>(),                   \
          reinterpret_cast<const half2*>(a),  \
          reinterpret_cast<const half2*>(b),  \
          reinterpret_cast<half2*>(y));       \
    } else {                                  \
      _SimpleBinaryFunc<<<                    \
          CUDA_BLOCKS(n),                     \
          CUDA_THREADS,                       \
          0,                                  \
          ctx->cuda_stream()>>>(              \
          n,                                  \
          Functor<half>(),                    \
          reinterpret_cast<const half*>(a),   \
          reinterpret_cast<const half*>(b),   \
          reinterpret_cast<half*>(y));        \
    }                                         \
  }

DEFINE_BINARY_FUNC(Add, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, math::PowFunctor);
DEFINE_BINARY_FUNC(Minimum, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, math::MaxFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, Functor)     \
  template <>                                 \
  DRAGON_API void name<float16, CUDAContext>( \
      const int n,                            \
      const float16* a,                       \
      const float16* b,                       \
      bool* y,                                \
      CUDAContext* ctx) {                     \
    _SimpleBinaryFunc<<<                      \
        CUDA_BLOCKS(n),                       \
        CUDA_THREADS,                         \
        0,                                    \
        ctx->cuda_stream()>>>(                \
        n,                                    \
        Functor<half>(),                      \
        reinterpret_cast<const half*>(a),     \
        reinterpret_cast<const half*>(b),     \
        y);                                   \
  }

DEFINE_BINARY_FUNC(Equal, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, math::GreaterEqualFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_WHERE_FUNC(T)                                         \
  template <>                                                        \
  DRAGON_API void Where<T, CUDAContext>(                             \
      const int n,                                                   \
      const T* a,                                                    \
      const T* b,                                                    \
      const bool* c,                                                 \
      T* y,                                                          \
      CUDAContext* ctx) {                                            \
    _Where<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, a, b, c, y);                                              \
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

#endif // USE_CUDA
