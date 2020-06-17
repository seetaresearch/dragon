#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"
#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/utils.h"

namespace dragon {

namespace math {

namespace {

/*!
 * UnaryOp Wrappers
 */

#define DEFINE_UNARY_OPERATOR(name, func)              \
  template <typename T>                                \
  struct name##Op {                                    \
    inline __device__ T operator()(const T& x) const { \
      return func(x);                                  \
    }                                                  \
  }

DEFINE_UNARY_OPERATOR(Ceil, ceil);
DEFINE_UNARY_OPERATOR(Cos, cos);
DEFINE_UNARY_OPERATOR(Exp, exp);
DEFINE_UNARY_OPERATOR(Floor, floor);
DEFINE_UNARY_OPERATOR(Log, log);
DEFINE_UNARY_OPERATOR(Round, round);
DEFINE_UNARY_OPERATOR(Rsqrt, rsqrt);
DEFINE_UNARY_OPERATOR(Sin, sin);
DEFINE_UNARY_OPERATOR(Sqrt, sqrt);
#if __CUDA_ARCH__ >= 530
DEFINE_UNARY_OPERATOR(CeilHalf, hceil);
DEFINE_UNARY_OPERATOR(CeilHalf2, h2ceil);
DEFINE_UNARY_OPERATOR(CosHalf, hcos);
DEFINE_UNARY_OPERATOR(CosHalf2, h2cos);
DEFINE_UNARY_OPERATOR(ExpHalf, hexp);
DEFINE_UNARY_OPERATOR(ExpHalf2, h2exp);
DEFINE_UNARY_OPERATOR(FloorHalf, hfloor);
DEFINE_UNARY_OPERATOR(FloorHalf2, h2floor);
DEFINE_UNARY_OPERATOR(InvHalf, hrcp);
DEFINE_UNARY_OPERATOR(InvHalf2, h2rcp);
DEFINE_UNARY_OPERATOR(LogHalf, hlog);
DEFINE_UNARY_OPERATOR(LogHalf2, h2log);
DEFINE_UNARY_OPERATOR(RoundHalf, hrint);
DEFINE_UNARY_OPERATOR(RoundHalf2, h2rint);
DEFINE_UNARY_OPERATOR(RsqrtHalf, hrsqrt);
DEFINE_UNARY_OPERATOR(RsqrtHalf2, h2rsqrt);
DEFINE_UNARY_OPERATOR(SinHalf, hsin);
DEFINE_UNARY_OPERATOR(SinHalf2, h2sin);
DEFINE_UNARY_OPERATOR(SqrtHalf, hsqrt);
DEFINE_UNARY_OPERATOR(SqrtHalf2, h2sqrt);
#endif
#undef DEFINE_UNARY_OPERATOR

#define DEFINE_UNARY_OPERATOR(name, func)              \
  template <typename T>                                \
  struct name##Op {                                    \
    inline __device__ T operator()(const T& x) const { \
      return __float2half(func(__half2float(x)));      \
    }                                                  \
  }

#if __CUDA_ARCH__ < 530
DEFINE_UNARY_OPERATOR(CeilHalf, ceil);
DEFINE_UNARY_OPERATOR(CosHalf, cos);
DEFINE_UNARY_OPERATOR(ExpHalf, exp);
DEFINE_UNARY_OPERATOR(FloorHalf, floor);
DEFINE_UNARY_OPERATOR(InvHalf, __frcp_rn);
DEFINE_UNARY_OPERATOR(LogHalf, log);
DEFINE_UNARY_OPERATOR(RoundHalf, round);
DEFINE_UNARY_OPERATOR(RsqrtHalf, rsqrt);
DEFINE_UNARY_OPERATOR(SinHalf, sin);
DEFINE_UNARY_OPERATOR(SqrtHalf, sqrt);
#endif
#undef DEFINE_UNARY_OPERATOR

#define DEFINE_UNARY_OPERATOR(name, func)                 \
  template <typename T>                                   \
  struct name##Op {                                       \
    inline __device__ T operator()(const T& x) const {    \
      const float2 val = __half22float2(x);               \
      return __floats2half2_rn(func(val.x), func(val.y)); \
    }                                                     \
  }

#if __CUDA_ARCH__ < 530
DEFINE_UNARY_OPERATOR(CeilHalf2, ceil);
DEFINE_UNARY_OPERATOR(CosHalf2, cos);
DEFINE_UNARY_OPERATOR(ExpHalf2, exp);
DEFINE_UNARY_OPERATOR(FloorHalf2, floor);
DEFINE_UNARY_OPERATOR(InvHalf2, __frcp_rn);
DEFINE_UNARY_OPERATOR(LogHalf2, log);
DEFINE_UNARY_OPERATOR(RoundHalf2, round);
DEFINE_UNARY_OPERATOR(RsqrtHalf2, rsqrt);
DEFINE_UNARY_OPERATOR(SinHalf2, sin);
DEFINE_UNARY_OPERATOR(SqrtHalf2, sqrt);
#endif
#undef DEFINE_UNARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, TOut, expr)                      \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ TOut operator()(const T& a, const T& b) const { \
      return a expr b;                                                \
    }                                                                 \
  }

/*!
 * BinaryOp Wrappers
 */

DEFINE_BINARY_OPERATOR(Add, T, +);
DEFINE_BINARY_OPERATOR(Sub, T, -);
DEFINE_BINARY_OPERATOR(Mul, T, *);
DEFINE_BINARY_OPERATOR(Div, T, /);
DEFINE_BINARY_OPERATOR(Equal, bool, ==);
DEFINE_BINARY_OPERATOR(NotEqual, bool, !=);
DEFINE_BINARY_OPERATOR(Less, bool, <);
DEFINE_BINARY_OPERATOR(LessEqual, bool, <=);
DEFINE_BINARY_OPERATOR(Greater, bool, >);
DEFINE_BINARY_OPERATOR(GreaterEqual, bool, >=);
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, TOut, func)                      \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ TOut operator()(const T& a, const T& b) const { \
      return func(a, b);                                              \
    }                                                                 \
  }

DEFINE_BINARY_OPERATOR(Pow, T, pow);
DEFINE_BINARY_OPERATOR(Min, T, min);
DEFINE_BINARY_OPERATOR(Max, T, max);
#if __CUDA_ARCH__ >= 530
DEFINE_BINARY_OPERATOR(AddHalf, T, __hadd);
DEFINE_BINARY_OPERATOR(AddHalf2, T, __hadd2);
DEFINE_BINARY_OPERATOR(SubHalf, T, __hsub);
DEFINE_BINARY_OPERATOR(SubHalf2, T, __hsub2);
DEFINE_BINARY_OPERATOR(MulHalf, T, __hmul);
DEFINE_BINARY_OPERATOR(MulHalf2, T, __hmul2);
DEFINE_BINARY_OPERATOR(DivHalf, T, __hdiv);
DEFINE_BINARY_OPERATOR(EqualHalf, bool, __heq);
DEFINE_BINARY_OPERATOR(NotEqualHalf, bool, __hne);
DEFINE_BINARY_OPERATOR(LessHalf, bool, __hlt);
DEFINE_BINARY_OPERATOR(LessEqualHalf, bool, __hle);
DEFINE_BINARY_OPERATOR(GreaterHalf, bool, __hgt);
DEFINE_BINARY_OPERATOR(GreaterEqualHalf, bool, __hge);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, expr)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      return __float2half(__half2float(a) expr __half2float(b));   \
    }                                                              \
  }

#if __CUDA_ARCH__ < 530
DEFINE_BINARY_OPERATOR(AddHalf, +);
DEFINE_BINARY_OPERATOR(SubHalf, -);
DEFINE_BINARY_OPERATOR(MulHalf, *);
DEFINE_BINARY_OPERATOR(DivHalf, /);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, expr)                            \
  template <typename T>                                               \
  struct name##Op {                                                   \
    inline __device__ bool operator()(const T& a, const T& b) const { \
      return __half2float(a) expr __half2float(b);                    \
    }                                                                 \
  }

#if __CUDA_ARCH__ < 530
DEFINE_BINARY_OPERATOR(EqualHalf, ==);
DEFINE_BINARY_OPERATOR(NotEqualHalf, !=);
DEFINE_BINARY_OPERATOR(LessHalf, <);
DEFINE_BINARY_OPERATOR(LessEqualHalf, <=);
DEFINE_BINARY_OPERATOR(GreaterHalf, >);
DEFINE_BINARY_OPERATOR(GreaterEqualHalf, >=);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, func)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      return __float2half(func(__half2float(a), __half2float(b))); \
    }                                                              \
  }

DEFINE_BINARY_OPERATOR(PowHalf, pow);
DEFINE_BINARY_OPERATOR(MinHalf, min);
DEFINE_BINARY_OPERATOR(MaxHalf, max);
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, expr)                         \
  template <typename T>                                            \
  struct name##Op {                                                \
    inline __device__ T operator()(const T& a, const T& b) const { \
      const float2 v1 = __half22float2(a);                         \
      const float2 v2 = __half22float2(b);                         \
      return __floats2half2_rn(v1.x expr v2.x, v1.y expr v2.y);    \
    }                                                              \
  }

#if __CUDA_ARCH__ < 530
DEFINE_BINARY_OPERATOR(AddHalf2, +);
DEFINE_BINARY_OPERATOR(SubHalf2, -);
DEFINE_BINARY_OPERATOR(MulHalf2, *);
#endif
#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_OPERATOR(name, func)                          \
  template <typename T>                                             \
  struct name##Op {                                                 \
    inline __device__ T operator()(const T& a, const T& b) const {  \
      const float2 v1 = __half22float2(a);                          \
      const float2 v2 = __half22float2(b);                          \
      return __floats2half2_rn(func(v1.x, v2.x), func(v1.y, v2.y)); \
    }                                                               \
  }

DEFINE_BINARY_OPERATOR(PowHalf2, pow);
DEFINE_BINARY_OPERATOR(MinHalf2, min);
DEFINE_BINARY_OPERATOR(MaxHalf2, max);
#undef DEFINE_BINARY_OPERATOR

/*!
 * UnaryOp Kernels
 */

template <typename T, class Operator>
__global__ void
_SimpleUnaryFunc(const int nthreads, const Operator op, const T* x, T* y) {
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

template <>
__global__ void
_InvStd<half>(const int n, const half eps, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
    y[i] = hrsqrt(__hadd(x[i], eps));
#endif
  }
}

template <>
__global__ void
_InvStd<half2>(const int n, const half2 eps, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
    y[i] = h2rsqrt(__hadd2(x[i], eps));
#endif
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
#if __CUDA_ARCH__ >= 530
    y[i] = __float2half(pow(__half2float(x[i]), exponent));
#endif
  }
}

__global__ void
_Powx(const int n, const float exponent, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 530
    const float2 val = __half22float2(x[i]);
    y[i] = __floats2half2_rn(pow(val.x, exponent), pow(val.y, exponent));
#endif
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
    y[i] = utils::math::Sign(x[i]);
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
    y[i] = __float2half(utils::math::Sign(val));
  }
}

template <>
__global__ void _Sign<half2>(const int n, const half2* x, half2* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const float2 val = __half22float2(x[i]);
    y[i] =
        __floats2half2_rn(utils::math::Sign(val.x), utils::math::Sign(val.y));
  }
}

template <typename T>
__global__ void _Square(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = utils::math::Square(x[i]);
  }
}

template <typename T>
__global__ void _NotZero(const int nthreads, const T* x, bool* y) {
  const T kZero = T(0);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = x[i] != kZero ? true : false;
  }
}

template <>
__global__ void _NotZero<half>(const int nthreads, const half* x, bool* y) {
#if __CUDA_ARCH__ >= 530
  const half kZero = __float2half(0.f);
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    y[i] = __hne(x[i], kZero) ? true : false;
  }
#endif
}

template <typename T>
__global__ void _IsInf(const int n, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = utils::math::IsInf(x[i]);
  }
}

template <>
__global__ void _IsInf<half>(const int n, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = utils::math::IsInf(x[i]);
  }
}

template <typename T>
__global__ void _IsNaN(const int n, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = utils::math::IsNaN(x[i]);
  }
}

template <>
__global__ void _IsNaN<half>(const int n, const half* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = utils::math::IsNaN(x[i]);
  }
}

template <typename T>
__global__ void _ReplaceNaN(const int n, const T value, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = utils::math::IsNaN(__ldg(x + i)) ? value : __ldg(x + i);
#else
    y[i] = utils::math::IsNaN(x[i]) ? value : x[i];
#endif
  }
}

template <>
__global__ void
_ReplaceNaN<half>(const int n, const half value, const half* x, half* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 350
    y[i] = utils::math::IsNaN(__ldg(x + i)) ? value : __ldg(x + i);
#else
    y[i] = utils::math::IsNaN(x[i]) ? value : x[i];
#endif
  }
}

template <typename T, class Operator>
__global__ void
_Bias(const int n, const T beta, const Operator op, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = op(x[i], beta);
  }
}

/*!
 * BinaryOp Kernels
 */

template <typename TIn, typename TOut, class Operator>
__global__ void _SimpleBinaryFunc(
    const int n,
    const Operator op,
    const TIn* a,
    const TIn* b,
    TOut* y) {
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

#define DEFINE_UNARY_FUNC(name, T, Op)                                         \
  template <>                                                                  \
  DRAGON_API void name<T, CUDAContext>(                                        \
      const int n, const T* x, T* y, CUDAContext* ctx) {                       \
    _SimpleUnaryFunc<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, Op<T>(), x, y);                                                     \
  }

DEFINE_UNARY_FUNC(Ceil, float, CeilOp);
DEFINE_UNARY_FUNC(Ceil, double, CeilOp);
DEFINE_UNARY_FUNC(Cos, float, CosOp);
DEFINE_UNARY_FUNC(Cos, double, CosOp);
DEFINE_UNARY_FUNC(Exp, float, ExpOp);
DEFINE_UNARY_FUNC(Exp, double, ExpOp);
DEFINE_UNARY_FUNC(Floor, float, FloorOp);
DEFINE_UNARY_FUNC(Floor, double, FloorOp);
DEFINE_UNARY_FUNC(Log, float, LogOp);
DEFINE_UNARY_FUNC(Log, double, LogOp);
DEFINE_UNARY_FUNC(Round, float, RoundOp);
DEFINE_UNARY_FUNC(Round, double, RoundOp);
DEFINE_UNARY_FUNC(Rsqrt, float, RsqrtOp);
DEFINE_UNARY_FUNC(Rsqrt, double, RsqrtOp);
DEFINE_UNARY_FUNC(Sin, float, SinOp);
DEFINE_UNARY_FUNC(Sin, double, SinOp);
DEFINE_UNARY_FUNC(Sqrt, float, SqrtOp);
DEFINE_UNARY_FUNC(Sqrt, double, SqrtOp);
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

#define DEFINE_UNARY_FUNC(name, HalfOp, Half2Op)                     \
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
          Half2Op<half2>(),                                          \
          reinterpret_cast<const half2*>(x),                         \
          reinterpret_cast<half2*>(y));                              \
    } else {                                                         \
      _SimpleUnaryFunc<<<                                            \
          CUDA_BLOCKS(n),                                            \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          n,                                                         \
          HalfOp<half>(),                                            \
          reinterpret_cast<const half*>(x),                          \
          reinterpret_cast<half*>(y));                               \
    }                                                                \
  }

DEFINE_UNARY_FUNC(Ceil, CeilHalfOp, CeilHalf2Op);
DEFINE_UNARY_FUNC(Cos, CosHalfOp, CosHalf2Op);
DEFINE_UNARY_FUNC(Exp, ExpHalfOp, ExpHalf2Op);
DEFINE_UNARY_FUNC(Floor, FloorHalfOp, FloorHalf2Op);
DEFINE_UNARY_FUNC(Log, LogHalfOp, LogHalf2Op);
DEFINE_UNARY_FUNC(Inv, InvHalfOp, InvHalf2Op);
DEFINE_UNARY_FUNC(Round, RoundHalfOp, RoundHalf2Op);
DEFINE_UNARY_FUNC(Rsqrt, RsqrtHalfOp, RsqrtHalf2Op);
DEFINE_UNARY_FUNC(Sin, SinHalfOp, SinHalf2Op);
DEFINE_UNARY_FUNC(Sqrt, SqrtHalfOp, SqrtHalf2Op);
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
        n >> 1, cast::to<half2>(value), reinterpret_cast<half2*>(y));
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
        cast::to<half2>(eps),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _InvStd<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n,
        cast::to<half>(eps),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
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
      cast::to<half>(value),
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
        n, (T)beta, AddOp<T>(), x, y);                                     \
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
        cast::to<half2>(beta),
        AddHalf2Op<half2>(),
        reinterpret_cast<const half2*>(x),
        reinterpret_cast<half2*>(y));
  } else {
    _Bias<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
        n,
        cast::to<half>(beta),
        AddHalfOp<half>(),
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

#define DEFINE_BINARY_FUNC(name, TIn, TOut, Op)                             \
  template <>                                                               \
  DRAGON_API void name<TIn, CUDAContext>(                                   \
      const int n, const TIn* a, const TIn* b, TOut* y, CUDAContext* ctx) { \
    _SimpleBinaryFunc<<<                                                    \
        CUDA_BLOCKS(n),                                                     \
        CUDA_THREADS,                                                       \
        0,                                                                  \
        ctx->cuda_stream()>>>(n, Op<TIn>(), a, b, y);                       \
  }

DEFINE_BINARY_FUNC(Add, int8_t, int8_t, AddOp);
DEFINE_BINARY_FUNC(Add, uint8_t, uint8_t, AddOp);
DEFINE_BINARY_FUNC(Add, int, int, AddOp);
DEFINE_BINARY_FUNC(Add, int64_t, int64_t, AddOp);
DEFINE_BINARY_FUNC(Add, float, float, AddOp);
DEFINE_BINARY_FUNC(Add, double, double, AddOp);
DEFINE_BINARY_FUNC(Sub, int8_t, int8_t, SubOp);
DEFINE_BINARY_FUNC(Sub, uint8_t, uint8_t, SubOp);
DEFINE_BINARY_FUNC(Sub, int, int, SubOp);
DEFINE_BINARY_FUNC(Sub, int64_t, int64_t, SubOp);
DEFINE_BINARY_FUNC(Sub, float, float, SubOp);
DEFINE_BINARY_FUNC(Sub, double, double, SubOp);
DEFINE_BINARY_FUNC(Mul, int8_t, int8_t, MulOp);
DEFINE_BINARY_FUNC(Mul, uint8_t, uint8_t, MulOp);
DEFINE_BINARY_FUNC(Mul, int, int, MulOp);
DEFINE_BINARY_FUNC(Mul, int64_t, int64_t, MulOp);
DEFINE_BINARY_FUNC(Mul, float, float, MulOp);
DEFINE_BINARY_FUNC(Mul, double, double, MulOp);
DEFINE_BINARY_FUNC(Div, int8_t, int8_t, DivOp);
DEFINE_BINARY_FUNC(Div, uint8_t, uint8_t, DivOp);
DEFINE_BINARY_FUNC(Div, int, int, DivOp);
DEFINE_BINARY_FUNC(Div, int64_t, int64_t, DivOp);
DEFINE_BINARY_FUNC(Div, float, float, DivOp);
DEFINE_BINARY_FUNC(Div, double, double, DivOp);
DEFINE_BINARY_FUNC(Pow, float, float, PowOp);
DEFINE_BINARY_FUNC(Pow, double, double, PowOp);
DEFINE_BINARY_FUNC(Minimum, int8_t, int8_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, uint8_t, uint8_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, int, int, MinOp);
DEFINE_BINARY_FUNC(Minimum, int64_t, int64_t, MinOp);
DEFINE_BINARY_FUNC(Minimum, float, float, MinOp);
DEFINE_BINARY_FUNC(Minimum, double, double, MinOp);
DEFINE_BINARY_FUNC(Maximum, int8_t, int8_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, uint8_t, uint8_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, int, int, MaxOp);
DEFINE_BINARY_FUNC(Maximum, int64_t, int64_t, MaxOp);
DEFINE_BINARY_FUNC(Maximum, float, float, MaxOp);
DEFINE_BINARY_FUNC(Maximum, double, double, MaxOp);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, int, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, float, bool, EqualOp);
DEFINE_BINARY_FUNC(Equal, double, bool, EqualOp);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, int, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, float, bool, NotEqualOp);
DEFINE_BINARY_FUNC(NotEqual, double, bool, NotEqualOp);
DEFINE_BINARY_FUNC(Less, int8_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, int, bool, LessOp);
DEFINE_BINARY_FUNC(Less, int64_t, bool, LessOp);
DEFINE_BINARY_FUNC(Less, float, bool, LessOp);
DEFINE_BINARY_FUNC(Less, double, bool, LessOp);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, int, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, float, bool, LessEqualOp);
DEFINE_BINARY_FUNC(LessEqual, double, bool, LessEqualOp);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, int, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, float, bool, GreaterOp);
DEFINE_BINARY_FUNC(Greater, double, bool, GreaterOp);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, float, bool, GreaterEqualOp);
DEFINE_BINARY_FUNC(GreaterEqual, double, bool, GreaterEqualOp);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, T)                                   \
  template <>                                                         \
  DRAGON_API void name<T, CUDAContext>(                               \
      const int n, const T* a, const T* b, T* y, CUDAContext* ctx) {  \
    _##name<<<CUDA_BLOCKS(n), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        n, a, b, y);                                                  \
  }

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

#define DEFINE_BINARY_FUNC(name, HalfOp, Half2Op) \
  template <>                                     \
  DRAGON_API void name<float16, CUDAContext>(     \
      const int n,                                \
      const float16* a,                           \
      const float16* b,                           \
      float16* y,                                 \
      CUDAContext* ctx) {                         \
    if ((n & 1) == 0) {                           \
      _SimpleBinaryFunc<<<                        \
          CUDA_BLOCKS(n >> 1),                    \
          CUDA_THREADS,                           \
          0,                                      \
          ctx->cuda_stream()>>>(                  \
          n >> 1,                                 \
          Half2Op<half2>(),                       \
          reinterpret_cast<const half2*>(a),      \
          reinterpret_cast<const half2*>(b),      \
          reinterpret_cast<half2*>(y));           \
    } else {                                      \
      _SimpleBinaryFunc<<<                        \
          CUDA_BLOCKS(n),                         \
          CUDA_THREADS,                           \
          0,                                      \
          ctx->cuda_stream()>>>(                  \
          n,                                      \
          HalfOp<half>(),                         \
          reinterpret_cast<const half*>(a),       \
          reinterpret_cast<const half*>(b),       \
          reinterpret_cast<half*>(y));            \
    }                                             \
  }

DEFINE_BINARY_FUNC(Add, AddHalfOp, AddHalf2Op);
DEFINE_BINARY_FUNC(Sub, SubHalfOp, SubHalf2Op);
DEFINE_BINARY_FUNC(Mul, MulHalfOp, MulHalf2Op);
DEFINE_BINARY_FUNC(Pow, PowHalfOp, PowHalf2Op);
DEFINE_BINARY_FUNC(Minimum, MinHalfOp, MinHalf2Op);
DEFINE_BINARY_FUNC(Maximum, MaxHalfOp, MaxHalf2Op);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, TOut1, TOut2, HalfOp) \
  template <>                                          \
  DRAGON_API void name<float16, CUDAContext>(          \
      const int n,                                     \
      const float16* a,                                \
      const float16* b,                                \
      TOut1* y,                                        \
      CUDAContext* ctx) {                              \
    _SimpleBinaryFunc<<<                               \
        CUDA_BLOCKS(n),                                \
        CUDA_THREADS,                                  \
        0,                                             \
        ctx->cuda_stream()>>>(                         \
        n,                                             \
        HalfOp<half>(),                                \
        reinterpret_cast<const half*>(a),              \
        reinterpret_cast<const half*>(b),              \
        reinterpret_cast<TOut2*>(y));                  \
  }

DEFINE_BINARY_FUNC(Div, float16, half, DivHalfOp);
DEFINE_BINARY_FUNC(Equal, bool, bool, EqualHalfOp);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, NotEqualHalfOp);
DEFINE_BINARY_FUNC(Less, bool, bool, LessHalfOp);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, LessEqualHalfOp);
DEFINE_BINARY_FUNC(Greater, bool, bool, GreaterHalfOp);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, GreaterEqualHalfOp);
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
