#include "dragon/utils/math/elementwise.h"
#include "dragon/utils/math/functional.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace math {

namespace {

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
__global__ void _Set(const int N, const T value, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = value;
  }
}

template <typename T, typename AccT>
__global__ void _NotZero(const int N, const T* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<AccT>(x[i]) != AccT(0) ? true : false;
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
__global__ void _NaNToNum(const int N, const T nan, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = math::utils::IsNaN(math::utils::LDG(x + i))
        ? nan
        : math::utils::LDG(x + i);
  }
}

template <typename T>
__global__ void _NaNToNum(
    const int N,
    const T nan,
    const T pos_inf,
    const T neg_inf,
    const T* x,
    T* y) {
  const T kZero = convert::To<T>(0.f);
  const auto gt = math::GreaterFunctor<T>();
  CUDA_1D_KERNEL_LOOP(i, N) {
    if (math::utils::IsNaN(math::utils::LDG(x + i))) {
      y[i] = nan;
    } else if (math::utils::IsInf(math::utils::LDG(x + i))) {
      y[i] = gt(math::utils::LDG(x + i), kZero) ? pos_inf : neg_inf;
    } else {
      y[i] = math::utils::LDG(x + i);
    }
  }
}

template <typename T, class Functor>
__global__ void
_Bias(const int N, const T beta, const Functor op, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = op(x[i], beta);
  }
}

template <typename T, typename AccT>
__global__ void _ApplyMask(
    const int N,
    const AccT alpha,
    const uint8_t* mask,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(convert::To<AccT>(x[i]) * (alpha * AccT(mask[i])));
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
        Functor<math::Traits<InputT>::scalar_type>(),                          \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(x),         \
        reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));             \
  }

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

#define DEFINE_UNARY_FUNC(name, InputT, OutputT, Functor)                 \
  template <>                                                             \
  DRAGON_API void name<InputT, CUDAContext>(                              \
      const int N, const InputT* x, OutputT* y, CUDAContext* ctx) {       \
    if ((N & 1) == 0 && math::Traits<InputT>::HasPack2()) {               \
      _SimpleUnaryFunc<<<                                                 \
          CUDA_BLOCKS(N >> 1),                                            \
          CUDA_THREADS,                                                   \
          0,                                                              \
          ctx->cuda_stream()>>>(                                          \
          N >> 1,                                                         \
          Functor<math::Traits<InputT>::scalar2_type>(),                  \
          reinterpret_cast<const math::Traits<InputT>::scalar2_type*>(x), \
          reinterpret_cast<math::Traits<OutputT>::scalar2_type*>(y));     \
    } else {                                                              \
      _SimpleUnaryFunc<<<                                                 \
          CUDA_BLOCKS(N),                                                 \
          CUDA_THREADS,                                                   \
          0,                                                              \
          ctx->cuda_stream()>>>(                                          \
          N,                                                              \
          Functor<math::Traits<InputT>::scalar_type>(),                   \
          reinterpret_cast<const math::Traits<InputT>::scalar_type*>(x),  \
          reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));      \
    }                                                                     \
  }

DEFINE_UNARY_FUNC(Neg, int8_t, int8_t, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, int, int, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, int64_t, int64_t, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, float16, float16, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, bfloat16, bfloat16, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, float, float, math::NegateFunctor);
DEFINE_UNARY_FUNC(Neg, double, double, math::NegateFunctor);
DEFINE_UNARY_FUNC(Abs, int8_t, int8_t, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, int, int, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, int64_t, int64_t, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, float16, float16, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, bfloat16, bfloat16, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, float, float, math::AbsFunctor);
DEFINE_UNARY_FUNC(Abs, double, double, math::AbsFunctor);
DEFINE_UNARY_FUNC(Square, uint8_t, uint8_t, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, int8_t, int8_t, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, int, int, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, int64_t, int64_t, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, float16, float16, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, bfloat16, bfloat16, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, float, float, math::SqrFunctor);
DEFINE_UNARY_FUNC(Square, double, double, math::SqrFunctor);
DEFINE_UNARY_FUNC(Sign, uint8_t, uint8_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int8_t, int8_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int, int, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, int64_t, int64_t, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, float16, float16, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, bfloat16, bfloat16, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, float, float, math::SignFunctor);
DEFINE_UNARY_FUNC(Sign, double, double, math::SignFunctor);
DEFINE_UNARY_FUNC(Ceil, float16, float16, math::CeilFunctor);
DEFINE_UNARY_FUNC(Ceil, bfloat16, bfloat16, math::CeilFunctor);
DEFINE_UNARY_FUNC(Ceil, float, float, math::CeilFunctor);
DEFINE_UNARY_FUNC(Ceil, double, double, math::CeilFunctor);
DEFINE_UNARY_FUNC(Floor, float16, float16, math::FloorFunctor);
DEFINE_UNARY_FUNC(Floor, bfloat16, bfloat16, math::FloorFunctor);
DEFINE_UNARY_FUNC(Floor, float, float, math::FloorFunctor);
DEFINE_UNARY_FUNC(Floor, double, double, math::FloorFunctor);
DEFINE_UNARY_FUNC(Round, float16, float16, math::RoundFunctor);
DEFINE_UNARY_FUNC(Round, bfloat16, bfloat16, math::RoundFunctor);
DEFINE_UNARY_FUNC(Round, float, float, math::RoundFunctor);
DEFINE_UNARY_FUNC(Round, double, double, math::RoundFunctor);
DEFINE_UNARY_FUNC(Exp, float16, float16, math::ExpFunctor);
DEFINE_UNARY_FUNC(Exp, bfloat16, bfloat16, math::ExpFunctor);
DEFINE_UNARY_FUNC(Exp, float, float, math::ExpFunctor);
DEFINE_UNARY_FUNC(Exp, double, double, math::ExpFunctor);
DEFINE_UNARY_FUNC(Log, float16, float16, math::LogFunctor);
DEFINE_UNARY_FUNC(Log, bfloat16, bfloat16, math::LogFunctor);
DEFINE_UNARY_FUNC(Log, float, float, math::LogFunctor);
DEFINE_UNARY_FUNC(Log, double, double, math::LogFunctor);
DEFINE_UNARY_FUNC(Inv, float16, float16, math::InvFunctor);
DEFINE_UNARY_FUNC(Inv, bfloat16, bfloat16, math::InvFunctor);
DEFINE_UNARY_FUNC(Inv, float, float, math::InvFunctor);
DEFINE_UNARY_FUNC(Inv, double, double, math::InvFunctor);
DEFINE_UNARY_FUNC(Sqrt, float16, float16, math::SqrtFunctor);
DEFINE_UNARY_FUNC(Sqrt, bfloat16, bfloat16, math::SqrtFunctor);
DEFINE_UNARY_FUNC(Sqrt, float, float, math::SqrtFunctor);
DEFINE_UNARY_FUNC(Sqrt, double, double, math::SqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, float16, float16, math::RsqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, bfloat16, bfloat16, math::RsqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, float, float, math::RsqrtFunctor);
DEFINE_UNARY_FUNC(Rsqrt, double, double, math::RsqrtFunctor);
DEFINE_UNARY_FUNC(Sin, float16, float16, math::SinFunctor);
DEFINE_UNARY_FUNC(Sin, bfloat16, bfloat16, math::SinFunctor);
DEFINE_UNARY_FUNC(Sin, float, float, math::SinFunctor);
DEFINE_UNARY_FUNC(Sin, double, double, math::SinFunctor);
DEFINE_UNARY_FUNC(Cos, float16, float16, math::CosFunctor);
DEFINE_UNARY_FUNC(Cos, bfloat16, bfloat16, math::CosFunctor);
DEFINE_UNARY_FUNC(Cos, float, float, math::CosFunctor);
DEFINE_UNARY_FUNC(Cos, double, double, math::CosFunctor);
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

DEFINE_SET_FUNC(bool);
DEFINE_SET_FUNC(uint8_t);
DEFINE_SET_FUNC(int8_t);
DEFINE_SET_FUNC(int);
DEFINE_SET_FUNC(int64_t);
DEFINE_SET_FUNC(float);
DEFINE_SET_FUNC(double);
#undef DEFINE_SET_FUNC

#define DEFINE_SET_FUNC(T)                                                  \
  template <>                                                               \
  DRAGON_API void Set<T, CUDAContext>(                                      \
      const int N, const T value, T* y, CUDAContext* ctx) {                 \
    if (value.x == (unsigned short)0) {                                     \
      CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(T) * N, ctx->cuda_stream())); \
      return;                                                               \
    }                                                                       \
    if ((N & 1) == 0) {                                                     \
      _Set<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
          N >> 1,                                                           \
          convert::To<math::Traits<T>::scalar2_type>(value),                \
          reinterpret_cast<math::Traits<T>::scalar2_type*>(y));             \
    } else {                                                                \
      _Set<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(        \
          N, value, y);                                                     \
    }                                                                       \
  }

DEFINE_SET_FUNC(float16);
DEFINE_SET_FUNC(bfloat16);
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
        N, T(exponent), x, y);                                                 \
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

#define DEFINE_NOT_ZERO_FUNC(T)                                               \
  template <>                                                                 \
  DRAGON_API void NotZero<T, CUDAContext>(                                    \
      const int N, const T* x, bool* y, CUDAContext* ctx) {                   \
    _NotZero<math::Traits<T>::scalar_type, math::Traits<T>::accumulator_type> \
        <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(            \
            N, reinterpret_cast<const math::Traits<T>::scalar_type*>(x), y);  \
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

#define DEFINE_IS_FUNC(name, T)                                          \
  template <>                                                            \
  DRAGON_API void name<T, CUDAContext>(                                  \
      const int N, const T* x, bool* y, CUDAContext* ctx) {              \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
        N, reinterpret_cast<const math::Traits<T>::scalar_type*>(x), y); \
  }

DEFINE_IS_FUNC(IsInf, float16);
DEFINE_IS_FUNC(IsInf, bfloat16);
DEFINE_IS_FUNC(IsInf, float);
DEFINE_IS_FUNC(IsInf, double);
DEFINE_IS_FUNC(IsNaN, float16);
DEFINE_IS_FUNC(IsNaN, bfloat16);
DEFINE_IS_FUNC(IsNaN, float);
DEFINE_IS_FUNC(IsNaN, double);
DEFINE_IS_FUNC(IsFinite, float16);
DEFINE_IS_FUNC(IsFinite, bfloat16);
DEFINE_IS_FUNC(IsFinite, float);
DEFINE_IS_FUNC(IsFinite, double);
#undef DEFINE_IS_FUNC

#define DEFINE_NAN_TO_NUM_FUNC(T)                                           \
  template <>                                                               \
  DRAGON_API void NaNToNum<T, CUDAContext>(                                 \
      const int N, const float nan, const T* x, T* y, CUDAContext* ctx) {   \
    _NaNToNum<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
        N,                                                                  \
        convert::To<math::Traits<T>::scalar_type>(nan),                     \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),           \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));                \
  }                                                                         \
  template <>                                                               \
  DRAGON_API void NaNToNum<T, CUDAContext>(                                 \
      const int N,                                                          \
      const float nan,                                                      \
      const float pos_inf,                                                  \
      const float neg_inf,                                                  \
      const T* x,                                                           \
      T* y,                                                                 \
      CUDAContext* ctx) {                                                   \
    auto pos_inf_arg = std::min(pos_inf, float(math::Traits<T>::Max()));    \
    auto neg_inf_arg = std::max(neg_inf, float(math::Traits<T>::Lowest())); \
    _NaNToNum<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(     \
        N,                                                                  \
        convert::To<math::Traits<T>::scalar_type>(nan),                     \
        convert::To<math::Traits<T>::scalar_type>(pos_inf_arg),             \
        convert::To<math::Traits<T>::scalar_type>(neg_inf_arg),             \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),           \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));                \
  }

DEFINE_NAN_TO_NUM_FUNC(float16);
DEFINE_NAN_TO_NUM_FUNC(bfloat16);
DEFINE_NAN_TO_NUM_FUNC(float);
DEFINE_NAN_TO_NUM_FUNC(double);
#undef DEFINE_NAN_TO_NUM_FUNC

#define DEFINE_BIAS_FUNC(T)                                                \
  template <>                                                              \
  DRAGON_API void Bias<T, CUDAContext>(                                    \
      const int N, const float beta, const T* x, T* y, CUDAContext* ctx) { \
    if (beta == 0.f && x == y) return;                                     \
    if ((N & 1) == 0 && math::Traits<T>::HasPack2()) {                     \
      _Bias<<<CUDA_BLOCKS(N >> 1), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          N >> 1,                                                          \
          convert::To<math::Traits<T>::scalar2_type>(beta),                \
          math::PlusFunctor<math::Traits<T>::scalar2_type>(),              \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(x),       \
          reinterpret_cast<math::Traits<T>::scalar2_type*>(y));            \
    } else {                                                               \
      _Bias<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(      \
          N,                                                               \
          convert::To<math::Traits<T>::scalar_type>(beta),                 \
          math::PlusFunctor<math::Traits<T>::scalar_type>(),               \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
          reinterpret_cast<math::Traits<T>::scalar_type*>(y));             \
    }                                                                      \
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

#define DEFINE_APPLY_MASK_FUNC(T)                                        \
  template <>                                                            \
  DRAGON_API void ApplyMask<T, CUDAContext>(                             \
      const int N,                                                       \
      const float alpha,                                                 \
      const uint8_t* mask,                                               \
      const T* x,                                                        \
      T* y,                                                              \
      CUDAContext* ctx) {                                                \
    using AccT = math::Traits<T>::accumulator_type;                      \
    _ApplyMask<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                               \
        convert::To<AccT>(alpha),                                        \
        mask,                                                            \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x),        \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));             \
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

#define DEFINE_BINARY_FUNC(name, T, Functor)                         \
  template <>                                                        \
  DRAGON_API void name<T, CUDAContext>(                              \
      const int N, const T* a, const T* b, T* y, CUDAContext* ctx) { \
    if ((N & 1) == 0 && math::Traits<T>::HasPack2()) {               \
      _SimpleBinaryFunc<<<                                           \
          CUDA_BLOCKS(N >> 1),                                       \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N >> 1,                                                    \
          Functor<math::Traits<T>::scalar2_type>(),                  \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(a), \
          reinterpret_cast<const math::Traits<T>::scalar2_type*>(b), \
          reinterpret_cast<math::Traits<T>::scalar2_type*>(y));      \
    } else {                                                         \
      _SimpleBinaryFunc<<<                                           \
          CUDA_BLOCKS(N),                                            \
          CUDA_THREADS,                                              \
          0,                                                         \
          ctx->cuda_stream()>>>(                                     \
          N,                                                         \
          Functor<math::Traits<T>::scalar_type>(),                   \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(a),  \
          reinterpret_cast<const math::Traits<T>::scalar_type*>(b),  \
          reinterpret_cast<math::Traits<T>::scalar_type*>(y));       \
    }                                                                \
  }

DEFINE_BINARY_FUNC(Add, uint8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int8_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, int64_t, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float16, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, bfloat16, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, float, math::PlusFunctor);
DEFINE_BINARY_FUNC(Add, double, math::PlusFunctor);
DEFINE_BINARY_FUNC(Sub, uint8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int8_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, int64_t, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float16, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, bfloat16, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, float, math::MinusFunctor);
DEFINE_BINARY_FUNC(Sub, double, math::MinusFunctor);
DEFINE_BINARY_FUNC(Mul, uint8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int8_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, int64_t, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float16, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, bfloat16, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, float, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Mul, double, math::MultipliesFunctor);
DEFINE_BINARY_FUNC(Div, uint8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int8_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, int64_t, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float16, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, bfloat16, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, float, math::DividesFunctor);
DEFINE_BINARY_FUNC(Div, double, math::DividesFunctor);
DEFINE_BINARY_FUNC(Pow, float16, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, bfloat16, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, float, math::PowFunctor);
DEFINE_BINARY_FUNC(Pow, double, math::PowFunctor);
DEFINE_BINARY_FUNC(Atan2, float16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, bfloat16, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, float, math::Atan2Functor);
DEFINE_BINARY_FUNC(Atan2, double, math::Atan2Functor);
DEFINE_BINARY_FUNC(Minimum, uint8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int8_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, int64_t, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float16, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, bfloat16, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, float, math::MinFunctor);
DEFINE_BINARY_FUNC(Minimum, double, math::MinFunctor);
DEFINE_BINARY_FUNC(Maximum, uint8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int8_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, int64_t, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float16, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, bfloat16, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, float, math::MaxFunctor);
DEFINE_BINARY_FUNC(Maximum, double, math::MaxFunctor);
#undef DEFINE_BINARY_FUNC

#define DEFINE_BINARY_FUNC(name, InputT, OutputT, Functor)             \
  template <>                                                          \
  DRAGON_API void name<InputT, CUDAContext>(                           \
      const int N,                                                     \
      const InputT* a,                                                 \
      const InputT* b,                                                 \
      OutputT* y,                                                      \
      CUDAContext* ctx) {                                              \
    _SimpleBinaryFunc<<<                                               \
        CUDA_BLOCKS(N),                                                \
        CUDA_THREADS,                                                  \
        0,                                                             \
        ctx->cuda_stream()>>>(                                         \
        N,                                                             \
        Functor<math::Traits<InputT>::scalar_type>(),                  \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(a), \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(b), \
        reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));     \
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
DEFINE_BINARY_FUNC(Equal, bool, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, uint8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int8_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, int64_t, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float16, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, bfloat16, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, float, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(Equal, double, bool, math::EqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, bool, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, uint8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int8_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, int64_t, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float16, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, bfloat16, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, float, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(NotEqual, double, bool, math::NotEqualFunctor);
DEFINE_BINARY_FUNC(Less, bool, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, uint8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int8_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, int64_t, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float16, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, bfloat16, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, float, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(Less, double, bool, math::LessFunctor);
DEFINE_BINARY_FUNC(LessEqual, bool, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, uint8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int8_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, int64_t, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float16, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, bfloat16, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, float, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(LessEqual, double, bool, math::LessEqualFunctor);
DEFINE_BINARY_FUNC(Greater, bool, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, uint8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int8_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, int64_t, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float16, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, bfloat16, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, float, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(Greater, double, bool, math::GreaterFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, bool, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, uint8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int8_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, int64_t, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, float16, bool, math::GreaterEqualFunctor);
DEFINE_BINARY_FUNC(GreaterEqual, bfloat16, bool, math::GreaterEqualFunctor);
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
DEFINE_WHERE_FUNC(bfloat16);
DEFINE_WHERE_FUNC(float);
DEFINE_WHERE_FUNC(double);
#undef DEFINE_WHERE_FUNC

} // namespace math

} // namespace dragon
