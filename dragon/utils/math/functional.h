/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_MATH_FUNCTIONAL_H_
#define DRAGON_UTILS_MATH_FUNCTIONAL_H_

#include "dragon/core/types.h"
#include "dragon/utils/math/utils.h"

#if defined(__CUDACC__)
#define HOSTDEVICE_DECL inline __host__ __device__
#else
#define HOSTDEVICE_DECL inline
#endif

// clang-format off
#define CUDA_UNARY_FP16_FALLBACK_EXPR(Func, Arg) \
  __float2half(Func(__half2float(Arg)))

#define CUDA_UNARY_BF16_FALLBACK_EXPR(Func, Arg) \
  __float2bfloat16(Func(__bfloat162float(Arg)))

#define CUDA_UNARY_FP162_FALLBACK_EXPR(Func, Arg) \
  __floats2half2_rn(Func(__half2float(Arg.x)), Func(__half2float(Arg.y)))

#define CUDA_UNARY_BF162_FALLBACK_EXPR(Func, Arg) \
  __floats2bfloat162_rn(Func(__bfloat162float(Arg.x)), Func(__bfloat162float(Arg.y)))
// clang-format on

#define DEFINE_HOST_BINARY_HALF_FUNCTOR(name, InputT, OutputT, Expr)        \
  template <>                                                               \
  struct name##Functor<InputT> {                                            \
    inline OutputT operator()(const InputT& lhs, const InputT& rhs) const { \
      return convert::To<OutputT>(convert::To<float>(lhs)                   \
                                      Expr convert::To<float>(rhs));        \
    }                                                                       \
  };

namespace dragon {

namespace math {

/*
 * Arithmetic Functors
 */

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE_DECL T operator()(const T& val) const {
    return val;
  }
};

template <typename T>
struct NegateFunctor {
  HOSTDEVICE_DECL T operator()(const T& val) const {
    return -val;
  }
};

#if defined(__CUDACC__)
template <>
struct NegateFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return __hneg(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(-, val);
#endif
  }
};

template <>
struct NegateFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return __hneg2(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(-, val);
#endif
  }
};

template <>
struct NegateFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return __hneg(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(-, val);
#endif
  }
};

template <>
struct NegateFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return __hneg2(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(-, val);
#endif
  }
};
#endif

template <typename T>
struct AbsFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return abs(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::abs(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct AbsFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return __habs(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(fabsf, val);
#endif
  }
};

template <>
struct AbsFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return __habs2(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(fabsf, val);
#endif
  }
};

template <>
struct AbsFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return __habs(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(fabsf, val);
#endif
  }
};

template <>
struct AbsFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return __habs2(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(fabsf, val);
#endif
  }
};
#endif

template <typename T>
struct SqrFunctor {
  HOSTDEVICE_DECL T operator()(const T& val) const {
    return val * val;
  }
};

#if defined(__CUDACC__)
template <>
struct SqrFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return __hmul(val, val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(math::utils::Sqr, val);
#endif
  }
};

template <>
struct SqrFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return __hmul2(val, val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(math::utils::Sqr, val);
#endif
  }
};

template <>
struct SqrFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return __hmul(val, val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(math::utils::Sqr, val);
#endif
  }
};

template <>
struct SqrFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return __hmul2(val, val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(math::utils::Sqr, val);
#endif
  }
};
#endif

template <typename T>
struct SignFunctor {
  HOSTDEVICE_DECL T operator()(const T& val) const {
    return (val > T(0)) - (val < T(0));
  }
};

template <>
struct SignFunctor<uint8_t> {
  HOSTDEVICE_DECL uint8_t operator()(const uint8_t& val) const {
    return val > 0 ? uint8_t(1) : uint8_t(0);
  }
};

template <>
struct SignFunctor<float16> {
  inline float16 operator()(const float16& val) const {
    const float valf = convert::To<float>(val);
    return convert::To<float16>(float((valf > 0.f) - (valf < 0.f)));
  }
};

template <>
struct SignFunctor<bfloat16> {
  inline bfloat16 operator()(const bfloat16& val) const {
    const float valf = convert::To<float>(val);
    return convert::To<bfloat16>(float((valf > 0.f) - (valf < 0.f)));
  }
};

#if defined(__CUDACC__)
template <>
struct SignFunctor<half> {
  inline __device__ half operator()(const half& val) const {
    return CUDA_UNARY_FP16_FALLBACK_EXPR(math::utils::Sign, val);
  }
};

template <>
struct SignFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
    return CUDA_UNARY_FP162_FALLBACK_EXPR(math::utils::Sign, val);
  }
};

template <>
struct SignFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
    return CUDA_UNARY_BF16_FALLBACK_EXPR(math::utils::Sign, val);
  }
};

template <>
struct SignFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
    return CUDA_UNARY_BF162_FALLBACK_EXPR(math::utils::Sign, val);
  }
};
#endif

template <typename T>
struct CeilFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return ceil(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::ceil(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct CeilFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hceil(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(ceilf, val);
#endif
  }
};

template <>
struct CeilFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2ceil(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(ceilf, val);
#endif
  }
};

template <>
struct CeilFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hceil(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(ceilf, val);
#endif
  }
};

template <>
struct CeilFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2ceil(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(ceilf, val);
#endif
  }
};
#endif

template <typename T>
struct FloorFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return floor(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::floor(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct FloorFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hfloor(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(floorf, val);
#endif
  }
};

template <>
struct FloorFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2floor(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(floorf, val);
#endif
  }
};

template <>
struct FloorFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hfloor(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(floorf, val);
#endif
  }
};

template <>
struct FloorFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2floor(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(floorf, val);
#endif
  }
};
#endif

template <typename T>
struct RoundFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return round(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::round(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct RoundFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hrint(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(roundf, val);
#endif
  }
};

template <>
struct RoundFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2rint(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(roundf, val);
#endif
  }
};

template <>
struct RoundFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hrint(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(roundf, val);
#endif
  }
};

template <>
struct RoundFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2rint(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(roundf, val);
#endif
  }
};
#endif

template <typename T>
struct ExpFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return exp(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::exp(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct ExpFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hexp(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(expf, val);
#endif
  }
};

template <>
struct ExpFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2exp(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(expf, val);
#endif
  }
};

template <>
struct ExpFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hexp(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(expf, val);
#endif
  }
};

template <>
struct ExpFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2exp(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(expf, val);
#endif
  }
};
#endif

template <typename T>
struct LogFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return log(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::log(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct LogFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hlog(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(logf, val);
#endif
  }
};

template <>
struct LogFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2log(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(logf, val);
#endif
  }
};

template <>
struct LogFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hlog(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(logf, val);
#endif
  }
};

template <>
struct LogFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2log(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(logf, val);
#endif
  }
};
#endif

template <typename T>
struct InvFunctor {
  HOSTDEVICE_DECL T operator()(const T& val) const {
    return T(1) / val;
  }
};

#if defined(__CUDACC__)
template <>
struct InvFunctor<float> {
  inline __device__ float operator()(const float& val) const {
    return __frcp_rn(val);
  }
};

template <>
struct InvFunctor<double> {
  inline __device__ double operator()(const double& val) const {
    return __drcp_rn(val);
  }
};

template <>
struct InvFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hrcp(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(__frcp_rn, val);
#endif
  }
};

template <>
struct InvFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2rcp(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(__frcp_rn, val);
#endif
  }
};

template <>
struct InvFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hrcp(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(__frcp_rn, val);
#endif
  }
};

template <>
struct InvFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2rcp(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(__frcp_rn, val);
#endif
  }
};
#endif

template <typename T>
struct SqrtFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return sqrt(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::sqrt(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct SqrtFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hsqrt(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(sqrtf, val);
#endif
  }
};

template <>
struct SqrtFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2sqrt(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(sqrtf, val);
#endif
  }
};

template <>
struct SqrtFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hsqrt(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(sqrtf, val);
#endif
  }
};

template <>
struct SqrtFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2sqrt(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(sqrtf, val);
#endif
  }
};
#endif

template <typename T>
struct RsqrtFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return rsqrt(val);
  }
#else
  inline T operator()(const T& val) const {
    return T(1) / std::sqrt(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct RsqrtFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hrsqrt(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(rsqrtf, val);
#endif
  }
};

template <>
struct RsqrtFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2rsqrt(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(rsqrtf, val);
#endif
  }
};

template <>
struct RsqrtFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hrsqrt(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(rsqrtf, val);
#endif
  }
};

template <>
struct RsqrtFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2rsqrt(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(rsqrtf, val);
#endif
  }
};
#endif

template <typename T>
struct SinFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return sin(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::sin(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct SinFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hsin(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(sinf, val);
#endif
  }
};

template <>
struct SinFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2sin(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(sinf, val);
#endif
  }
};

template <>
struct SinFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hsin(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(sinf, val);
#endif
  }
};

template <>
struct SinFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2sin(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(sinf, val);
#endif
  }
};
#endif

template <typename T>
struct CosFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& val) const {
    return cos(val);
  }
#else
  inline T operator()(const T& val) const {
    return std::cos(val);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct CosFunctor<half> {
  inline __device__ half operator()(const half& val) const {
#if __CUDA_ARCH__ >= 530
    return hcos(val);
#else
    return CUDA_UNARY_FP16_FALLBACK_EXPR(cosf, val);
#endif
  }
};

template <>
struct CosFunctor<half2> {
  inline __device__ half2 operator()(const half2& val) const {
#if __CUDA_ARCH__ >= 530
    return h2cos(val);
#else
    return CUDA_UNARY_FP162_FALLBACK_EXPR(cosf, val);
#endif
  }
};

template <>
struct CosFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(const nv_bfloat16& val) const {
#if __CUDA_ARCH__ >= 800
    return hcos(val);
#else
    return CUDA_UNARY_BF16_FALLBACK_EXPR(cosf, val);
#endif
  }
};

template <>
struct CosFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(const nv_bfloat162& val) const {
#if __CUDA_ARCH__ >= 800
    return h2cos(val);
#else
    return CUDA_UNARY_BF162_FALLBACK_EXPR(cosf, val);
#endif
  }
};
#endif

template <typename T>
struct MaxFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? rhs : lhs;
  }
};

template <>
struct MaxFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? rhs : lhs;
  }
};

template <>
struct MaxFunctor<bfloat16> {
  inline bfloat16 operator()(const bfloat16& lhs, const bfloat16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? rhs : lhs;
  }
};

#if defined(__CUDACC__)
template <>
struct MaxFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(lhs, rhs) ? rhs : lhs;
#else
    return __half2float(lhs) < __half2float(rhs) ? rhs : lhs;
#endif
  }
};

template <>
struct MaxFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    half2 ret;
#if __CUDA_ARCH__ >= 530
    ret.x = __hlt(lhs.x, rhs.x) ? rhs.x : lhs.x;
    ret.y = __hlt(lhs.y, rhs.y) ? rhs.y : lhs.y;
#else
    ret.x = __half2float(lhs.x) < __half2float(rhs.x) ? rhs.x : lhs.x;
    ret.y = __half2float(lhs.y) < __half2float(rhs.y) ? rhs.y : lhs.y;
#endif
    return ret;
  }
};

template <>
struct MaxFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hlt(lhs, rhs) ? rhs : lhs;
#else
    return __bfloat162float(lhs) < __bfloat162float(rhs) ? rhs : lhs;
#endif
  }
};

template <>
struct MaxFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
    nv_bfloat162 ret;
#if __CUDA_ARCH__ >= 800
    ret.x = __hlt(lhs.x, rhs.x) ? rhs.x : lhs.x;
    ret.y = __hlt(lhs.y, rhs.y) ? rhs.y : lhs.y;
#else
    ret.x = __bfloat162float(lhs.x) < __bfloat162float(rhs.x) ? rhs.x : lhs.x;
    ret.y = __bfloat162float(lhs.y) < __bfloat162float(rhs.y) ? rhs.y : lhs.y;
#endif
    return ret;
  }
};
#endif

template <typename T>
struct MinFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
};

template <>
struct MinFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? lhs : rhs;
  }
};

template <>
struct MinFunctor<bfloat16> {
  inline bfloat16 operator()(const bfloat16& lhs, const bfloat16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? lhs : rhs;
  }
};

#if defined(__CUDACC__)
template <>
struct MinFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(lhs, rhs) ? lhs : rhs;
#else
    return __half2float(lhs) < __half2float(rhs) ? lhs : rhs;
#endif
  }
};

template <>
struct MinFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    half2 ret;
#if __CUDA_ARCH__ >= 530
    ret.x = __hlt(lhs.x, rhs.x) ? lhs.x : rhs.x;
    ret.y = __hlt(lhs.y, rhs.y) ? lhs.y : rhs.y;
#else
    ret.x = __half2float(lhs.x) < __half2float(rhs.x) ? lhs.x : rhs.x;
    ret.y = __half2float(lhs.y) < __half2float(rhs.y) ? lhs.y : rhs.y;
#endif
    return ret;
  }
};

template <>
struct MinFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hlt(lhs, rhs) ? lhs : rhs;
#else
    return __bfloat162float(lhs) < __bfloat162float(rhs) ? lhs : rhs;
#endif
  }
};

template <>
struct MinFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
    nv_bfloat162 ret;
#if __CUDA_ARCH__ >= 800
    ret.x = __hlt(lhs.x, rhs.x) ? lhs.x : rhs.x;
    ret.y = __hlt(lhs.y, rhs.y) ? lhs.y : rhs.y;
#else
    ret.x = __bfloat162float(lhs.x) < __bfloat162float(rhs.x) ? lhs.x : rhs.x;
    ret.y = __bfloat162float(lhs.y) < __bfloat162float(rhs.y) ? lhs.y : rhs.y;
#endif
    return ret;
  }
};
#endif

template <typename T>
struct ClampFunctor {
#if defined(__CUDACC__)
  inline __device__ T
  operator()(const T& val, const T& low, const T& high) const {
    return max(low, min(val, high));
  }
#else
  inline T operator()(const T& val, const T& low, const T& high) const {
    return std::max(low, std::min(val, high));
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct ClampFunctor<half> {
  inline __device__ half
  operator()(const half& val, const half& low, const half& high) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(__hgt(val, high) ? high : val, low) ? low : val;
#else
    const float valf = __half2float(val);
    const float highf = __half2float(high);
    return ((valf > highf ? valf : highf) < __half2float(low)) ? low : val;
#endif
  }
};

template <>
struct ClampFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(
      const nv_bfloat16& val,
      const nv_bfloat16& low,
      const nv_bfloat16& high) const {
#if __CUDA_ARCH__ >= 800
    return __hlt(__hgt(val, high) ? high : val, low) ? low : val;
#else
    const float valf = __bfloat162float(val);
    const float highf = __bfloat162float(high);
    return ((valf > highf ? valf : highf) < __bfloat162float(low)) ? low : val;
#endif
  }
};
#endif

template <typename T>
struct PlusFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Plus, float16, float16, +);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Plus, bfloat16, bfloat16, +);

#if defined(__CUDACC__)
template <>
struct PlusFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hadd(lhs, rhs);
#else
    return __float2half(__half2float(lhs) + __half2float(rhs));
#endif
  }
};

template <>
struct PlusFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hadd2(lhs, rhs);
#else
    return __floats2half2_rn(
        __half2float(lhs.x) + __half2float(rhs.x),
        __half2float(lhs.y) + __half2float(rhs.y));
#endif
  }
};

template <>
struct PlusFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hadd(lhs, rhs);
#else
    return __float2bfloat16(__bfloat162float(lhs) + __bfloat162float(rhs));
#endif
  }
};

template <>
struct PlusFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hadd2(lhs, rhs);
#else
    return __floats2bfloat162_rn(
        __bfloat162float(lhs.x) + __bfloat162float(rhs.x),
        __bfloat162float(lhs.y) + __bfloat162float(rhs.y));
#endif
  }
};
#endif

template <typename T>
struct MinusFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs - rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Minus, float16, float16, -);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Minus, bfloat16, bfloat16, -);

#if defined(__CUDACC__)
template <>
struct MinusFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hsub(lhs, rhs);
#else
    return __float2half(__half2float(lhs) - __half2float(rhs));
#endif
  }
};

template <>
struct MinusFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hsub2(lhs, rhs);
#else
    return __floats2half2_rn(
        __half2float(lhs.x) - __half2float(rhs.x),
        __half2float(lhs.y) - __half2float(rhs.y));
#endif
  }
};

template <>
struct MinusFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hsub(lhs, rhs);
#else
    return __float2bfloat16(__bfloat162float(lhs) - __bfloat162float(rhs));
#endif
  }
};

template <>
struct MinusFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hsub2(lhs, rhs);
#else
    return __floats2bfloat162_rn(
        __bfloat162float(lhs.x) - __bfloat162float(rhs.x),
        __bfloat162float(lhs.y) - __bfloat162float(rhs.y));
#endif
  }
};
#endif

template <typename T>
struct MultipliesFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs * rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Multiplies, float16, float16, *);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Multiplies, bfloat16, bfloat16, *);

#if defined(__CUDACC__)
template <>
struct MultipliesFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hmul(lhs, rhs);
#else
    return __float2half(__half2float(lhs) * __half2float(rhs));
#endif
  }
};

template <>
struct MultipliesFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hmul2(lhs, rhs);
#else
    return __floats2half2_rn(
        __half2float(lhs.x) * __half2float(rhs.x),
        __half2float(lhs.y) * __half2float(rhs.y));
#endif
  }
};

template <>
struct MultipliesFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hmul(lhs, rhs);
#else
    return __float2bfloat16(__bfloat162float(lhs) * __bfloat162float(rhs));
#endif
  }
};

template <>
struct MultipliesFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hmul2(lhs, rhs);
#else
    return __floats2bfloat162_rn(
        __bfloat162float(lhs.x) * __bfloat162float(rhs.x),
        __bfloat162float(lhs.y) * __bfloat162float(rhs.y));
#endif
  }
};
#endif

template <typename T>
struct DividesFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs / rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Divides, float16, float16, /);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Divides, bfloat16, bfloat16, /);

#if defined(__CUDACC__)
template <>
struct DividesFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hdiv(lhs, rhs);
#else
    return __float2half(__half2float(lhs) / __half2float(rhs));
#endif
  }
};

template <>
struct DividesFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __h2div(lhs, rhs);
#else
    return __floats2half2_rn(
        __half2float(lhs.x) / __half2float(rhs.x),
        __half2float(lhs.y) / __half2float(rhs.y));
#endif
  }
};

template <>
struct DividesFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hdiv(lhs, rhs);
#else
    return __float2bfloat16(__bfloat162float(lhs) / __bfloat162float(rhs));
#endif
  }
};

template <>
struct DividesFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __h2div(lhs, rhs);
#else
    return __floats2bfloat162_rn(
        __bfloat162float(lhs.x) / __bfloat162float(rhs.x),
        __bfloat162float(lhs.y) / __bfloat162float(rhs.y));
#endif
  }
};
#endif

template <typename T>
struct PowFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return pow(lhs, rhs);
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return std::pow(lhs, rhs);
  }
#endif
};

template <>
struct PowFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        std::pow(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
};

template <>
struct PowFunctor<bfloat16> {
  inline bfloat16 operator()(const bfloat16& lhs, const bfloat16& rhs) const {
    return convert::To<bfloat16>(
        std::pow(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
};

#if defined(__CUDACC__)
template <>
struct PowFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct PowFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    return __floats2half2_rn(
        pow(__half2float(lhs.x), __half2float(rhs.x)),
        pow(__half2float(lhs.y), __half2float(rhs.y)));
  }
};

template <>
struct PowFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
    return __float2bfloat16(pow(__bfloat162float(lhs), __bfloat162float(rhs)));
  }
};

template <>
struct PowFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
    return __floats2bfloat162_rn(
        pow(__bfloat162float(lhs.x), __bfloat162float(rhs.x)),
        pow(__bfloat162float(lhs.y), __bfloat162float(rhs.y)));
  }
};
#endif

template <typename T>
struct Atan2Functor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return atan2(lhs, rhs);
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return std::atan2(lhs, rhs);
  }
#endif
};

template <>
struct Atan2Functor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        std::atan2(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
};

template <>
struct Atan2Functor<bfloat16> {
  inline bfloat16 operator()(const bfloat16& lhs, const bfloat16& rhs) const {
    return convert::To<bfloat16>(
        std::atan2(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
};

#if defined(__CUDACC__)
template <>
struct Atan2Functor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
    return __float2half(atan2f(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct Atan2Functor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    return __floats2half2_rn(
        atan2f(__half2float(lhs.x), __half2float(rhs.x)),
        atan2f(__half2float(lhs.y), __half2float(rhs.y)));
  }
};

template <>
struct Atan2Functor<nv_bfloat16> {
  inline __device__ nv_bfloat16
  operator()(const nv_bfloat16& lhs, const nv_bfloat16& rhs) const {
    return __float2bfloat16(
        atan2f(__bfloat162float(lhs), __bfloat162float(rhs)));
  }
};

template <>
struct Atan2Functor<nv_bfloat162> {
  inline __device__ nv_bfloat162
  operator()(const nv_bfloat162& lhs, const nv_bfloat162& rhs) const {
    return __floats2bfloat162_rn(
        atan2f(__bfloat162float(lhs.x), __bfloat162float(rhs.x)),
        atan2f(__bfloat162float(lhs.y), __bfloat162float(rhs.y)));
  }
};
#endif

template <typename T>
struct FMAFunctor {
#if defined(__CUDACC__)
  inline __device__ T operator()(const T& a, const T& b, const T& c) const {
    return fma(a, b, c);
  }
#else
  inline T operator()(const T& a, const T& b, const T& c) const {
    return std::fma(a, b, c);
  }
#endif
};

#if defined(__CUDACC__)
template <>
struct FMAFunctor<half> {
  inline __device__ half
  operator()(const half& a, const half& b, const half& c) const {
#if __CUDA_ARCH__ >= 530
    return __hfma(a, b, c);
#else
    return __float2half(
        fmaf(__half2float(a), __half2float(b), __half2float(c)));
#endif
  }
};

template <>
struct FMAFunctor<half2> {
  inline __device__ half2
  operator()(const half2& a, const half2& b, const half2& c) const {
#if __CUDA_ARCH__ >= 530
    return __hfma2(a, b, c);
#else
    return __floats2half2_rn(
        fmaf(__half2float(a.x), __half2float(b.x), __half2float(c.x)),
        fmaf(__half2float(a.y), __half2float(b.y), __half2float(c.y)));
#endif
  }
};

template <>
struct FMAFunctor<nv_bfloat16> {
  inline __device__ nv_bfloat16 operator()(
      const nv_bfloat16& a,
      const nv_bfloat16& b,
      const nv_bfloat16& c) const {
#if __CUDA_ARCH__ >= 800
    return __hfma(a, b, c);
#else
    return __float2bfloat16(
        fmaf(__bfloat162float(a), __bfloat162float(b), __bfloat162float(c)));
#endif
  }
};

template <>
struct FMAFunctor<nv_bfloat162> {
  inline __device__ nv_bfloat162 operator()(
      const nv_bfloat162& a,
      const nv_bfloat162& b,
      const nv_bfloat162& c) const {
#if __CUDA_ARCH__ >= 800
    return __hfma2(a, b, c);
#else
    return __floats2bfloat162_rn(
        fmaf(
            __bfloat162float(a.x),
            __bfloat162float(b.x),
            __bfloat162float(c.x)),
        fmaf(
            __bfloat162float(a.y),
            __bfloat162float(b.y),
            __bfloat162float(c.y)));
#endif
  }
};
#endif

/*
 * Logical Functors
 */

template <typename T>
struct NotFunctor {
  HOSTDEVICE_DECL bool operator()(const T& x) const {
    return !x;
  }
};

template <>
struct NotFunctor<float16> {
  inline bool operator()(const float16& x) const {
    return !convert::To<float>(x);
  }
};

template <>
struct NotFunctor<bfloat16> {
  inline bool operator()(const bfloat16& x) const {
    return !convert::To<float>(x);
  }
};

#if defined(__CUDACC__)
template <>
struct NotFunctor<half> {
  inline __device__ bool operator()(const half& x) const {
    return !__half2float(x);
  }
};

template <>
struct NotFunctor<nv_bfloat16> {
  inline __device__ bool operator()(const nv_bfloat16& x) const {
    return !__bfloat162float(x);
  }
};
#endif

template <typename T>
struct AndFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs && rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(And, float16, bool, &&);
DEFINE_HOST_BINARY_HALF_FUNCTOR(And, bfloat16, bool, &&);

#if defined(__CUDACC__)
template <>
struct AndFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return __half2float(lhs) && __half2float(rhs);
  }
};

template <>
struct AndFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
    return __bfloat162float(lhs) && __bfloat162float(rhs);
  }
};
#endif

template <typename T>
struct OrFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs || rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Or, float16, bool, ||);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Or, bfloat16, bool, ||);

#if defined(__CUDACC__)
template <>
struct OrFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return __half2float(lhs) || __half2float(rhs);
  }
};

template <>
struct OrFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
    return __bfloat162float(lhs) || __bfloat162float(rhs);
  }
};
#endif

template <typename T>
struct XorFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return convert::To<bool>(lhs) ^ convert::To<bool>(rhs);
  }
};

template <>
struct XorFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<bool>(convert::To<float>(lhs)) ^
        convert::To<bool>(convert::To<float>(rhs));
  }
};

template <>
struct XorFunctor<bfloat16> {
  inline bool operator()(const bfloat16& lhs, const bfloat16& rhs) const {
    return convert::To<bool>(convert::To<float>(lhs)) ^
        convert::To<bool>(convert::To<float>(rhs));
  }
};

#if defined(__CUDACC__)
template <>
struct XorFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return convert::To<bool>(__half2float(lhs)) ^
        convert::To<bool>(__half2float(rhs));
  }
};

template <>
struct XorFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
    return convert::To<bool>(__bfloat162float(lhs)) ^
        convert::To<bool>(__bfloat162float(rhs));
  }
};
#endif

/*
 * Bitwise Functors
 */

template <typename T>
struct BitNotFunctor {
  HOSTDEVICE_DECL T operator()(const T& x) const {
    return ~x;
  }
};

template <typename T>
struct BitAndFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs & rhs;
  }
};

template <typename T>
struct BitOrFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs | rhs;
  }
};

template <typename T>
struct BitXorFunctor {
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs ^ rhs;
  }
};

/*
 * Compare Functors
 */

template <typename T>
struct EqualFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs == rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Equal, float16, bool, ==);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Equal, bfloat16, bool, ==);

#if defined(__CUDACC__)
template <>
struct EqualFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __heq(lhs, rhs);
#else
    return __half2float(lhs) == __half2float(rhs);
#endif
  }
};

template <>
struct EqualFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __heq(lhs, rhs);
#else
    return __bfloat162float(lhs) == __bfloat162float(rhs);
#endif
  }
};
#endif

template <typename T>
struct NotEqualFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs != rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(NotEqual, float16, bool, !=);
DEFINE_HOST_BINARY_HALF_FUNCTOR(NotEqual, bfloat16, bool, !=);

#if defined(__CUDACC__)
template <>
struct NotEqualFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hne(lhs, rhs);
#else
    return __half2float(lhs) != __half2float(rhs);
#endif
  }
};

template <>
struct NotEqualFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hne(lhs, rhs);
#else
    return __bfloat162float(lhs) != __bfloat162float(rhs);
#endif
  }
};
#endif

template <typename T>
struct GreaterFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Greater, float16, bool, >);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Greater, bfloat16, bool, >);

#if defined(__CUDACC__)
template <>
struct GreaterFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hgt(lhs, rhs);
#else
    return __half2float(lhs) > __half2float(rhs);
#endif
  }
};

template <>
struct GreaterFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hgt(lhs, rhs);
#else
    return __bfloat162float(lhs) > __bfloat162float(rhs);
#endif
  }
};
#endif

template <typename T>
struct LessFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(Less, float16, bool, <);
DEFINE_HOST_BINARY_HALF_FUNCTOR(Less, bfloat16, bool, <);

#if defined(__CUDACC__)
template <>
struct LessFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(lhs, rhs);
#else
    return __half2float(lhs) < __half2float(rhs);
#endif
  }
};

template <>
struct LessFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hlt(lhs, rhs);
#else
    return __bfloat162float(lhs) < __bfloat162float(rhs);
#endif
  }
};
#endif

template <typename T>
struct GreaterEqualFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs >= rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(GreaterEqual, float16, bool, >=);
DEFINE_HOST_BINARY_HALF_FUNCTOR(GreaterEqual, bfloat16, bool, >=);

#if defined(__CUDACC__)
template <>
struct GreaterEqualFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hge(lhs, rhs);
#else
    return __half2float(lhs) >= __half2float(rhs);
#endif
  }
};

template <>
struct GreaterEqualFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hge(lhs, rhs);
#else
    return __bfloat162float(lhs) >= __bfloat162float(rhs);
#endif
  }
};
#endif

template <typename T>
struct LessEqualFunctor {
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs <= rhs;
  }
};
DEFINE_HOST_BINARY_HALF_FUNCTOR(LessEqual, float16, bool, <=);
DEFINE_HOST_BINARY_HALF_FUNCTOR(LessEqual, bfloat16, bool, <=);

#if defined(__CUDACC__)
template <>
struct LessEqualFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hle(lhs, rhs);
#else
    return __half2float(lhs) <= __half2float(rhs);
#endif
  }
};

template <>
struct LessEqualFunctor<nv_bfloat16> {
  inline __device__ bool operator()(
      const nv_bfloat16& lhs,
      const nv_bfloat16& rhs) const {
#if __CUDA_ARCH__ >= 800
    return __hle(lhs, rhs);
#else
    return __bfloat162float(lhs) <= __bfloat162float(rhs);
#endif
  }
};
#endif

} // namespace math

} // namespace dragon

#undef HOSTDEVICE_DECL
#undef DEFINE_HOST_BINARY_HALF_FUNCTOR

#endif // DRAGON_UTILS_MATH_FUNCTIONAL_H_
