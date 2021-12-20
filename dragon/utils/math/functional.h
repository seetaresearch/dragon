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
#include "dragon/utils/conversions.h"

namespace dragon {

namespace math {

/*
 * Arithmetic Functors */

template <typename T>
struct IdentityFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& x) const {
    return x;
  }
#else
  inline T operator()(const T& x) const {
    return x;
  }
#endif
};

template <typename T>
struct AbsFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& x) const {
    return abs(x);
  }
#else
  inline T operator()(const T& x) const {
    return std::abs(x);
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct AbsFunctor<half> {
  inline __device__ half operator()(const half& x) const {
#if __CUDA_ARCH__ >= 530
    return __habs(x);
#else
    return __float2half(fabsf(__half2float(x)));
#endif
  }
};

template <>
struct AbsFunctor<half2> {
  inline __device__ half2 operator()(const half2& x) const {
#if __CUDA_ARCH__ >= 530
    return __habs2(x);
#else
    const float2 v = __half22float2(x);
    return __floats2half2_rn(fabsf(v.x), fabsf(v.y));
#endif
  }
};
#endif

template <typename T>
struct SqrFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& x) const {
    return x * x;
  }
#else
  inline T operator()(const T& x) const {
    return x * x;
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct SqrFunctor<half> {
  inline __device__ half operator()(const half& x) const {
#if __CUDA_ARCH__ >= 530
    return __hmul(x, x);
#else
    const float v = __half2float(x);
    return __float2half(v * v);
#endif
  }
};

template <>
struct SqrFunctor<half2> {
  inline __device__ half2 operator()(const half2& x) const {
#if __CUDA_ARCH__ >= 530
    return __hmul2(x, x);
#else
    const float2 v = __half22float2(x);
    return __floats2half2_rn(v.x * v.x, v.y * v.y);
#endif
  }
};
#endif

template <typename T>
struct MaxFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? rhs : lhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? rhs : lhs;
  }
#endif
};

template <>
struct MaxFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(
               *reinterpret_cast<const half*>(&lhs),
               *reinterpret_cast<const half*>(&rhs))
        ? rhs
        : lhs;
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) <
            __half2float(*reinterpret_cast<const half*>(&rhs))
        ? rhs
        : lhs;
#endif
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? rhs : lhs;
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(
        v1.x < v2.x ? v2.x : v1.x, v1.y < v2.y ? v2.y : v1.y);
  }
};
#endif

template <typename T>
struct MinFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs ? lhs : rhs;
  }
#endif
};

template <>
struct MinFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    return __hlt(
               *reinterpret_cast<const half*>(&lhs),
               *reinterpret_cast<const half*>(&rhs))
        ? lhs
        : rhs;
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) <
            __half2float(*reinterpret_cast<const half*>(&rhs))
        ? lhs
        : rhs;
#endif
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs) ? lhs : rhs;
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(
        v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y);
  }
};
#endif

template <typename T>
struct PlusFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
#endif
};

template <>
struct PlusFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    half ret = __hadd(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    half ret = __float2half(
        __half2float(*reinterpret_cast<const half*>(&lhs)) +
        __half2float(*reinterpret_cast<const half*>(&rhs)));
#endif
    return *reinterpret_cast<float16*>(&ret);
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) + convert::To<float>(rhs));
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(v1.x + v2.x, v1.y + v2.y);
#endif
  }
};
#endif

template <typename T>
struct MinusFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs - rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs - rhs;
  }
#endif
};

template <>
struct MinusFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    half ret = __hsub(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    half ret = __float2half(
        __half2float(*reinterpret_cast<const half*>(&lhs)) -
        __half2float(*reinterpret_cast<const half*>(&rhs)));
#endif
    return *reinterpret_cast<float16*>(&ret);
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) - convert::To<float>(rhs));
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(v1.x - v2.x, v1.y - v2.y);
#endif
  }
};
#endif

template <typename T>
struct MultipliesFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs * rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs * rhs;
  }
#endif
};

template <>
struct MultipliesFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    half ret = __hmul(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    half ret = __float2half(
        __half2float(*reinterpret_cast<const half*>(&lhs)) *
        __half2float(*reinterpret_cast<const half*>(&rhs)));
#endif
    return *reinterpret_cast<float16*>(&ret);
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) * convert::To<float>(rhs));
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(v1.x * v2.x, v1.y * v2.y);
#endif
  }
};
#endif

template <typename T>
struct DividesFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs / rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs / rhs;
  }
#endif
};

template <>
struct DividesFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
#if __CUDA_ARCH__ >= 530
    half ret = __hdiv(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    half ret = __float2half(
        __half2float(*reinterpret_cast<const half*>(&lhs)) /
        __half2float(*reinterpret_cast<const half*>(&rhs)));
#endif
    return *reinterpret_cast<float16*>(&ret);
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) / convert::To<float>(rhs));
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(v1.x / v2.x, v1.y / v2.y);
  }
};
#endif

template <typename T>
struct PowFunctor {
#if defined(__CUDA_ARCH__)
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
#if defined(__CUDA_ARCH__)
  inline __device__ float16
  operator()(const float16& lhs, const float16& rhs) const {
    half ret = __float2half(
        pow(__half2float(*reinterpret_cast<const half*>(&lhs)),
            __half2float(*reinterpret_cast<const half*>(&rhs))));
    return *reinterpret_cast<float16*>(&ret);
  }
#else
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        std::pow(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct PowFunctor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct PowFunctor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(pow(v1.x, v2.x), pow(v1.y, v2.y));
  }
};
#endif

/*
 * Logical Functors
 */

template <typename T>
struct NotFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& x) const {
    return !x;
  }
#else
  inline bool operator()(const T& x) const {
    return !x;
  }
#endif
};

template <>
struct NotFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& x) const {
    return !__half2float(*reinterpret_cast<const half*>(&x));
  }
#else
  inline bool operator()(const float16& x) const {
    return !convert::To<float>(x);
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct NotFunctor<half> {
  inline __device__ bool operator()(const half& x) const {
    return !__half2float(x);
  }
};
#endif

template <typename T>
struct AndFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs && rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs && rhs;
  }
#endif
};

template <>
struct AndFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
    return __half2float(*reinterpret_cast<const half*>(&lhs)) &&
        __half2float(*reinterpret_cast<const half*>(&rhs));
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) && convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct AndFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return __half2float(lhs) && __half2float(rhs);
  }
};
#endif

template <typename T>
struct OrFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs || rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs || rhs;
  }
#endif
};

template <>
struct OrFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
    return __half2float(*reinterpret_cast<const half*>(&lhs)) ||
        __half2float(*reinterpret_cast<const half*>(&rhs));
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) || convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct OrFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return __half2float(lhs) || __half2float(rhs);
  }
};
#endif

template <typename T>
struct XorFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return convert::To<bool>(lhs) ^ convert::To<bool>(rhs);
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return convert::To<bool>(lhs) ^ convert::To<bool>(rhs);
  }
#endif
};

template <>
struct XorFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
    return convert::To<bool>(
               __half2float(*reinterpret_cast<const half*>(&lhs))) ^
        convert::To<bool>(__half2float(*reinterpret_cast<const half*>(&rhs)));
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<bool>(convert::To<float>(lhs)) ^
        convert::To<bool>(convert::To<float>(rhs));
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct XorFunctor<half> {
  inline __device__ bool operator()(const half& lhs, const half& rhs) const {
    return convert::To<bool>(__half2float(lhs)) ^
        convert::To<bool>(__half2float(rhs));
  }
};
#endif

/*
 * Bitwise Functors
 */

template <typename T>
struct BitNotFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& x) const {
    return ~x;
  }
#else
  inline T operator()(const T& x) const {
    return ~x;
  }
#endif
};

template <typename T>
struct BitAndFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs & rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs & rhs;
  }
#endif
};

template <typename T>
struct BitOrFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs | rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs | rhs;
  }
#endif
};

template <typename T>
struct BitXorFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& lhs, const T& rhs) const {
    return lhs ^ rhs;
  }
#else
  inline T operator()(const T& lhs, const T& rhs) const {
    return lhs ^ rhs;
  }
#endif
};

/*
 * Compare Functors
 */

template <typename T>
struct EqualFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs == rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs == rhs;
  }
#endif
};

template <>
struct EqualFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __heq(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) ==
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) == convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

template <typename T>
struct NotEqualFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs != rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs != rhs;
  }
#endif
};

template <>
struct NotEqualFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __hne(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) !=
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) != convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

template <typename T>
struct GreaterFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }
#endif
};

template <>
struct GreaterFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __hgt(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) >
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) > convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

template <typename T>
struct LessFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }
#endif
};

template <>
struct LessFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __hlt(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) <
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

template <typename T>
struct GreaterEqualFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs >= rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs >= rhs;
  }
#endif
};

template <>
struct GreaterEqualFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __hge(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) >=
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) >= convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

template <typename T>
struct LessEqualFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const T& lhs, const T& rhs) const {
    return lhs <= rhs;
  }
#else
  inline bool operator()(const T& lhs, const T& rhs) const {
    return lhs <= rhs;
  }
#endif
};

template <>
struct LessEqualFunctor<float16> {
#if defined(__CUDA_ARCH__)
  inline __device__ bool operator()(const float16& lhs, const float16& rhs)
      const {
#if __CUDA_ARCH__ >= 530
    return __hle(
        *reinterpret_cast<const half*>(&lhs),
        *reinterpret_cast<const half*>(&rhs));
#else
    return __half2float(*reinterpret_cast<const half*>(&lhs)) <
        __half2float(*reinterpret_cast<const half*>(&rhs));
#endif
  }
#else
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) <= convert::To<float>(rhs);
  }
#endif
};

#if defined(__CUDA_ARCH__)
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
#endif

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_FUNCTIONAL_H_
