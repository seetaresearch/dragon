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

#if defined(__CUDA_ARCH__)
#define HOSTDEVICE_DECL inline __host__ __device__
#else
#define HOSTDEVICE_DECL inline
#endif

namespace dragon {

namespace math {

/*
 * Arithmetic Functors
 */

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE_DECL T operator()(const T& x) const {
    return x;
  }
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
  HOSTDEVICE_DECL T operator()(const T& x) const {
    return x * x;
  }
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
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs + rhs;
  }
};

template <>
struct PlusFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) + convert::To<float>(rhs));
  }
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
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs - rhs;
  }
};

template <>
struct MinusFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) - convert::To<float>(rhs));
  }
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
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs * rhs;
  }
};

template <>
struct MultipliesFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) * convert::To<float>(rhs));
  }
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
  HOSTDEVICE_DECL T operator()(const T& lhs, const T& rhs) const {
    return lhs / rhs;
  }
};

template <>
struct DividesFunctor<float16> {
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        convert::To<float>(lhs) / convert::To<float>(rhs));
  }
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
  inline float16 operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float16>(
        std::pow(convert::To<float>(lhs), convert::To<float>(rhs)));
  }
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

template <typename T>
struct Atan2Functor {
#if defined(__CUDA_ARCH__)
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

#if defined(__CUDA_ARCH__)
template <>
struct Atan2Functor<half> {
  inline __device__ half operator()(const half& lhs, const half& rhs) const {
    return __float2half(atan2f(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct Atan2Functor<half2> {
  inline __device__ half2 operator()(const half2& lhs, const half2& rhs) const {
    const float2 v1 = __half22float2(lhs);
    const float2 v2 = __half22float2(rhs);
    return __floats2half2_rn(atan2f(v1.x, v2.x), atan2f(v1.y, v2.y));
  }
};
#endif

template <typename T>
struct FMAFunctor {
#if defined(__CUDA_ARCH__)
  inline __device__ T operator()(const T& x, const T& y, const T& z) const {
    return fma(x, y, z);
  }
#else
  inline T operator()(const T& x, const T& y, const T& z) const {
    return std::fma(x, y, z);
  }
#endif
};

#if defined(__CUDA_ARCH__)
template <>
struct FMAFunctor<half> {
  inline __device__ half
  operator()(const half& x, const half& y, const half& z) const {
#if __CUDA_ARCH__ >= 530
    return __hfma(x, y, z);
#else
    return __float2half(
        fmaf(__half2float(x), __half2float(y), __half2float(z)));
#endif
  }
};

template <>
struct FMAFunctor<half2> {
  inline __device__ half2
  operator()(const half2& x, const half2& y, const half2& z) const {
#if __CUDA_ARCH__ >= 530
    return __hfma2(x, y, z);
#else
    const float2 v1 = __half22float2(x);
    const float2 v2 = __half22float2(y);
    const float2 v3 = __half22float2(z);
    return __floats2half2_rn(fmaf(v1.x, v2.x, v3.x), fmaf(v1.y, v2.y, v3.y));
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs && rhs;
  }
};

template <>
struct AndFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) && convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs || rhs;
  }
};

template <>
struct OrFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) || convert::To<float>(rhs);
  }
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

template <>
struct EqualFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) == convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs != rhs;
  }
};

template <>
struct NotEqualFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) != convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }
};

template <>
struct GreaterFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) > convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }
};

template <>
struct LessFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) < convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs >= rhs;
  }
};

template <>
struct GreaterEqualFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) >= convert::To<float>(rhs);
  }
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
  HOSTDEVICE_DECL bool operator()(const T& lhs, const T& rhs) const {
    return lhs <= rhs;
  }
};

template <>
struct LessEqualFunctor<float16> {
  inline bool operator()(const float16& lhs, const float16& rhs) const {
    return convert::To<float>(lhs) <= convert::To<float>(rhs);
  }
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

#undef HOSTDEVICE_DECL

#endif // DRAGON_UTILS_MATH_FUNCTIONAL_H_
