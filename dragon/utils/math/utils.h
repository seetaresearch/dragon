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

#ifndef DRAGON_UTILS_MATH_UTILS_H_
#define DRAGON_UTILS_MATH_UTILS_H_

#include "dragon/utils/conversions.h"

#if defined(__CUDACC__)
#define HOSTDEVICE_DECL inline __host__ __device__
#elif defined(__mlu_host__)
#define HOSTDEVICE_DECL inline __mlu_host__ __mlu_func__
#else
#define HOSTDEVICE_DECL inline
#endif

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

namespace dragon {

namespace math {

namespace utils {

/*
 * Common Functions.
 */

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
HOSTDEVICE_DECL T IsInf(const T x) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
HOSTDEVICE_DECL T IsNaN(const T x) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
HOSTDEVICE_DECL T IsFinite(const T x) {
  return true;
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
HOSTDEVICE_DECL bool IsInf(T x) {
#if defined(__CUDACC__)
  return isinf(x);
#else
  return std::isinf(x);
#endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
HOSTDEVICE_DECL bool IsNaN(T x) {
#if defined(__CUDACC__)
  return isnan(x);
#else
  return std::isnan(x);
#endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
HOSTDEVICE_DECL bool IsFinite(T x) {
#if defined(__CUDACC__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

inline bool IsInf(float16 x) {
  return std::abs(convert::To<float>(x)) > HFLT_MAX;
}

inline bool IsNaN(float16 x) {
  return IsNaN(convert::To<float>(x));
}

inline bool IsFinite(float16 x) {
  const float v = convert::To<float>(x);
  return !(std::abs(v) > HFLT_MAX || IsNaN(v));
}

template <typename T>
HOSTDEVICE_DECL bool IsAGeZeroAndALtB(const T a, const T b) {
  return static_cast<unsigned int>(a) < static_cast<unsigned int>(b);
}

template <typename T>
HOSTDEVICE_DECL T Sign(const T x) {
  return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
}

template <typename T>
HOSTDEVICE_DECL T Identity(const T x) {
  return x;
}

template <typename T>
HOSTDEVICE_DECL T Square(const T x) {
  return x * x;
}

template <typename T>
HOSTDEVICE_DECL T Cube(const T x) {
  return x * x * x;
}

/*
 * CUDA Functions.
 */

#if defined(__CUDACC__)
inline __device__ bool IsInf(half x) {
#if __CUDA_ARCH__ >= 530
  return __hisinf(x);
#else
  float fp32 = __half2float(x);
  return fp32 > HFLT_MAX || fp32 < HFLT_MAX;
#endif
}

inline __device__ bool IsNaN(half x) {
#if __CUDA_ARCH__ >= 530
  return __hisnan(x);
#else
  return isnan(__half2float(x));
#endif
}

inline __device__ bool IsFinite(half x) {
#if __CUDA_ARCH__ >= 530
  return !(__hisinf(x) || __hisnan(x));
#else
  const float v = __half2float(x);
  return !(isinf(v) || isnan(v));
#endif
}

inline __device__ half Square(half x) {
#if __CUDA_ARCH__ >= 530
  return __hmul(x, x);
#else
  return __float2half(Square(__half2float(x)));
#endif
}

inline __device__ half2 Square(half2 x) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(x, x);
#else
  const float2 val = __half22float2(x);
  return __floats2half2_rn(Square(val.x), Square(val.y));
#endif
}

inline __device__ half Cube(half x) {
#if __CUDA_ARCH__ >= 530
  return __hmul(__hmul(x, x), x);
#else
  return __float2half(Cube(__half2float(x)));
#endif
}

inline __device__ half2 Cube(half2 x) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(__hmul2(x, x), x);
#else
  const float2 val = __half22float2(x);
  return __floats2half2_rn(Cube(val.x), Cube(val.y));
#endif
}
#endif // defined(__CUDACC__)

/*
 * MLU Functions.
 */

#if defined(__mlu_func__)
template <typename DstT, typename SrcT>
__mlu_func__ void Convert(DstT* dst, SrcT* src, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i] = DstT(src[i]);
  }
}

template <>
__mlu_func__ void Convert<int, uint8_t>(int* dst, uint8_t* src, int count) {
  __bang_uchar2int32(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<half, uint8_t>(half* dst, uint8_t* src, int count) {
  __bang_uchar2half(dst, src, count);
}

template <>
__mlu_func__ void Convert<float, uint8_t>(float* dst, uint8_t* src, int count) {
  __bang_uchar2float(dst, src, count);
}

template <>
__mlu_func__ void Convert<int, int8_t>(int* dst, int8_t* src, int count) {
  __bang_int82int32(dst, src, count, 0, 0);
}

template <>
__mlu_func__ void Convert<int, char>(int* dst, char* src, int count) {
  __bang_int82int32(dst, (int8_t*)src, count, 0, 0);
}

template <>
__mlu_func__ void Convert<half, int8_t>(half* dst, int8_t* src, int count) {
  __bang_int82half(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<half, char>(half* dst, char* src, int count) {
  __bang_int82half(dst, (int8_t*)src, count, 0);
}

template <>
__mlu_func__ void Convert<float, int8_t>(float* dst, int8_t* src, int count) {
  __bang_int82float(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<float, char>(float* dst, char* src, int count) {
  __bang_int82float(dst, (int8_t*)src, count, 0);
}

template <>
__mlu_func__ void Convert<uint8_t, int>(uint8_t* dst, int* src, int count) {
  __bang_int322uchar(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<int8_t, int>(int8_t* dst, int* src, int count) {
  __bang_int322int8(dst, src, count, 0, 0);
}

template <>
__mlu_func__ void Convert<char, int>(char* dst, int* src, int count) {
  __bang_int322int8((int8_t*)dst, src, count, 0, 0);
}

template <>
__mlu_func__ void Convert<half, int>(half* dst, int* src, int count) {
  __bang_int322half(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<float, int>(float* dst, int* src, int count) {
  __bang_int322float(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<uint8_t, half>(uint8_t* dst, half* src, int count) {
  __bang_half2uchar_dn(dst, src, count);
}

template <>
__mlu_func__ void Convert<float, half>(float* dst, half* src, int count) {
  __bang_half2float(dst, src, count);
}

template <>
__mlu_func__ void Convert<uint8_t, float>(uint8_t* dst, float* src, int count) {
  __bang_float2uchar(dst, src, count);
}

template <>
__mlu_func__ void Convert<int8_t, float>(int8_t* dst, float* src, int count) {
  __bang_float2int8_rn(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<char, float>(char* dst, float* src, int count) {
  __bang_float2int8_rn((int8_t*)dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<int, float>(int* dst, float* src, int count) {
  __bang_float2int32(dst, src, count, 0);
}

template <>
__mlu_func__ void Convert<half, float>(half* dst, float* src, int count) {
  __bang_float2half_rn(dst, src, count);
}
#endif // defined(__mlu_func__)

/*
 * Math Utilities.
 */

template <typename T>
HOSTDEVICE_DECL T DivUp(const T a, const T b) {
  return (a + b - T(1)) / b;
}

template <typename T, typename DimT>
inline T Prod(const DimT N, const T* v) {
  return std::accumulate(v, v + N, T(1), std::multiplies<T>());
}

template <typename T>
inline T Prod(const vector<T> v) {
  return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
}

/*
 * Indexing Utilities.
 */

template <typename DimT, typename IndexT>
IndexT GetIndexFromDims(const int num_dims, const DimT* dims, IndexT* index) {
  IndexT ret = 0;
  for (int i = 0; i < num_dims; ++i) {
    if (dims[i] > 1) ret = ret * dims[i] + index[i];
  }
  return ret;
}

template <typename DimT, typename IndexT>
void IncreaseIndexInDims(const int num_dims, const DimT* dims, IndexT* index) {
  for (int i = num_dims - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

} // namespace utils

} // namespace math

} // namespace dragon

#undef HOSTDEVICE_DECL

#endif // DRAGON_UTILS_MATH_UTILS_H_
