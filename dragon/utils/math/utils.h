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
HOSTDEVICE_DECL bool IsInf(const T x) {
#if defined(__CUDA_ARCH__)
  return isinf(x);
#else
  return std::isinf(x);
#endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
HOSTDEVICE_DECL bool IsNaN(const T x) {
#if defined(__CUDA_ARCH__)
  return isnan(x);
#else
  return std::isnan(x);
#endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
HOSTDEVICE_DECL bool IsFinite(const T x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

inline bool IsInf(const float16 x) {
  return std::isinf(convert::To<float>(x));
}

inline bool IsInf(const bfloat16 x) {
  return std::isinf(convert::To<float>(x));
}

inline bool IsNaN(float16 x) {
  return std::isnan(convert::To<float>(x));
}

inline bool IsNaN(bfloat16 x) {
  return std::isnan(convert::To<float>(x));
}

inline bool IsFinite(float16 x) {
  const float v = convert::To<float>(x);
  return !(std::isinf(v) || std::isnan(v));
}

inline bool IsFinite(bfloat16 x) {
  const float v = convert::To<float>(x);
  return !(std::isinf(v) || std::isnan(v));
}

template <typename T>
HOSTDEVICE_DECL bool IsAGeZeroAndALtB(const T a, const T b) {
  return static_cast<unsigned int>(a) < static_cast<unsigned int>(b);
}

template <typename T>
HOSTDEVICE_DECL T Sign(const T x) {
  return (x > T(0)) - (x < T(0));
}

template <typename T>
HOSTDEVICE_DECL T Identity(const T x) {
  return x;
}

template <typename T>
HOSTDEVICE_DECL T Sqr(const T x) {
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
template <typename T>
inline __device__ T LDG(const T* ptr) {
  return __ldg(ptr);
}

template <typename AccT, typename T>
inline __device__ AccT LDGC(const T* ptr) {
  return convert::To<AccT>(LDG(ptr));
}

template <>
inline __device__ nv_bfloat16 LDG<nv_bfloat16>(const __nv_bfloat16* ptr) {
#if __CUDA_ARCH__ >= 800
  return __ldg(ptr);
#else
  return *ptr;
#endif
}

inline __device__ bool IsInf(const half x) {
#if __CUDA_ARCH__ >= 530
  return __hisinf(x);
#else
  return isinf(__half2float(x));
#endif
}

inline __device__ bool IsInf(const nv_bfloat16 x) {
#if __CUDA_ARCH__ >= 800
  return __hisinf(x);
#else
  return isinf(__bfloat162float(x));
#endif
}

inline __device__ bool IsNaN(const half x) {
#if __CUDA_ARCH__ >= 530
  return __hisnan(x);
#else
  return isnan(__half2float(x));
#endif
}

inline __device__ bool IsNaN(const nv_bfloat16 x) {
#if __CUDA_ARCH__ >= 800
  return __hisnan(x);
#else
  return isnan(__bfloat162float(x));
#endif
}

inline __device__ bool IsFinite(const half x) {
#if __CUDA_ARCH__ >= 530
  return !(__hisinf(x) || __hisnan(x));
#else
  const float v = __half2float(x);
  return !(isinf(v) || isnan(v));
#endif
}

inline __device__ bool IsFinite(const nv_bfloat16 x) {
#if __CUDA_ARCH__ >= 800
  return !(__hisinf(x) || __hisnan(x));
#else
  const float v = __bfloat162float(x);
  return !(isinf(v) || isnan(v));
#endif
}
#endif // defined(__CUDACC__)

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
