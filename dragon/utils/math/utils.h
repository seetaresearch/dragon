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

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/utils/cast.h"

#ifdef USE_CUDA
#define MATH_UTILS_DECL inline __host__ __device__
#else
#define MATH_UTILS_DECL inline
#endif

#define FIXED_DIVISOR_DIV_MOD(d, n, q, r) \
  do {                                    \
    const auto n_copy = n;                \
    *q = n_copy / d;                      \
    *r = n_copy % d;                      \
  } while (0)

namespace dragon {

namespace utils {

namespace math {

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
MATH_UTILS_DECL T IsInf(const T x) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
MATH_UTILS_DECL T IsNaN(const T x) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
MATH_UTILS_DECL bool IsInf(T x) {
#if defined(__CUDACC__)
  return isinf(x);
#else
  return std::isinf(x);
#endif
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
MATH_UTILS_DECL bool IsNaN(T x) {
#if defined(__CUDACC__)
  return isnan(x);
#else
  return std::isnan(x);
#endif
}

inline bool IsInf(float16 x) {
  return std::abs(cast::to<float>(x)) > HFLT_MAX;
}

inline bool IsNaN(float16 x) {
  return IsNaN(cast::to<float>(x));
}

template <typename T>
MATH_UTILS_DECL T Sign(const T x) {
  return x > T(0) ? T(1) : (x < T(0) ? T(-1) : T(0));
}

template <typename T>
MATH_UTILS_DECL T Square(const T x) {
  return x * x;
}

template <typename T>
MATH_UTILS_DECL T Cube(const T x) {
  return x * x * x;
}

#if defined(__CUDACC__)
MATH_UTILS_DECL bool IsInf(half x) {
#if __CUDA_ARCH__ >= 530
  return __hisinf(x);
#else
  float fp32 = __half2float(x);
  return fp32 > HFLT_MAX || fp32 < HFLT_MAX;
#endif
}

MATH_UTILS_DECL bool IsNaN(half x) {
#if __CUDA_ARCH__ >= 530
  return __hisnan(x);
#else
  return isnan(__half2float(x));
#endif
}

MATH_UTILS_DECL half Square(half x) {
#if __CUDA_ARCH__ >= 530
  return __hmul(x, x);
#else
  return __float2half(Square(__half2float(x)));
#endif
}

MATH_UTILS_DECL half2 Square(half2 x) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(x, x);
#else
  const float2 val = __half22float2(x);
  return __floats2half2_rn(Square(val.x), Square(val.y));
#endif
}

MATH_UTILS_DECL half Cube(half x) {
#if __CUDA_ARCH__ >= 530
  return __hmul(__hmul(x, x), x);
#else
  return __float2half(Cube(__half2float(x)));
#endif
}

MATH_UTILS_DECL half2 Cube(half2 x) {
#if __CUDA_ARCH__ >= 530
  return __hmul2(__hmul2(x, x), x);
#else
  const float2 val = __half22float2(x);
  return __floats2half2_rn(Cube(val.x), Cube(val.y));
#endif
}
#endif // defined(__CUDACC__)

template <typename T>
inline void ArgPartition(
    const int count,
    const int kth,
    const bool descend,
    const T* v,
    vec64_t& indices) {
  indices.resize(count);
  std::iota(indices.begin(), indices.end(), 0);
  if (descend) {
    std::nth_element(
        indices.begin(),
        indices.begin() + kth,
        indices.end(),
        [&v](int64_t i1, int64_t i2) { return v[i1] > v[i2]; });
  } else {
    std::nth_element(
        indices.begin(),
        indices.begin() + kth,
        indices.end(),
        [&v](int64_t i1, int64_t i2) { return v[i1] < v[i2]; });
  }
}

DRAGON_API bool IsBinaryBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& Y_dims);

DRAGON_API bool IsRowwiseBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    int* rows,
    int* cols,
    int* broadcast_1st = nullptr);

DRAGON_API bool IsColwiseBroadcast(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    int* rows,
    int* cols,
    int* broadcast_1st = nullptr);

DRAGON_API bool IsRowwiseReduce(
    const int num_dims,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols);

DRAGON_API bool IsColwiseReduce(
    const int num_dims,
    const int* A_dims,
    const int* B_dims,
    int* rows,
    int* cols);

DRAGON_API void ComputeBinaryBroadcastDims(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& A_broadcast_dims,
    vec64_t& B_broadcast_dims);

DRAGON_API void ComputeBinaryBroadcastStrides(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    vec64_t& A_broadcast_strides,
    vec64_t& B_broadcast_strides,
    vec64_t& Y_dims);

DRAGON_API void ComputeBinaryBroadcastAxes(
    const vec64_t& A_dims,
    const vec64_t& B_dims,
    const vec64_t& Y_dims,
    vec32_t& A_broadcast_axes,
    vec32_t& B_broadcast_axes);

DRAGON_API void TransposeAxesForReduce(
    const int num_dims,
    const int num_axes,
    const int* reduce_axes,
    int* transpose_axes);

DRAGON_API void ComputeTransposeStrides(
    const int num_dims,
    const int* dims,
    const int* transpose_axes,
    int* transpose_strides);

template <typename T>
inline void IncreaseIndexInDims(const int num_dims, const T* dims, T* index) {
  for (int i = num_dims - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

} // namespace math

} // namespace utils

} // namespace dragon

#endif // DRAGON_UTILS_MATH_UTILS_H_
