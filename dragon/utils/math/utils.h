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

namespace math {

/*
 * Math Wrappers
 */

template <typename T>
class ScalarType {
 public:
  typedef T type;
};

#if defined(__CUDACC__)
template <>
class ScalarType<float16> {
 public:
  typedef half type;
};
#endif

template <typename T>
class AccmulatorType {
 public:
  typedef float type;
};

template <>
class AccmulatorType<int64_t> {
 public:
  typedef double type;
};

template <>
class AccmulatorType<double> {
 public:
  typedef double type;
};

namespace utils {

/*
 * Common Functions
 */

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
  return std::abs(convert::To<float>(x)) > HFLT_MAX;
}

inline bool IsNaN(float16 x) {
  return IsNaN(convert::To<float>(x));
}

template <typename T>
MATH_UTILS_DECL bool IsAGeZeroAndALtB(const T a, const T b) {
  return static_cast<unsigned int>(a) < static_cast<unsigned int>(b);
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

/*
 * CUDA Functions
 */

#if defined(__CUDACC__)

template <typename T>
inline __device__ T AtomicAdd(T* address, T val) {
  return atomicAdd(address, val);
}

#if __CUDA_ARCH__ < 600
inline __device__ double AtomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

#if __CUDA_ARCH__ < 700
inline __device__ half AtomicAdd(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui, assumed;
  __half_raw result;
  do {
    assumed = old;
    result.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
#if __CUDA_ARCH__ >= 530
    result = __hadd(half(result), val);
#else
    result = __float2half(__half2float(half(result)) + __half2float(val));
#endif
    old = (size_t)address & 2 ? (old & 0xffff) | (result.x << 16)
                              : (old & 0xffff0000) | result.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  result.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return half(result);
}
#endif

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
 * Math Utilities
 */

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

template <typename dim_t, typename stride_t>
inline void
ComputeStrides(const int num_dims, const dim_t* dims, stride_t* strides) {
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    strides[i] = stride_t(cur_stride);
    cur_stride *= int64_t(dims[i]);
  }
}

template <typename dim_t, typename axis_t, typename stride_t>
inline void ComputeTransposeStrides(
    const int num_dims,
    const dim_t* dims,
    const axis_t* axes,
    stride_t* strides) {
  vec64_t buf(num_dims);
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    buf[i] = cur_stride;
    cur_stride *= int64_t(dims[i]);
  }
  for (int i = 0; i < num_dims; ++i) {
    strides[i] = stride_t(buf[axes[i]]);
  }
}

template <typename dim_t, typename index_t>
inline void
IncreaseIndexInDims(const int num_dims, const dim_t* dims, index_t* index) {
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

#endif // DRAGON_UTILS_MATH_UTILS_H_
