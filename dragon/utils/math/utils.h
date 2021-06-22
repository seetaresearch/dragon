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
inline T DivUp(const T a, const T b) {
  return (a + b - T(1)) / b;
}

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
    vec64_t& B_broadcast_dims,
    int64_t* C_broadcast_dims = nullptr);

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

template <typename DimT, typename StrideT>
inline void
ComputeStrides(const int num_dims, const DimT* dims, StrideT* strides) {
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    strides[i] = StrideT(cur_stride);
    cur_stride *= int64_t(dims[i]);
  }
}

template <typename DimT, typename AxisT, typename StrideT>
inline void ComputeTransposeStrides(
    const int num_dims,
    const DimT* dims,
    const AxisT* axes,
    StrideT* strides) {
  vec64_t buf(num_dims);
  int64_t cur_stride = 1;
  for (int i = num_dims - 1; i >= 0; --i) {
    buf[i] = cur_stride;
    cur_stride *= int64_t(dims[i]);
  }
  for (int i = 0; i < num_dims; ++i) {
    strides[i] = StrideT(buf[axes[i]]);
  }
}

template <typename DimT, typename AxisT>
inline void CollapseTransposeAxes(
    const int num_dims,
    const DimT* dims,
    const AxisT* axes,
    vector<DimT>& new_dims,
    vector<AxisT>& new_axes) {
  new_dims = vector<DimT>(dims, dims + num_dims);
  new_axes = vector<AxisT>({axes[0]});
  vector<AxisT> collapse_axes;
  for (int i = 1; i < num_dims; ++i) {
    if (axes[i] - 1 == axes[i - 1]) {
      collapse_axes.push_back(axes[i]);
      new_dims[axes[i]] *= new_dims[axes[i] - 1];
      new_dims[axes[i] - 1] = -1;
    } else {
      new_axes.push_back(axes[i]);
    }
  }
  const auto& erase_iter = std::remove_if(
      new_dims.begin(), new_dims.end(), [](int x) { return x == -1; });
  new_dims.erase(erase_iter, new_dims.end());
  for (int i = 0; i < new_axes.size(); ++i) {
    for (auto collapse_axis : collapse_axes) {
      if (new_axes[i] > collapse_axis) new_axes[i]--;
    }
  }
}

template <typename DimT, typename IndexT>
inline IndexT
GetIndexFromDims(const int num_dims, const DimT* dims, IndexT* index) {
  IndexT ret = 0;
  for (int i = 0; i < num_dims; ++i) {
    if (dims[i] > 1) ret = ret * dims[i] + index[i];
  }
  return ret;
}

template <typename DimT, typename IndexT>
inline void
IncreaseIndexInDims(const int num_dims, const DimT* dims, IndexT* index) {
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
