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

#ifndef DRAGON_UTILS_MATH_TYPES_H_
#define DRAGON_UTILS_MATH_TYPES_H_

#include "dragon/core/types.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace math {

/*
 * Type Traits.
 */

template <typename T>
class Traits {
 public:
  using scalar_type = T;
  using scalar2_type = T;
  using eigen_type = T;
  using accumulator_type = T;
  static T Max() {
    return std::numeric_limits<T>::max();
  }
  static T Lowest() {
    return std::numeric_limits<T>::lowest();
  }
  static bool HasPack2() {
    return sizeof(scalar2_type) != sizeof(scalar_type);
  }
};

template <>
class Traits<int8_t> {
 public:
#if defined(__mlu_func__)
  using scalar_type = char;
  using scalar2_type = char;
#else
  using scalar_type = int8_t;
  using scalar2_type = int8_t;
#endif
  using eigen_type = int8_t;
  using accumulator_type = int8_t;
  static int8_t Max() {
    return std::numeric_limits<int8_t>::max();
  }
  static int8_t Lowest() {
    return std::numeric_limits<int8_t>::lowest();
  }
  static bool HasPack2() {
    return false;
  }
};

template <>
class Traits<float16> {
 public:
#if defined(__CUDACC__)
  using scalar_type = half;
  using scalar2_type = half2;
#elif defined(__mlu_func__)
  using scalar_type = half;
  using scalar2_type = half;
#else
  using scalar_type = float16;
  using scalar2_type = float16;
#endif
#if defined(EIGEN_WORLD_VERSION)
  using eigen_type = Eigen::half;
#else
  using eigen_type = float16;
#endif
  using accumulator_type = float;
  static float Max() {
    return 65504.f;
  }
  static float Lowest() {
    return -65505.f;
  }
  static bool HasPack2() {
    return sizeof(scalar2_type) != sizeof(scalar_type);
  }
};

template <>
class Traits<bfloat16> {
 public:
#if defined(__CUDACC__)
  using scalar_type = nv_bfloat16;
  using scalar2_type = nv_bfloat162;
#elif defined(__mlu_func__) && __BANG_ARCH__ >= 520
  using scalar_type = bfloat16_t;
  using scalar2_type = bfloat16_t;
#elif defined(__mlu_func__) && __BANG_ARCH__ < 520
  using scalar_type = half;
  using scalar2_type = half;
#else
  using scalar_type = bfloat16;
  using scalar2_type = bfloat16;
#endif
#if defined(EIGEN_WORLD_VERSION)
  using eigen_type = Eigen::bfloat16;
#else
  using eigen_type = bfloat16;
#endif
  using accumulator_type = float;
  static float Max() {
    return std::numeric_limits<float>::max();
  }
  static float Lowest() {
    return std::numeric_limits<float>::lowest();
  }
  static bool HasPack2() {
    return sizeof(scalar2_type) != sizeof(scalar_type);
  }
};

/*
 * Type Utilities.
 */

namespace utils {

template <typename T, size_t kMaxAlignedSize>
size_t GetAlignedSize(const int64_t N) {
  size_t itemsize = sizeof(T), size = N * sizeof(T);
  for (size_t v = kMaxAlignedSize; v > itemsize; v /= 2) {
    if (size % v == 0) return v;
  }
  return itemsize;
}

template <typename T, size_t kMaxAlignedSize>
size_t GetAlignedSize(const int64_t N, const T* x, T* y) {
  auto src = reinterpret_cast<std::uintptr_t>(x);
  auto dest = reinterpret_cast<std::uintptr_t>(y);
  size_t itemsize = sizeof(T), size = N * sizeof(T);
  for (size_t v = kMaxAlignedSize; v > itemsize; v /= 2) {
    if (size % v == 0 && src % v == 0 && dest % v == 0) return v;
  }
  return itemsize;
}

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_TYPES_H_
