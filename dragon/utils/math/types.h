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

namespace dragon {

namespace math {

/*
 * Type Classes.
 */

template <typename T>
class ScalarType {
 public:
  typedef T type;
  typedef T type2;
};

#if defined(__CUDACC__)
template <>
class ScalarType<float16> {
 public:
  typedef half type;
  typedef half2 type2;
};
#endif

template <typename T>
class AccumulatorType {
 public:
  typedef float type;
};

template <>
class AccumulatorType<int64_t> {
 public:
  typedef double type;
};

template <>
class AccumulatorType<double> {
 public:
  typedef double type;
};

/*
 * Type Utilities.
 */

namespace utils {

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
