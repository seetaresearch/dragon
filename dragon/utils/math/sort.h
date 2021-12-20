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

#ifndef DRAGON_UTILS_MATH_SORT_H_
#define DRAGON_UTILS_MATH_SORT_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

/*
 * Sort Utilities.
 */

namespace utils {

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
        [&v](int64_t lhs, int64_t rhs) { return v[lhs] > v[rhs]; });
  } else {
    std::nth_element(
        indices.begin(),
        indices.begin() + kth,
        indices.end(),
        [&v](int64_t lhs, int64_t rhs) { return v[lhs] < v[rhs]; });
  }
}

} // namespace utils

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_SORT_H_
