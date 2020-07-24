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

#ifndef DRAGON_UTILS_OMP_UTILS_H_
#define DRAGON_UTILS_OMP_UTILS_H_

#ifdef USE_OPENMP

#include <omp.h>
#include <algorithm>

namespace dragon {

#define OMP_MIN_ITERATORS_PER_CORE 200000

inline int OMP_THREADS(const int N) {
  int threads = std::max(N / OMP_MIN_ITERATORS_PER_CORE, 1);
  return std::min(threads, omp_get_num_procs());
}

} // namespace dragon

#endif // USE_OPENMP

#endif // DRAGON_UTILS_OMP_UTILS_H_
