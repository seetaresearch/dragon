// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_UTILS_OMP_ALTERNATIVE_H_
#define DRAGON_UTILS_OMP_ALTERNATIVE_H_

#ifdef WITH_OMP

#include <algorithm>
#include <omp.h>

namespace dragon {

#define OMP_MIN_ITERATORS_PER_CORE 200000

inline int GET_OMP_THREADS(const int N) { 
   int threads = std::max(N / OMP_MIN_ITERATORS_PER_CORE, 1); 
   return std::min(threads, omp_get_num_procs());
}

}

#endif  // WITH_OMP

#endif  // DRAGON_UTILS_OMP_ALTERNATIVE_H_