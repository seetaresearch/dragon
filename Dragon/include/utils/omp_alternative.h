// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_OMP_ALTERNATIVE_H_
#define DRAGON_UTILS_OMP_ALTERNATIVE_H_

#ifdef WITH_OMP

#include <algorithm>
#include <omp.h>

namespace dragon {

#define OMP_MIN_ITERATORS_PER_CORE 256

inline int GET_OMP_THREADS(const int N) { 
   int threads = std::max(N / OMP_MIN_ITERATORS_PER_CORE, 1); 
   return std::min(threads, omp_get_num_procs());
}

}

#endif  // WITH_OMP

#endif  // DRAGON_UTILS_OMP_ALTERNATIVE_H_