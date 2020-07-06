#include "dragon/utils/cast.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void MixedPrecL2Penalty<float16, CPUContext>(
    const int count,
    const float alpha,
    const float16* x,
    float* dx,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    dx[i] += (cast::to<float>(x[i]) * alpha);
  }
}

template <>
void MixedPrecUpdate<float16, CPUContext>(
    const int count,
    const float* dx,
    float16* x,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    x[i] = cast::to<float16>(cast::to<float>(x[i]) - dx[i]);
  }
}

} // namespace kernel

} // namespace dragon
