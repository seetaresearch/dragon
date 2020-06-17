#include "dragon/utils/cast.h"
#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void MixedPrecL2Decay<float16, CPUContext>(
    const int count,
    const float alpha,
    const float16* w,
    float* dx,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    dx[i] += (cast::to<float>(w[i]) * alpha);
  }
}

template <>
void MixedPrecUpdate<float16, CPUContext>(
    const int count,
    const float* updates,
    float16* w,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    w[i] = cast::to<float16>(cast::to<float>(w[i]) - updates[i]);
  }
}

} // namespace kernel

} // namespace dragon
