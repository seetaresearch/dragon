#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void SGDUpdate<float, CPUContext>(
    const int count,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    float mi = m[i];
    g[i] = m[i] = momentum * mi + lr * g[i];
  }
}

} // namespace kernel

} // namespace dragon
