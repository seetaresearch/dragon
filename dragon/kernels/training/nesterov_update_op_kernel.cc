#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void NesterovUpdate<float, CPUContext>(
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
    float mi_new = m[i] = momentum * mi + lr * g[i];
    g[i] = (1 + momentum) * mi_new - momentum * mi;
  }
}

} // namespace kernel

} // namespace dragon
