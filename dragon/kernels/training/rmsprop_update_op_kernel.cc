#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void RMSPropUpdate<float, CPUContext>(
    const int count,
    const float lr,
    const float momentum,
    const float decay,
    const float eps,
    float* g,
    float* m,
    float* v,
    CPUContext* ctx) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    float gi = g[i];
    float vi = v[i] = decay * v[i] + (1 - decay) * gi * gi;
    g[i] = m[i] = momentum * m[i] + (lr * gi / (std::sqrt(vi) + eps));
  }
}

} // namespace kernel

} // namespace dragon
