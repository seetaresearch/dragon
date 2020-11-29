#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

template <>
void AdamUpdate<float, CPUContext>(
    const int count,
    const float lr,
    const float beta1,
    const float beta2,
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
    float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = lr * mi / (std::sqrt(vi) + eps);
  }
}

} // namespace kernel

} // namespace dragon
