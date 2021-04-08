#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void RMSPropUpdate<float, CPUContext>(
    const int N,
    const float lr,
    const float momentum,
    const float decay,
    const float eps,
    float* g,
    float* m,
    float* v,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float gi = g[i];
    float vi = v[i] = decay * v[i] + (1 - decay) * gi * gi;
    g[i] = m[i] = momentum * m[i] + (lr * gi / (std::sqrt(vi) + eps));
  }
}

} // namespace kernels

} // namespace dragon
