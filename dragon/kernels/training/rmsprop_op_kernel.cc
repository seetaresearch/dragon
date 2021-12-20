#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void RMSprop<float, CPUContext>(
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
    float mi = m[i] = std::fma(momentum, m[i], gi / (std::sqrt(vi) + eps));
    g[i] = lr * mi;
  }
}

} // namespace kernels

} // namespace dragon
