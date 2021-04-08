#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void SGDUpdate<float, CPUContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float mi = m[i];
    g[i] = m[i] = momentum * mi + lr * g[i];
  }
}

} // namespace kernels

} // namespace dragon
