#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void MomentumSGD<float, CPUContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float mi = m[i] = std::fma(momentum, m[i], g[i]);
    g[i] = lr * mi;
  }
}

template <>
void NesterovSGD<float, CPUContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = m[i] = std::fma(momentum, m[i], gi);
    g[i] = lr * std::fma(momentum, mi, gi);
  }
}

} // namespace kernels

} // namespace dragon
