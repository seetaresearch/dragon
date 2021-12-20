#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void Adam<float, CPUContext>(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    float* g,
    float* m,
    float* v,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = lr * mi / (std::sqrt(vi) + eps);
  }
}

template <>
void AdamW<float, CPUContext>(
    const int N,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const float* x,
    float* g,
    float* m,
    float* v,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float gi = g[i];
    float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = lr * mi / (std::sqrt(vi) + eps) + wd * x[i];
  }
}

} // namespace kernels

} // namespace dragon
