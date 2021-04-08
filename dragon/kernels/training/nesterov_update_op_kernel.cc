#include "dragon/utils/device/common_openmp.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

template <>
void NesterovUpdate<float, CPUContext>(
    const int N,
    const float lr,
    const float momentum,
    float* g,
    float* m,
    CPUContext* ctx) {
  for (int i = 0; i < N; ++i) {
    float mi = m[i];
    float mi_new = m[i] = momentum * mi + lr * g[i];
    g[i] = (1 + momentum) * mi_new - momentum * mi;
  }
}

} // namespace kernels

} // namespace dragon
