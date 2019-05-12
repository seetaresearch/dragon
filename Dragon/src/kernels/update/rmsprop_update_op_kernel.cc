#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template <> void RMSPropUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float gi = g[i];
        float hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (std::sqrt(hi) + eps);
    }
}

}  // namespace kernel

}  // namepsace dragon