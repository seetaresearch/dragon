#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! RMSPropUpdate <T = float32, Device = CPU> */

template <> void RMSPropUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float gi = g[i];
        float hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (std::sqrt(hi) + eps);
    }
}

/*! RMSPropUpdate <T = float32, Device = CPU> */

template <> void RMSPropUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float16*                g,
    float16*                h,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon