#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! SGDUpdate <T = float32, Device = CPU> */

template <> void SGDUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float hi = h[i];
        g[i] = h[i] = momentum * hi + lr * g[i];
    }
}

/*! SGDUpdate <T = float16, Device = CPU> */

template <> void SGDUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon