#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! NesterovUpdate <T = float32, Device = CPU> */

template <> void NesterovUpdate<float, CPUContext>(
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
        float hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

/*! NesterovUpdate <T = float16, Device = CPU> */

template <> void NesterovUpdate<float16, CPUContext>(
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