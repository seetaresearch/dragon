#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template <> void NesterovUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float hi = h[i];
        float hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

}  // namespace kernel

}  // namepsace dragon