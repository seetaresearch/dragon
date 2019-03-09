#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! AdamUpdate <T = float32, Device = CPU> */

template <> void AdamUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float*                  g,
    float*                  m,
    float*                  v,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float gi = g[i];
        float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        g[i] = lr * mi / (std::sqrt(vi) + eps);
    }
}

}  // namespace kernel

}  // namepsace dragon