#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float16, Device = CPU> */

template <> void MixedPrecL2Decay<float16, CPUContext>(
    const int               count,
    const float             alpha,
    const float16*          w,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] += (cast::to<float>(w[i]) * alpha);
    }
}

/* <T = float16, Device = CPU> */

template <> void MixedPrecUpdate<float16, CPUContext>(
    const int               count,
    const float*            updates,
    float16*                w,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        w[i] = cast::to<float16>(
            cast::to<float>(
                w[i]) - updates[i]);
    }
}

}  // namespace kernel

}  // namepsace dragon