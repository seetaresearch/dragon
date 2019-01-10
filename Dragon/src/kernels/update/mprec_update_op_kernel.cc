#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! MixedPrecisionUpdate <T = float16, Device = CPU> */

template <> void MixedPrecisionUpdate<float16, CPUContext>(
    const int               count,
    const float*            updates,
    float16*                w,
    float16*                g,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float master_wi = cast::to<float>(w[i]) - updates[i];
        w[i] = cast::to<float16>(master_wi);
        g[i] = cast::to<float16>(0.f);
    }
}

}  // namespace kernel

}  // namepsace dragon