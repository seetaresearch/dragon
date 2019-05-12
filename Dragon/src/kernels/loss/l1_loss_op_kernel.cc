#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! AbsGrad <T = float32, Device = CPU> */

template<> void AbsGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = dy[i];
        // val > 0: 1 | val == 0: 0 | val < 0: -1
        dx[i] = (float)((val > 0.f) - (val < 0.f));
    }
}

}  // namespace kernel

}  // namepsace dragon