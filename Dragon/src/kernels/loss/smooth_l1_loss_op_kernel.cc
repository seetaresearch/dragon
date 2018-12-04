#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! SmoothL1 <T = float32, Device = CPU> */

template<> void SmoothL1<float, CPUContext>(
    const int               count,
    const float             beta,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = x[i];
        const float abs_val = abs(val);
        if (abs_val < beta) y[i] = 0.5f * val * val / beta;
        else y[i] = abs_val - 0.5f * beta;
    }
}

/*! SmoothL1Grad <T = float32, Device = CPU> */

template<> void SmoothL1Grad<float, CPUContext>(
    const int               count,
    const float             beta,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = dy[i];
        const float abs_val = abs(val);
        if (abs_val < beta) dx[i] = val / beta;
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        else dx[i] = (float)((val > 0.f) - (val < 0.f));
    }
}

}  // namespace kernel

}  // namepsace dragon