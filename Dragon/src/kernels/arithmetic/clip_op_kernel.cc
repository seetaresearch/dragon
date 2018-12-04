#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Clip <T = float32, Device = CPU> */

template <> void Clip<float, CPUContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(low, std::min(x[i], high));
    }
}

/*! ClipGrad <T = float32, Device = CPU> */

template <> void ClipGrad<float, CPUContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float xi = x[i];
        dx[i] = (xi < low || xi > high) ? 0 : dy[i];
    }
}

}  // namespace kernel

}  // namepsace dragon