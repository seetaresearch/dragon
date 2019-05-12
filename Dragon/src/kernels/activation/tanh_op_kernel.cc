#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template<> void Tanh<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::tanh(x[i]);
    }
}

/* <T = float32, Device = CPU> */

template<> void TanhGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

}  // namespace kernel

}  // namepsace dragon