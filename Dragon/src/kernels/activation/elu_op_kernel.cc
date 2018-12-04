#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Elu <T = float32, Device = CPU> */

template<> void Elu<float, CPUContext>(
    const int               count,
    const float             alpha,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], 0.f) + alpha *
            (std::exp(std::min(x[i], 0.f)) - 1.f);
    }
}

/*! EluGrad <T = float32, Device = CPU> */

template<> void EluGrad<float, CPUContext>(
    const int               count,
    const float             alpha,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * (
            (y[i] > 0) + (alpha + y[i]) * (y[i] <= 0)
        );
    }
}

}  // namespace kernel

}  // namepsace dragon