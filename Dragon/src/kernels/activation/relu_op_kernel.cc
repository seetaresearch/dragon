#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Relu <T = float32, Device = CPU> */

template<> void Relu<float, CPUContext>(
    const int               count,
    const float             slope,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], 0.f) + slope * std::min(x[i], 0.f);
    }
}

/*! Relu <T = float16, Device = CPU> */

template<> void Relu<float16, CPUContext>(
    const int               count,
    const float             slope,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ReluGrad <T = float32, Device = CPU> */

template<> void ReluGrad<float, CPUContext>(
    const int               count,
    const float             slope,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * ((y[i] > 0) + slope * (y[i] <= 0));
    }
}

}  // namespace kernel

}  // namepsace dragon