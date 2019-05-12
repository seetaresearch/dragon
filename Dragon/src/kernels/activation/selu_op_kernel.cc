#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template<> void SElu<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = 1.0507f * std::max(x[i], 0.f)
             + 1.7581f * (std::exp(std::min(x[i], 0.f)) - 1.f);
    }
}

/* <T = float16, Device = CPU> */

template<> void SElu<float16, CPUContext>(
    const int               count,
    const float16*          x,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/* <T = float32, Device = CPU> */

template<> void SEluGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = y[i] > 0 ? 1.0507f * dy[i] :
            (1.7581f + y[i]) * dy[i];
    }
}

/* <T = float16, Device = CPU> */

template<> void SEluGrad<float16, CPUContext>(
    const int               count,
    const float16*          dy,
    const float16*          y,
    float16*                dx,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon