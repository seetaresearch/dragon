#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Sigmoid <T = float32, Device = CPU> */

template <typename T>
T _SigmoidUnit(T x) { return T(1) / (T(1) + exp(-x)); }

template<> void Sigmoid<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = _SigmoidUnit<float>(x[i]);
    }
}

/*! SigmoidGrad <T = float32, Device = CPU> */

template<> void SigmoidGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * y[i] * (1 - y[i]);
    }
}

}  // namespace kernel

}  // namepsace dragon