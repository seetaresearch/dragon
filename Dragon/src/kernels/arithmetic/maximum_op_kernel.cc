#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! MaximumE <T = float32, Device = CPU> */

template <> void MaximumE<float, CPUContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x1[i], x2[i]);
    }
}

/*! MaximumB <T = float32, Device = CPU> */

template <> void MaximumB<float, CPUContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x1[i], x2);
    }
}

/*! MaximumEGrad <T = float32, Device = CPU> */

template <> void MaximumEGrad<float, CPUContext>(
    const int               count,
    const float*            x1,
    const float*            x2,
    const float*            dy,
    float*                  dx1,
    float*                  dx2,
    CPUContext*             ctx) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const bool dy_to_dx1 = x1[i] > x2[i];
        dx1[i] = dy_to_dx1 ? dy[i] : 0;
        dx2[i] = dy_to_dx1 ? 0 : dy[i];
    }
}

/*! MaximumBGrad <T = float32, Device = CPU> */

template <> void MaximumBGrad<float, CPUContext>(
    const int               count,
    const float*            x1,
    const float             x2,
    const float*            dy,
    float*                  dx1,
    /* float*                  dx2, */
    CPUContext*             ctx) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx1[i] = (x1[i] > x2) ? dy[i] : 0;
    }
}

}  // namespace kernel

}  // namepsace dragon