#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Sum <T = float32, Device = CPU> */

template<> void Sum<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float sum_val = 0.f;
        for (int j = 0; j < axis_dim; ++j)
            sum_val += x[(i / inner_dim * axis_dim + j)
                          * inner_dim + i % inner_dim];
        y[i] = sum_val;
    }
}

/*! SumGrad <T = float32, Device = CPU> */

template<> void SumGrad<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             coeff,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j)
            dx[(i / inner_dim * axis_dim + j)
                * inner_dim + i % inner_dim] = dy[i] * coeff;
    }
}

}  // namespace kernel

}  // namepsace dragon