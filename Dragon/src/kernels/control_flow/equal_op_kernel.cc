 #include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Equal <T = float32, Device = CPU> */

template <> void Equal<float, CPUContext>(
    const int               count,
    const float*            a,
    const float*            b,
    float*                  y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = fabs(a[i] - b[i]) < FLT_EPSILON ? 1.f : 0.f;
    }
}

}  // namespace kernel

}  // namepsace dragon