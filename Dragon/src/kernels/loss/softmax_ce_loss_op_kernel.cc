#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! SoftmaxCrossEntropy <T = float32, Device = CPU> */

template <> void SoftmaxCrossEntropy<float, CPUContext>(
    const int               count,
    const float*            prob,
    const float*            target,
    float*                  loss,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        loss[i] = - target[i] * std::log(std::max(prob[i], FLT_MIN));
    }
}

}  // namespace kernel

}  // namepsace dragon