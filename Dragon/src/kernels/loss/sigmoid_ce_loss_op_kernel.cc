#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! SigmoidCrossEntropy <T = float32, Device = CPU> */

template <> void SigmoidCrossEntropy<float, CPUContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  losses,
    float*                  flags,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (targets[i] < 0) {
            losses[i] = flags[i] = 0;
        } else {
            losses[i] = std::log(
                1 + std::exp(logits[i] - 2 * logits[i] * (logits[i] >= 0))
            ) + logits[i] * ((logits[i] >= 0) - targets[i]);
            flags[i] = 1;
        }
    }
}

/*! SigmoidCrossEntropyGrad <T = float32, Device = CPU> */

template <> void SigmoidCrossEntropyGrad<float, CPUContext>(
    const int               count,
    const float*            logits,
    const float*            targets,
    float*                  dlogits,
    float*                  flags,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (targets[i] < 0) {
            dlogits[i] = flags[i] = 0;
        } else {
            dlogits[i] = 1 / (1 + std::exp(-logits[i])) - targets[i];
            flags[i] = 1;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon