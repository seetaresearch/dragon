#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! SigmoidCrossEntropy <T = float32, Device = CPU> */

template <> void SigmoidCrossEntropy<float, CPUContext>(
    const int               count,
    const float*            logit,
    const float*            target,
    float*                  loss,
    int*                    flag,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (target[i] < 0) {
            loss[i] = flag[i] = 0;
        } else {
            loss[i] = std::log(
                1.f + std::exp(
                    logit[i] - 2.f * logit[i] * (
                            logit[i] >= 0
                        )
                )
            ) + logit[i] * (
                (logit[i] >= 0) - target[i]
            );
            flag[i] = 1;
        }
    }
}

/*! SigmoidCrossEntropyGrad <T = float32, Device = CPU> */

template <> void SigmoidCrossEntropyGrad<float, CPUContext>(
    const int               count,
    const float*            logit,
    const float*            target,
    float*                  dlogit,
    int*                    flag,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (target[i] < 0) {
            dlogit[i] = flag[i] = 0;
        } else {
            dlogit[i] = 1.f / (
                1.f + std::exp(-logit[i])
            ) - target[i];
            flag[i] = 1;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon