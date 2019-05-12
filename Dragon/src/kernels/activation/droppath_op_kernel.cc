#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CPU> */

template<> void DropPath<float, CPUContext>(
    const int               rows,
    const int               cols,
    const float             scale,
    const float*            x,
    const float*            mask,
    float*                  y,
    CPUContext*             ctx) {
    auto count = rows * cols;
    auto thresh = 1.f - (1.f / scale);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = x[i] * (float)(
            mask[i / cols] > thresh
                ) * scale;
    }
}

/* <T = float16, Device = CPU> */

template<> void DropPath<float16, CPUContext>(
    const int               rows,
    const int               cols,
    const float             scale,
    const float16*          x,
    const float*            mask,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon