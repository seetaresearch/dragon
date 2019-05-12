#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! GradientTwoSum <T = ?, Device = CUDA> */

template <typename T>
void _GradientTwoSum(
    const int               count,
    const T*                dy1,
    const T*                dy2,
    T*                      dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] += (dy1[i] + dy2[i]);
    }
}

/*! Kernel Launchers */

#define DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(T) \
    template <> void GradientTwoSum<T, CPUContext>( \
        const int               count, \
        const T*                dy1, \
        const T*                dy2, \
        T*                      dx, \
        CPUContext*             ctx) { \
        _GradientTwoSum(count, dy1, dy2, dx); \
    }

DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int8_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(uint8_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(int64_t);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(float);
DEFINE_GRAD_SUM2_KERNEL_LAUNCHER(double);

template <> void GradientTwoSum<float16, CPUContext>(
    const int               count,
    const float16*          dy1,
    const float16*          dy2,
    float16*                dx,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_GRAD_SUM2_KERNEL_LAUNCHER

}  // namespace kernel

}  // namespace dragon