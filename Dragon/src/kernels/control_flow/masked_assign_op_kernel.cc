#include "utils/op_kernel.h"
#include "utils/math_utils.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _MaskedAssign(
    const int               count,
    const uint8_t*          mask,
    const T*                x,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = mask[i] ? x[i] : y[i];
    }
}

/* Kernel Launchers */

#define DEFINE_ASSIGN_KERNEL_LAUNCHER(T) \
    template<> void MaskedAssign<T, CPUContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _MaskedAssign(count, mask, x, y); \
    }

DEFINE_ASSIGN_KERNEL_LAUNCHER(bool);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int8_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(uint8_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int);
DEFINE_ASSIGN_KERNEL_LAUNCHER(int64_t);
DEFINE_ASSIGN_KERNEL_LAUNCHER(float16);
DEFINE_ASSIGN_KERNEL_LAUNCHER(float);
DEFINE_ASSIGN_KERNEL_LAUNCHER(double);

#undef DEFINE_ASSIGN_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon