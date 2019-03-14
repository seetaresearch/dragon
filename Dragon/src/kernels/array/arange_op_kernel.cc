#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Arange <T = ?, Device = CPU> */

template <typename T>
void _Arange(
    const int               count,
    const int               start,
    const int               step,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = static_cast<T>(start + i * step);
    }
}

#define DEFINE_ARANGE_KERNEL_LAUNCHER(T) \
    template <> void Arange<T, CPUContext>( \
        const int               count, \
        const int               start, \
        const int               step, \
        T*                      y, \
        CPUContext*             ctx) { \
        _Arange<T>(count, start, step, y); \
    }

DEFINE_ARANGE_KERNEL_LAUNCHER(int8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(uint8_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(int);
DEFINE_ARANGE_KERNEL_LAUNCHER(int64_t);
DEFINE_ARANGE_KERNEL_LAUNCHER(float);
DEFINE_ARANGE_KERNEL_LAUNCHER(double);

/*! Arange <T = float16, Device = CPU> */

template <> void Arange<float16, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    float16*                y,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = cast::to<float16>(
            cast::to<float>(start + i * step));
    }
}

#undef DEFINE_ARANGE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon