#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Where(
    const int               count,
    const uint8_t*          mask,
    const T*                a,
    const T*                b,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = mask[i] ? a[i] : b[i];
    }
}

template <typename T>
void _WhereGrad(
    const int               count,
    const uint8_t*          mask,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const bool dy_to_da = mask[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
}

/* Kernel Launchers */

#define DEFINE_WHERE_KERNEL_LAUNCHER(T) \
    template<> void Where<T, CPUContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                a, \
        const T*                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        _Where(count, mask, a, b, y); \
    }

#define DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(T) \
    template <> void WhereGrad<T, CPUContext>( \
        const int               count, \
        const uint8_t*          mask, \
        const T*                dy, \
        T*                      da, \
        T*                      db, \
        CPUContext*             ctx) { \
        _WhereGrad(count, mask, dy, da, db); \
    }

DEFINE_WHERE_KERNEL_LAUNCHER(bool);
DEFINE_WHERE_KERNEL_LAUNCHER(int8_t);
DEFINE_WHERE_KERNEL_LAUNCHER(uint8_t);
DEFINE_WHERE_KERNEL_LAUNCHER(int);
DEFINE_WHERE_KERNEL_LAUNCHER(int64_t);
DEFINE_WHERE_KERNEL_LAUNCHER(float16);
DEFINE_WHERE_KERNEL_LAUNCHER(float);
DEFINE_WHERE_KERNEL_LAUNCHER(double);

DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(bool);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_WHERE_GRAD_KERNEL_LAUNCHER(double);

template <> void WhereGrad<float16, CPUContext>(
    const int               count,
    const uint8_t*          mask,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_WHERE_KERNEL_LAUNCHER
#undef DEFINE_WHERE_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon