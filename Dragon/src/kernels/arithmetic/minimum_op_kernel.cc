#include "utils/op_kernel.h"
#include "utils/eigen_utils.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Minimum(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y) {
    EigenVectorArrayMap<T>(y, count) = \
        ConstEigenVectorArrayMap<T>(a, count).min(
            ConstEigenVectorArrayMap<T>(b, count));
}

/* <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMinimum(
    const int               count,
    const T*                a,
    const T                 b,
    T*                      y) {
    EigenVectorArrayMap<T>(y, count) = \
        ConstEigenVectorArrayMap<T>(a, count).min(b);
}

/* <T = float32, Device = CPU> */

template <typename T>
void _MinimumGrad(
    const int               count,
    const T*                a,
    const T*                b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const bool dy_to_da = a[i] < b[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
}

/* <T = float32, Device = CPU> */

template <typename T>
void _BroadcastMinimumGrad(
    const int               count,
    const T*                a,
    const T                 b,
    const T*                dy,
    T*                      da,
    T*                      db) {
    const T kZero = T(0);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        da[i] = a[i] < b ? dy[i] : kZero;
    }
}

/* Kernel Launchers */

#define DEFINE_MINIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                a, \
        const T2                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(count, a, b, y); \
    }

#define DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                a, \
        const T2                b, \
        const T*                dy, \
        T*                      da, \
        T*                      db, \
        CPUContext*             ctx) { \
        _##name(count, a, b, dy, da, db); \
    }

DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int8_t, int8_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, uint8_t, uint8_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int, int*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, int64_t, int64_t*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, float, float*);
DEFINE_MINIMUM_KERNEL_LAUNCHER(Minimum, double, double*);

DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int8_t, int8_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, uint8_t, uint8_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int, int);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, int64_t, int64_t);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, float, float);
DEFINE_MINIMUM_KERNEL_LAUNCHER(BroadcastMinimum, double, double);

DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int8_t, int8_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, uint8_t, uint8_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int, int*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, int64_t, int64_t*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, float, float*);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(MinimumGrad, double, double*);

DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int8_t, int8_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, uint8_t, uint8_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int, int);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, int64_t, int64_t);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, float, float);
DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMinimumGrad, double, double);

template <> void Minimum<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMinimum<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MinimumGrad<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMinimumGrad<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_MINIMUM_KERNEL_LAUNCHER
#undef DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon