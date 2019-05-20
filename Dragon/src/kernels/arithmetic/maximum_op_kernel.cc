#include "utils/op_kernel.h"
#include "utils/eigen_utils.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Maximum(
    const int               count,
    const T*                a,
    const T*                b,
    T*                      y) {
    EigenVectorArrayMap<T>(y, count) = \
        ConstEigenVectorArrayMap<T>(a, count).max(
            ConstEigenVectorArrayMap<T>(b, count));
}

/* <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMaximum(
    const int               count,
    const T*                a,
    const T                 b,
    T*                      y) {
    EigenVectorArrayMap<T>(y, count) = \
        ConstEigenVectorArrayMap<T>(a, count).max(b);
}

/* <T = ?, Device = CPU> */

template <typename T>
void _MaximumGrad(
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
        const bool dy_to_da = a[i] > b[i];
        da[i] = dy_to_da ? dy[i] : kZero;
        db[i] = dy_to_da ? kZero : dy[i];
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMaximumGrad(
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
        da[i] = a[i] > b ? dy[i] : kZero;
    }
}

/* Kernel Launchers */

#define DEFINE_MAXIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                a, \
        const T2                b, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(count, a, b, y); \
    }

#define DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
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

DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int8_t, int8_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, uint8_t, uint8_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int, int*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, int64_t, int64_t*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, float, float*);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(Maximum, double, double*);

DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int8_t, int8_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, uint8_t, uint8_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int, int);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, int64_t, int64_t);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, float, float);
DEFINE_MAXIMUM_KERNEL_LAUNCHER(BroadcastMaximum, double, double);

DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int8_t, int8_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, uint8_t, uint8_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int, int*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, int64_t, int64_t*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, float, float*);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(MaximumGrad, double, double*);

DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int8_t, int8_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, uint8_t, uint8_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int, int);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, int64_t, int64_t);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, float, float);
DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(BroadcastMaximumGrad, double, double);

template <> void Maximum<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMaximum<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MaximumGrad<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16*          b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMaximumGrad<float16, CPUContext>(
    const int               count,
    const float16*          a,
    const float16           b,
    const float16*          dy,
    float16*                da,
    float16*                db,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_MAXIMUM_KERNEL_LAUNCHER
#undef DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon