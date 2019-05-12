#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Maximum(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x1[i], x2[i]);
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMaximum(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x1[i], x2);
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _MaximumGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const bool dy_to_dx1 = x1[i] > x2[i];
        dx1[i] = dy_to_dx1 ? dy[i] : 0;
        dx2[i] = dy_to_dx1 ? 0 : dy[i];
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMaximumGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx1[i] = (x1[i] > x2) ? dy[i] : 0;
    }
}

/* Kernel Launchers */

#define DEFINE_MAXIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name(count, x1, x2, y); \
    }

#define DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        const T*                dy, \
        T*                      dx1, \
        T*                      dx2, \
        CPUContext*             ctx) { \
        _##name(count, x1, x2, dy, dx1, dx2); \
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
    const float16*          x1,
    const float16*          x2,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMaximum<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MaximumGrad<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16*          x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMaximumGrad<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_MAXIMUM_KERNEL_LAUNCHER
#undef DEFINE_MAXIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon