#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Minimum <T = ?, Device = CPU> */

template <typename T>
void _Minimum(
    const int               count,
    const T*                x1,
    const T*                x2,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::min(x1[i], x2[i]);
    }
}

/*! BroadcastMinimum <T = ?, Device = CPU> */

template <typename T>
void _BroadcastMinimum(
    const int               count,
    const T*                x1,
    const T                 x2,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::min(x1[i], x2);
    }
}

/*! MinimumGrad <T = float32, Device = CPU> */

template <typename T>
void _MinimumGrad(
    const int               count,
    const T*                x1,
    const T*                x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const bool dy_to_dx1 = x1[i] < x2[i];
        dx1[i] = dy_to_dx1 ? dy[i] : 0;
        dx2[i] = dy_to_dx1 ? 0 : dy[i];
    }
}

/*! BroadcastMinimumGrad <T = float32, Device = CPU> */

template <typename T>
void _BroadcastMinimumGrad(
    const int               count,
    const T*                x1,
    const T                 x2,
    const T*                dy,
    T*                      dx1,
    T*                      dx2) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx1[i] = (x1[i] < x2) ? dy[i] : 0;
    }
}

/*! Kernel Launchers */

#define DEFINE_MINIMUM_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name<T>(count, x1, x2, y); \
    }

#define DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER(name, T, T2) \
    template <> void name<T, CPUContext>( \
        const int               count, \
        const T*                x1, \
        const T2                x2, \
        const T*                dy, \
        T*                      dx1, \
        T*                      dx2, \
        CPUContext*             ctx) { \
        _##name<T>(count, x1, x2, dy, dx1, dx2); \
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
    const float16*          x1,
    const float16*          x2,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMinimum<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void MinimumGrad<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16*          x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void BroadcastMinimumGrad<float16, CPUContext>(
    const int               count,
    const float16*          x1,
    const float16           x2,
    const float16*          dy,
    float16*                dx1,
    float16*                dx2,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_MINIMUM_KERNEL_LAUNCHER
#undef DEFINE_MINIMUM_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon