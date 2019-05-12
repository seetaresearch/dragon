#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Clip(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    T*                      y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(low, std::min(x[i], high));
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _ClipGrad(
    const int               count,
    const T                 low,
    const T                 high,
    const T*                x,
    const T*                dy,
    T*                      dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const T xi = x[i];
        dx[i] = (xi < low || xi > high) ? T(0) : dy[i];
    }
}

/* Kernel Launchers */

#define DEFINE_CLIP_KERNEL_LAUNCHER(T) \
    template <> void Clip<T, CPUContext>( \
        const int               count, \
        const float             low, \
        const float             high, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _Clip( \
            count, \
            cast::to<T>(low), \
            cast::to<T>(high), \
            x, y \
        ); \
    }

#define DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(T) \
    template <> void ClipGrad<T, CPUContext>( \
        const int               count, \
        const float             low, \
        const float             high, \
        const T*                x, \
        const T*                dy, \
        T*                      dx, \
        CPUContext*             ctx) { \
        _ClipGrad( \
            count, \
            cast::to<T>(low), \
            cast::to<T>(high), \
            x, dy, dx \
        ); \
    }

DEFINE_CLIP_KERNEL_LAUNCHER(int8_t);
DEFINE_CLIP_KERNEL_LAUNCHER(uint8_t);
DEFINE_CLIP_KERNEL_LAUNCHER(int);
DEFINE_CLIP_KERNEL_LAUNCHER(int64_t);
DEFINE_CLIP_KERNEL_LAUNCHER(float);
DEFINE_CLIP_KERNEL_LAUNCHER(double);

DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int8_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(int64_t);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(float);
DEFINE_CLIP_GRAD_KERNEL_LAUNCHER(double);

/* <T = float16, Device = CPU> */

template <> void Clip<float16, CPUContext>(
    const int               count,
    const float             low,
    const float             high,
    const float16*          x,
    float16*                  y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/* <T = float16, Device = CPU> */

template <> void ClipGrad<float16, CPUContext>(
    const int               count,
    const float             low,
    const float             high,
    const float16*          x,
    const float16*          dy,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_CLIP_KERNEL_LAUNCHER
#undef DEFINE_CLIP_GRAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon