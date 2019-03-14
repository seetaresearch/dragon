#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Repeat <T = ?, Device = CPU> */

template <typename T>
void _Repeat(
    const int               outer_dim,
    const int               repeat_dim,
    const int               inner_dim,
    const int               repeats,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < repeat_dim; ++j) {
            for (int k = 0; k < repeats; ++k) {
                ctx->Copy<T, CPUContext, CPUContext>(
                    inner_dim, y, x); y += inner_dim;
            } x += inner_dim;
        }
    }
}

/*! RepeatGrad <T = ?, Device = CPU> */

template <typename T>
void _RepeatGrad(
    const int               outer_dim,
    const int               repeat_dim,
    const int               inner_dim,
    const int               repeats,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < repeat_dim; ++j) {
            ctx->Copy<T, CPUContext, CPUContext>(
                inner_dim, dx, dy); dy += inner_dim;
            for (int k = 1; k < repeats; ++k) {
                math::Add<T, CPUContext>(
                    inner_dim, dy, dx, dx, ctx);
                dy += inner_dim;
            } dx += inner_dim;
        }
    }
} 

#define DEFINE_REPEAT_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               repeat_dim, \
        const int               inner_dim, \
        const int               repeats, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name<T>(outer_dim, repeat_dim, \
            inner_dim, repeats, x, y, ctx); \
    }

DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, bool);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, int8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, uint8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, int);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, int64_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, float16);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, float);
DEFINE_REPEAT_KERNEL_LAUNCHER(Repeat, double);

DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, int8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, uint8_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, int);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, int64_t);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, float16);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, float);
DEFINE_REPEAT_KERNEL_LAUNCHER(RepeatGrad, double);

#undef DEFINE_REPEAT_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon