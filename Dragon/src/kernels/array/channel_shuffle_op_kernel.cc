#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _ChannelShuffle(
    const int               outer_dim,
    const int               inner_dim,
    const int               G,
    const int               K,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    int64_t x_ofs, y_ofs;
    for (int n = 0; n < outer_dim; ++n) {
        for (int gi = 0; gi < G; ++gi) {
            for (int ki = 0; ki < K; ++ki) {
                x_ofs = ((n * G + gi) * K + ki) * inner_dim;
                y_ofs = ((n * K + ki) * G + gi) * inner_dim;
                math::Copy(
                    inner_dim,
                    x + x_ofs,
                    y + y_ofs, ctx
                );
            }
        }
    }
}

/* Kernel Launchers */

#define DEFINE_SHUFFLE_KERNEL_LAUNCHER(T) \
    template <> void ChannelShuffle<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               group, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _ChannelShuffle( \
            outer_dim, \
            inner_dim, \
            group, \
            axis_dim / group, \
            x, y, ctx \
        ); \
    }

DEFINE_SHUFFLE_KERNEL_LAUNCHER(bool);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int8_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(uint8_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(int64_t);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(float16);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(float);
DEFINE_SHUFFLE_KERNEL_LAUNCHER(double);

#undef DEFINE_SHUFFLE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon