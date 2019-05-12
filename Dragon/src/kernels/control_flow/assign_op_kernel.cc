#include "utils/op_kernel.h"
#include "utils/math_utils.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Assign(
    const int               count,
    const int               ndims,
    const int*              x_dims,
    const int*              y_strides,
    const int*              starts,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int yi;
    for (int xi = 0; xi < count; ++xi) {
        yi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            yi += (
                index[d] + starts[d]
            ) * y_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, x_dims, index.data()
        );
    }
}

/* Kernel Launchers */

#define DEFINE_ASSIGN_KERNEL_LAUNCHER(T) \
    template<> void Assign<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              y_strides, \
        const int*              starts, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _Assign( \
            count, \
            ndims, \
            x_dims, \
            y_strides, \
            starts, \
            x, y \
        ); \
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