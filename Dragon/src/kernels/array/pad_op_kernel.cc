#include "utils/op_kernel.h"
#include "utils/cast.h"
#include "utils/math_utils.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _ConstPad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T                 value,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi, d, r;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (d = ndims - 1; d >= 0; --d) {
            r = index[d] - l_pads[d];
            if (r < 0 || r >= x_dims[d]) break;
            xi += r * x_strides[d];
        }
        y[yi] = d >= 0 ? value : x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _ReflectPad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi, d, r;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (d = ndims - 1; d >= 0; --d) {
            r = index[d] - l_pads[d];
            r = std::max(r, -r);
            r = std::min(r, 2 * x_dims[d] - r - 2);
            xi += r * x_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _EdgePad(
    const int               nthreads,
    const int               ndims,
    const int*              x_dims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              l_pads,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi, d, r;
    for (int yi = 0; yi < nthreads; ++yi) {
        xi = 0;
        for (d = ndims - 1; d >= 0; --d) {
            r = std::min(
                x_dims[d] - 1,
                std::max(index[d] - l_pads[d], 0)
            );
            xi += r * x_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* Kernel Launchers */

#define DEFINE_CONST_PAD_KERNEL_LAUNCHER(T) \
    template<> void ConstPad<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              l_pads, \
        const float             value, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _ConstPad( \
            count, ndims, \
            x_dims, x_strides, \
            y_dims, l_pads, \
            cast::to<T>(value), \
            x, y \
        ); \
    }

#define DEFINE_PAD_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_dims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              l_pads, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name( \
            count, ndims, \
            x_dims, x_strides, \
            y_dims, l_pads, \
            x, y \
        ); \
    }

DEFINE_CONST_PAD_KERNEL_LAUNCHER(bool);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int8_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(uint8_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(int64_t);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(float16);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(float);
DEFINE_CONST_PAD_KERNEL_LAUNCHER(double);

DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, bool);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int8_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, uint8_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, int64_t);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, float16);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, float);
DEFINE_PAD_KERNEL_LAUNCHER(ReflectPad, double);

DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, bool);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int8_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, uint8_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, int64_t);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, float16);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, float);
DEFINE_PAD_KERNEL_LAUNCHER(EdgePad, double);

#undef DEFINE_PAD_KERNEL_LAUNCHER
#undef DEFINE_CONST_PAD_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon