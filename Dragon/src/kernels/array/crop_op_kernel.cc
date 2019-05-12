#include "utils/op_kernel.h"
#include "utils/math_utils.h"

namespace dragon {

namespace kernel {

/* <T = ?, Device = CPU> */

template <typename T>
void _Crop(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                x,
    T*                      y) {
    vec32_t index(ndims, 0); int xi;
    for (int yi = 0; yi < count; ++yi) {
        xi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            xi += (
                index[d] + starts[d]
            ) * x_strides[d];
        }
        y[yi] = x[xi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* <T = ?, Device = CPU> */

template <typename T>
void _CropGrad(
    const int               count,
    const int               ndims,
    const int*              x_strides,
    const int*              y_dims,
    const int*              starts,
    const T*                dy,
    T*                      dx) {
    vec32_t index(ndims, 0); int xi;
    for (int yi = 0; yi < count; ++yi) {
        xi = 0;
        for (int d = ndims - 1; d >= 0; --d) {
            xi += (
                index[d] + starts[d]
            ) * x_strides[d];
        }
        dx[xi] = dy[yi];
        utils::IncreaseIndexInDims(
            ndims, y_dims, index.data()
        );
    }
}

/* Kernel Launchers */

#define DEFINE_CROP_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CPUContext>( \
        const int               count, \
        const int               ndims, \
        const int*              x_strides, \
        const int*              y_dims, \
        const int*              starts, \
        const T*                x, \
        T*                      y, \
        CPUContext*             ctx) { \
        _##name( \
            count, ndims, \
            x_strides, y_dims, \
            starts, x, y \
        ); \
    }

DEFINE_CROP_KERNEL_LAUNCHER(Crop, bool);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int8_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, uint8_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, int64_t);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, float16);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, float);
DEFINE_CROP_KERNEL_LAUNCHER(Crop, double);

DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, bool);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int8_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, uint8_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, int64_t);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, float16);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, float);
DEFINE_CROP_KERNEL_LAUNCHER(CropGrad, double);

#undef DEFINE_CROP_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon