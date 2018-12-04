#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Crop1d <T = ?, Device = CPU> */

template <typename T>
void _Crop1d(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const T*                x,
    T*                      y,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const T* x_ptr = x + (o * dim + ex_d + start) * inner_dim;
        T* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        ctx->Copy<T, CPUContext, CPUContext>(
            inner_dim, y_ptr, x_ptr);
    }
}

/*! Crop1d <T = float32, Device = CPU> */

template<> void Crop1d<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    _Crop1d<float>(count, dim, ex_dim,
        inner_dim, start, x, y, ctx);
}

/*! Crop1d <T = int32, Device = CPU> */

template<> void Crop1d<int, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int*              x,
    int*                    y,
    CPUContext*             ctx) {
    _Crop1d<int>(count, dim, ex_dim,
        inner_dim, start, x, y, ctx);
}

/*! Crop1dGrad <T = ?, Device = CPU> */

template <typename T>
void _Crop1dGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const T*                dy,
    T*                      dx,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int d = idx % dim;
        const int o = idx / dim;
        T* dx_ptr = dx + (o * dim + d) * inner_dim;
        if (d < start || d >= end) {
            for (int i = 0; i < inner_dim; ++i) dx_ptr[i] = 0;
        } else {
            const T* dy_ptr = dy + (o * ex_dim + d - start) * inner_dim;
            ctx->Copy<T, CPUContext, CPUContext>(
                inner_dim, dx_ptr, dy_ptr);
        }
    }
}

/*! Crop1dGrad <T = float32, Device = CPU> */

template<> void Crop1dGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    _Crop1dGrad<float>(
        count, dim, ex_dim, inner_dim,
        start, end, dy, dx, ctx);
}

/*! Crop1dGrad <T = int32, Device = CPU> */

template<> void Crop1dGrad<int, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const int*              dy,
    int*                    dx,
    CPUContext*             ctx) {
    _Crop1dGrad<int>(
        count, dim, ex_dim, inner_dim,
            start, end, dy, dx, ctx);
}

}  // namespace kernel

}  // namepsace dragon