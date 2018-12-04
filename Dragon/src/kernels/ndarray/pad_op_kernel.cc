#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! ConstPad1d <T = float32, Device = CPU> */

template <> void ConstPad1d<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float             value,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = ex_d - pad_l;
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) y_ptr[i] = value;
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            ctx->Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

/*! ReflectPad1d <T = float32, Device = CPU> */

template <> void ReflectPad1d<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        int d = ex_d - pad_l;
        d = std::max(d, -d);
        d = std::min(d, 2 * dim - d - 2);
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) 
                y_ptr[i] = x[(o * dim + d) * inner_dim + i];
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            ctx->Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

/*! EdgePad1d <T = float32, Device = CPU> */

template <> void EdgePad1d<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = std::min(dim - 1, std::max(ex_d - pad_l, 0));
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) 
                y_ptr[i] = x[(o * dim + d) * inner_dim + i];
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            ctx->Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

/*! ConstPad1dGrad <T = float32, Device = CPU> */

template <> void ConstPad1dGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int d = idx % dim;
        const int o = idx / dim;
        const int ex_d = d + pad_l;
        const float* dy_ptr = dy + (o * ex_dim + ex_d) * inner_dim;
        float* dx_ptr = dx + (o * dim + d) * inner_dim;
        ctx->Copy<float, CPUContext, CPUContext>(
            inner_dim, dx_ptr, dy_ptr);
    }
}

/*! ReflectPad1dGrad <T = float32, Device = CPU> */

template <> void ReflectPad1dGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    for (int idx = 0; idx < count; ++idx) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = std::max(d, -d);
        d = std::min(d, 2 * dim - d - 2);
        dx[(o * dim + d) * inner_dim + i] += dy[idx];
    }
}

/*! EdgePad1dGrad <T = float32, Device = CPU> */

template <> void EdgePad1dGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    const int count_v2 = count / inner_dim;
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = std::min(dim - 1, std::max(ex_d - pad_l, 0));
        const float* dy_ptr = dy + (o * ex_dim + ex_d) * inner_dim;
        if (d == 0 || d == dim - 1) {
            for (int i = 0; i < inner_dim; ++i)
                dx[(o * dim + d) * inner_dim + i] += dy_ptr[i];
        } else {
            float* dx_ptr = dx + (o * dim + d) * inner_dim;
            ctx->Copy<float, CPUContext, CPUContext>(
                inner_dim, dx_ptr, dy_ptr);
        }
    }
}

}  // namespace kernel

}  // namepsace dragon