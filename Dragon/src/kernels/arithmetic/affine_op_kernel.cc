#include "utils/op_kernel.h"
#include "utils/eigen_utils.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Affine <T = float32, Device = CPU> */

template<> void Affine<float, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    float*                  y,
    CPUContext*             ctx) {
    const auto* X = x; auto* Y = y;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            if (beta != nullptr) {
                EigenVectorArrayMap<float>(Y, inner_dim)
                    = ConstEigenVectorArrayMap<float>(
                        X, inner_dim) * alpha[d] + beta[d];
            } else {
                EigenVectorArrayMap<float>(Y, inner_dim)
                    = ConstEigenVectorArrayMap<float>(
                        X, inner_dim) * alpha[d];
            }
            X += inner_dim; Y += inner_dim;
        }
    }
}

/*! Affine <T = float16, Device = CPU> */

template<> void Affine<float16, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}


/*! AffineGrad <T = float32, Device = CPU> */

template <> void AffineGrad<float, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CPUContext*             ctx) {
    const auto* dY = dy; auto* dX = dx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            EigenVectorArrayMap<float>(dX, inner_dim)
                = ConstEigenVectorArrayMap<float>(
                    dY, inner_dim) * alpha[d];
            dY += inner_dim; dX += inner_dim;
        }
    }
}

/*! AffineGrad <T = float16, Device = CPU> */

template <> void AffineGrad<float16, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               scale_dim,
    const float16*          dy,
    const float16*          alpha,
    float16*                dx,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

}  // namespace kernel

}  // namepsace dragon