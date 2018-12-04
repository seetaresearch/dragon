#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Affine <T = float32, Device = CPU> */

template<> void Affine<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    const float*            beta_multiplier,
    float*                  y,
    CPUContext*             ctx) {
    //  Ax
    auto* Xdata = x; auto* Ydata = y;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            math::Scale<float, CPUContext>(
                inner_dim, alpha[d], Xdata, Ydata, ctx);
            Xdata += inner_dim; 
            Ydata += inner_dim;
        }
    }
    //  Pb
    if (beta != nullptr && beta_multiplier != nullptr) {
        int dim = scale_dim * inner_dim;
        Ydata = y;
        for (int n = 0; n < outer_dim; ++n) {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    scale_dim, inner_dim, 1,
                        1.f, beta, beta_multiplier,
                            1.f, Ydata, ctx);
             Ydata += dim;
        }
    }
}

/*! Affine <T = float16, Device = CPU> */

template<> void Affine<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    const float16*          beta_multiplier,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}


/*! AffineGrad <T = float32, Device = CPU> */

template <> void AffineGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CPUContext*             ctx) {
    auto* dYdata = dy; auto* dXdata = dx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            math::Scale<float, CPUContext>(
                inner_dim, alpha[d], dYdata, dXdata, ctx);
            dYdata += inner_dim; dXdata += inner_dim;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon