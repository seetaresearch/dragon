#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! BiasAdd <T = float32, Device = CPU> */

template<> void BiasAdd<float, CPUContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const string&           data_format,
    const float*            bias,
    const float*            multiplier,
    float*                  y,
    CPUContext*             ctx) {
    const int y_ofs = axis_dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        if (data_format == "NCHW") {
            math::Gemm(
                CblasNoTrans,
                CblasNoTrans,
                axis_dim, inner_dim, 1,
                1.f, bias, multiplier,
                1.f, y + n * y_ofs, ctx
            );
        } else if (data_format == "NHWC") {
            math::Gemm(
                CblasNoTrans,
                CblasNoTrans,
                inner_dim, axis_dim, 1,
                1.f, multiplier, bias,
                1.f, y + n * y_ofs, ctx
            );
        } else {
            LOG(FATAL) << "Unknown DataFormat: " << data_format;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon