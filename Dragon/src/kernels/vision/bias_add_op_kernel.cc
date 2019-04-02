#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! BiasAdd <T = float32, Device = CPU> */

template<> void BiasAdd<float, CPUContext>(
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const string&           data_format,
    const float*            bias,
    const float*            bias_multiplier,
    float*                  y,
    CPUContext*             ctx) {
    const int y_offset = dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        if (data_format == "NCHW") {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    dim, inner_dim, 1,
                        1.f, bias, bias_multiplier,
                            1.f, y, ctx);
        } else if (data_format == "NHWC") {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    inner_dim, dim, 1,
                        1.f, bias_multiplier, bias,
                            1.f, y, ctx);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
        y += y_offset;
    }
}

}  // namespace kernel

}  // namepsace dragon