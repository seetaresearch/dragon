#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__global__ void _BiasAddNCHW(
    const int               nthreads,
    const int               axis_dim,
    const int               inner_dim,
    const T*                bias,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
        y[i] += __ldg(bias + ((i / inner_dim) % axis_dim));
#else
        y[i] += bias[(i / inner_dim) % axis_dim];
#endif
    }
}

template <typename T>
__global__ void _BiasAddNHWC(
    const int               nthreads,
    const int               axis_dim,
    const T*                bias,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
#if __CUDA_ARCH__ >= 350
        y[i] += __ldg(bias + (i % axis_dim));
#else
        y[i] += bias[i % axis_dim];
#endif
    }
}

template<> void BiasAdd<float, CUDAContext>(
    const int               outer_dim,
    const int               axis_dim,
    const int               inner_dim,
    const string&           data_format,
    const float*            bias,
    const float*            multiplier,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = outer_dim * axis_dim * inner_dim;
    if (data_format == "NCHW") {
        _BiasAddNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, inner_dim, bias, y
        );
    } else if (data_format == "NHWC") {
        _BiasAddNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads, axis_dim, bias, y
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA