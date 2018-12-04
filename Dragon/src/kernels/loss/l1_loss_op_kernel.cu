#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! AbsGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _AbsGrad(
    const int               count,
    const T*                dy,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
       const T val = dy[idx];
       //  val > 0: 1 | val == 0: 0 | val < 0: -1
       dx[idx] = (val > T(0)) - (val < T(0));
    }
}

template<> void AbsGrad<float, CUDAContext>(
    const int               count,
    const float*            dy,
    float*                  dx,
    CUDAContext*            ctx) {
    _AbsGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, dy, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA