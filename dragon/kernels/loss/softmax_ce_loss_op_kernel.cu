#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _SoftmaxCrossEntropy(
    const int nthreads,
    const T* prob,
    const T* target,
    T* loss) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    loss[i] = -target[i] * log(max(prob[i], FLT_MIN));
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                         \
  template <>                                             \
  void SoftmaxCrossEntropy<T, CUDAContext>(               \
      const int count,                                    \
      const T* prob,                                      \
      const T* target,                                    \
      T* loss,                                            \
      CUDAContext* ctx) {                                 \
    _SoftmaxCrossEntropy<<<                               \
        CUDA_BLOCKS(count),                               \
        CUDA_THREADS,                                     \
        0,                                                \
        ctx->cuda_stream()>>>(count, prob, target, loss); \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
