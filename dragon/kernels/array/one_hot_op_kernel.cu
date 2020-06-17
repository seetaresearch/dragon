#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _OneHot(
    const int nthreads,
    const int depth,
    const int on_value,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int val = x[i];
    y[i * depth + val] = (T)on_value;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void OneHot<T, CUDAContext>(                                            \
      const int count,                                                    \
      const int depth,                                                    \
      const int on_value,                                                 \
      const T* x,                                                         \
      T* y,                                                               \
      CUDAContext* ctx) {                                                 \
    _OneHot<<<CUDA_BLOCKS(count), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        count, depth, on_value, x, y);                                    \
  }

DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
