#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_Range(const int N, const double start, const double delta, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(start + double(i) * delta);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void Range<T, CUDAContext>(                                              \
      const int N,                                                         \
      const double start,                                                  \
      const double delta,                                                  \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    _Range<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(       \
        N, start, delta, reinterpret_cast<math::ScalarType<T>::type*>(y)); \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
