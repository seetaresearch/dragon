#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _Dropout(
    const int N,
    const AccT scale,
    const uint32_t thresh,
    const uint32_t* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    y[i] = convert::To<T>(
        convert::To<AccT>(x[i]) * AccT(mask[i] = (r[i] > thresh)) * scale);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Dropout<T, CUDAContext>(                                        \
      const int N,                                                     \
      const float ratio,                                               \
      const float scale,                                               \
      const T* x,                                                      \
      T* y,                                                            \
      uint8_t* mask,                                                   \
      uint32_t* r,                                                     \
      CUDAContext* ctx) {                                              \
    math::Random(N, r, ctx);                                           \
    _Dropout<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                             \
        convert::To<math::AccmulatorType<T>::type>(scale),             \
        static_cast<uint32_t>(UINT_MAX * ratio),                       \
        r,                                                             \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),         \
        reinterpret_cast<math::ScalarType<T>::type*>(y),               \
        mask);                                                         \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
