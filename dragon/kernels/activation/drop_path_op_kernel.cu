#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _DropPath(
    const int NxC,
    const int C,
    const AccT scale,
    const uint32_t thresh,
    const uint32_t* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    y[i] = convert::To<T>(
        convert::To<AccT>(x[i]) *
        AccT(mask[i / C] = (__ldg(r + i / C) > thresh)) * scale);
  }
}

template <typename T, typename AccT>
__global__ void _DropPathGrad(
    const int NxC,
    const int C,
    const AccT scale,
    const uint8_t* mask,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxC) {
    dx[i] = convert::To<T>(
        convert::To<AccT>(dy[i]) * AccT(__ldg(mask + i / C)) * scale);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void DropPath<T, CUDAContext>(                                          \
      const int N,                                                        \
      const int C,                                                        \
      const float ratio,                                                  \
      const float scale,                                                  \
      const T* x,                                                         \
      T* y,                                                               \
      uint8_t* mask,                                                      \
      uint32_t* r,                                                        \
      CUDAContext* ctx) {                                                 \
    const auto NxC = N * C;                                               \
    math::Random(N, r, ctx);                                              \
    _DropPath<<<CUDA_BLOCKS(NxC), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxC,                                                              \
        C,                                                                \
        convert::To<math::AccmulatorType<T>::type>(scale),                \
        static_cast<uint32_t>(UINT_MAX * ratio),                          \
        r,                                                                \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
        reinterpret_cast<math::ScalarType<T>::type*>(y),                  \
        mask);                                                            \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                        \
  template <>                                                                 \
  void DropPathGrad<T, CUDAContext>(                                          \
      const int N,                                                            \
      const int C,                                                            \
      const float scale,                                                      \
      const uint8_t* mask,                                                    \
      const T* dy,                                                            \
      T* dx,                                                                  \
      CUDAContext* ctx) {                                                     \
    const auto NxC = N * C;                                                   \
    _DropPathGrad<<<CUDA_BLOCKS(NxC), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxC,                                                                  \
        C,                                                                    \
        convert::To<math::AccmulatorType<T>::type>(scale),                    \
        mask,                                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),               \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));                    \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
