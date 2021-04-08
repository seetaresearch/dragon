#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Repeat(
    const int NxCxS2,
    const int C,
    const int S1,
    const int S2,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, NxCxS2) {
    const int k = yi % S1;
    const int j = yi / S2 % C;
    const int i = yi / S2 / C;
    const int xi = (i * C + j) * S1 + k;
    y[yi] = x[xi];
  }
}

template <typename T, typename AccT>
__global__ void _RepeatGrad(
    const int NxCxS1,
    const int C,
    const int S1,
    const int S2,
    const T* dy,
    T* dx) {
  const int repeats = S2 / S1;
  CUDA_1D_KERNEL_LOOP(xi, NxCxS1) {
    const int k = xi % S1;
    const int j = xi / S1 % C;
    const int i = xi / S1 / C;
    const T* offset_dy = dy + ((i * C + j) * S2 + k);
    AccT val = AccT(0);
    for (int r = 0; r < repeats; ++r) {
      val += convert::To<AccT>(offset_dy[r * S1]);
    }
    dx[xi] = convert::To<T>(val);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void Repeat<T, CUDAContext>(                                             \
      const int N,                                                         \
      const int S,                                                         \
      const int C,                                                         \
      const int repeats,                                                   \
      const T* x,                                                          \
      T* y,                                                                \
      CUDAContext* ctx) {                                                  \
    const auto S2 = S * repeats;                                           \
    const auto NxCxS2 = N * C * S2;                                        \
    _Repeat<<<CUDA_BLOCKS(NxCxS2), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxCxS2, C, S, S2, x, y);                                           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                    \
  template <>                                                             \
  void RepeatGrad<T, CUDAContext>(                                        \
      const int N,                                                        \
      const int S,                                                        \
      const int C,                                                        \
      const int repeats,                                                  \
      const T* dy,                                                        \
      T* dx,                                                              \
      CUDAContext* ctx) {                                                 \
    const auto S2 = S * repeats;                                          \
    const auto NxCxS = N * C * S;                                         \
    _RepeatGrad<math::ScalarType<T>::type, math::AccmulatorType<T>::type> \
        <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(    \
            NxCxS,                                                        \
            C,                                                            \
            S,                                                            \
            S2,                                                           \
            reinterpret_cast<const math::ScalarType<T>::type*>(dy),       \
            reinterpret_cast<math::ScalarType<T>::type*>(dx));            \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
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
