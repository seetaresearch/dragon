#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _Dropout(
    const int N,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float alpha = float(mask[i] = (r[i] > ratio)) * scale;
    y[i] = convert::To<T>(convert::To<float>(x[i]) * alpha);
  }
}

template <typename T>
__global__ void _DropPath(
    const int NxC,
    const int C,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const int j = index / C;
    const float alpha = float(mask[j] = (r[j] > ratio)) * scale;
    y[index] = convert::To<T>(convert::To<float>(x[index]) * alpha);
  }
}

template <typename T>
__global__ void _DropPathGrad(
    const int NxC,
    const int C,
    const float scale,
    const uint8_t* mask,
    const T* dy,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(index, NxC) {
    const float alpha = float(mask[index / C]) * scale;
    dx[index] = convert::To<T>(convert::To<float>(dy[index]) * alpha);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                      \
  template <>                                                          \
  void Dropout<T, CUDAContext>(                                        \
      const int N,                                                     \
      const float ratio,                                               \
      const float scale,                                               \
      const float* r,                                                  \
      const T* x,                                                      \
      T* y,                                                            \
      uint8_t* mask,                                                   \
      CUDAContext* ctx) {                                              \
    _Dropout<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N,                                                             \
        ratio,                                                         \
        scale,                                                         \
        r,                                                             \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),         \
        reinterpret_cast<math::ScalarType<T>::type*>(y),               \
        mask);                                                         \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void DropPath<T, CUDAContext>(                                          \
      const int N,                                                        \
      const int C,                                                        \
      const float ratio,                                                  \
      const float scale,                                                  \
      const float* r,                                                     \
      const T* x,                                                         \
      T* y,                                                               \
      uint8_t* mask,                                                      \
      CUDAContext* ctx) {                                                 \
    const auto NxC = N * C;                                               \
    _DropPath<<<CUDA_BLOCKS(NxC), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxC,                                                              \
        C,                                                                \
        ratio,                                                            \
        scale,                                                            \
        r,                                                                \
        reinterpret_cast<const math::ScalarType<T>::type*>(x),            \
        reinterpret_cast<math::ScalarType<T>::type*>(y),                  \
        mask);                                                            \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

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
        scale,                                                                \
        mask,                                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(dy),               \
        reinterpret_cast<math::ScalarType<T>::type*>(dx));                    \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
