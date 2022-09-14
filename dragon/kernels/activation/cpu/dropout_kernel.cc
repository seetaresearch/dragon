#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Dropout(
    const int N,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  for (int i = 0; i < N; ++i) {
    const float alpha = float(mask[i] = (r[i] > ratio)) * scale;
    y[i] = convert::To<T>(convert::To<float>(x[i]) * alpha);
  }
}

template <typename T>
void _DropPath(
    const int N,
    const int C,
    const float ratio,
    const float scale,
    const float* r,
    const T* x,
    T* y,
    uint8_t* mask) {
  const auto NxC = N * C;
  for (int index = 0; index < NxC; ++index) {
    const int j = index / C;
    const float alpha = float(mask[j] = (r[j] > ratio)) * scale;
    y[index] = convert::To<T>(convert::To<float>(x[index]) * alpha);
  }
}

template <typename T>
void _DropPathGrad(
    const int N,
    const int C,
    const float scale,
    const uint8_t* mask,
    const T* dy,
    T* dx) {
  const auto NxC = N * C;
  for (int index = 0; index < NxC; ++index) {
    const float alpha = float(mask[index / C]) * scale;
    dx[index] = convert::To<T>(convert::To<float>(dy[index]) * alpha);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)             \
  template <>                                 \
  void Dropout<T, CPUContext>(                \
      const int N,                            \
      const float ratio,                      \
      const float scale,                      \
      const float* r,                         \
      const T* x,                             \
      T* y,                                   \
      uint8_t* mask,                          \
      CPUContext* ctx) {                      \
    _Dropout(N, ratio, scale, r, x, y, mask); \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                 \
  template <>                                     \
  void DropPath<T, CPUContext>(                   \
      const int N,                                \
      const int C,                                \
      const float ratio,                          \
      const float scale,                          \
      const float* r,                             \
      const T* x,                                 \
      T* y,                                       \
      uint8_t* mask,                              \
      CPUContext* ctx) {                          \
    _DropPath(N, C, ratio, scale, r, x, y, mask); \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)        \
  template <>                                 \
  void DropPathGrad<T, CPUContext>(           \
      const int N,                            \
      const int C,                            \
      const float scale,                      \
      const uint8_t* mask,                    \
      const T* dy,                            \
      T* dx,                                  \
      CPUContext* ctx) {                      \
    _DropPathGrad(N, C, scale, mask, dy, dx); \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
