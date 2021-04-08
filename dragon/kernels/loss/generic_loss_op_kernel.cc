#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _BroadcastLossGrad(
    const int N,
    const int S,
    const int C,
    const T* dl,
    T* dx) {
  const auto NxCxS = N * C * S;
  std::array<int, 3> dims = {N, C, S};
  std::array<int, 3> index = {0, 0, 0};
  for (int i = 0; i < NxCxS; ++i) {
    dx[i] *= dl[index[0] * S + index[2]];
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

} // namespace

template <>
void ReduceLoss<float16, CPUContext>(
    const int N,
    const int num_masks,
    const float normalizer,
    const float16* x,
    const float16* mask,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ReduceLossGrad<float16, CPUContext>(
    const int N,
    const int num_masks,
    const float normalizer,
    const float16* dl,
    const float16* mask,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void BroadcastLossGrad<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const float16* dl,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                         \
  template <>                                             \
  void ReduceLoss<T, CPUContext>(                         \
      const int N,                                        \
      const int num_masks,                                \
      const float normalizer,                             \
      const T* x,                                         \
      const T* mask,                                      \
      T* y,                                               \
      CPUContext* ctx) {                                  \
    float inv_scale = std::max(                           \
        1.f,                                              \
        num_masks > 0 && normalizer < 0.f                 \
            ? (float)math::Sum(num_masks, 1.f, mask, ctx) \
            : normalizer);                                \
    y[0] = math::Sum(N, 1.f / inv_scale, x, ctx);         \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                  \
  template <>                                                           \
  void ReduceLossGrad<T, CPUContext>(                                   \
      const int N,                                                      \
      const int num_masks,                                              \
      const float normalizer,                                           \
      const T* dl,                                                      \
      const T* mask,                                                    \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    float inv_scale = std::max(                                         \
        0.5f,                                                           \
        num_masks > 0 && normalizer < 0.f                               \
            ? (float)math::Sum(num_masks, 1.f, mask, ctx)               \
            : normalizer);                                              \
    math::Scale(N, convert::To<float>(dl[0]) / inv_scale, dx, dx, ctx); \
  }                                                                     \
  template <>                                                           \
  void BroadcastLossGrad<T, CPUContext>(                                \
      const int N,                                                      \
      const int S,                                                      \
      const int C,                                                      \
      const T* dl,                                                      \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    _BroadcastLossGrad(N, S, C, dl, dx);                                \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
