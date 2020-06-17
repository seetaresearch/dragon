#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _BroadcastLossGrad(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const T* dy,
    T* dx) {
  std::array<int, 3> dims = {outer_dim, axis_dim, inner_dim};
  std::array<int, 3> idx = {0, 0, 0};
  const int count = outer_dim * axis_dim * inner_dim;
  for (int i = 0; i < count; ++i) {
    dx[i] *= dy[idx[0] * inner_dim + idx[2]];
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <>
void _BroadcastLossGrad<float16>(
    const int outer_dim,
    const int axis_dim,
    const int inner_dim,
    const float16* dy,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // BroadcastLossGrad

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void ReduceLoss<T, CPUContext>(                                   \
      const int count,                                              \
      const int num_masks,                                          \
      const float normalizer,                                       \
      const T* x,                                                   \
      const int* mask,                                              \
      T* y,                                                         \
      CPUContext* ctx) {                                            \
    float inv_scale = std::max(                                     \
        1e-5F,                                                      \
        num_masks > 0 ? (float)math::Sum(num_masks, 1.f, mask, ctx) \
                      : normalizer);                                \
    y[0] = math::Sum(count, 1.f / inv_scale, x, ctx);               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void ReduceLossGrad<T, CPUContext>(                                    \
      const int count,                                                   \
      const int num_masks,                                               \
      const float normalizer,                                            \
      const T* dy,                                                       \
      const int* mask,                                                   \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    float inv_scale = std::max(                                          \
        1e-5F,                                                           \
        num_masks > 0 ? (float)math::Sum(num_masks, 1.f, mask, ctx)      \
                      : normalizer);                                     \
    math::Scale(count, cast::to<float>(dy[0]) / inv_scale, dx, dx, ctx); \
  }                                                                      \
  template <>                                                            \
  void BroadcastLossGrad<T, CPUContext>(                                 \
      const int outer_dim,                                               \
      const int axis_dim,                                                \
      const int inner_dim,                                               \
      const T* dy,                                                       \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    _BroadcastLossGrad(outer_dim, axis_dim, inner_dim, dy, dx);          \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
