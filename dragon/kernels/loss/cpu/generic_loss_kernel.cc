#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/math_functions.h"

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
  const auto mul = math::MultipliesFunctor<T>();
  std::array<int, 3> dims = {N, C, S};
  std::array<int, 3> index = {0, 0, 0};
  for (int i = 0; i < NxCxS; ++i) {
    dx[i] = mul(dx[i], dl[index[0] * S + index[2]]);
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                  \
  template <>                                                      \
  void ReduceLoss<T, CPUContext>(                                  \
      const int N,                                                 \
      const int num_masks,                                         \
      const float normalizer,                                      \
      const T* x,                                                  \
      const T* mask,                                               \
      T* y,                                                        \
      CPUContext* ctx) {                                           \
    float inv_scale = num_masks > 0 && normalizer < 0.f            \
        ? convert::To<float>(math::Sum(num_masks, 1.f, mask, ctx)) \
        : normalizer;                                              \
    y[0] = math::Sum(N, 1.f / std::max(inv_scale, 1.f), x, ctx);   \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void ReduceLossGrad<T, CPUContext>(                                    \
      const int N,                                                       \
      const int num_masks,                                               \
      const float normalizer,                                            \
      const T* dl,                                                       \
      const T* mask,                                                     \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    float inv_scale = num_masks > 0 && normalizer < 0.f                  \
        ? convert::To<float>(math::Sum(num_masks, 1.f, mask, ctx))       \
        : normalizer;                                                    \
    float scale = convert::To<float>(dl[0]) / std::max(inv_scale, 0.5f); \
    math::Scale(N, scale, dx, dx, ctx);                                  \
  }                                                                      \
  template <>                                                            \
  void BroadcastLossGrad<T, CPUContext>(                                 \
      const int N,                                                       \
      const int S,                                                       \
      const int C,                                                       \
      const T* dl,                                                       \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    _BroadcastLossGrad(N, S, C, dl, dx);                                 \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
