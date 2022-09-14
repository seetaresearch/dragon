#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void
_ReduceLossGrad(const int N, const AccT scale, const T* dl, T* dx) {
  const AccT val = convert::To<AccT>(__ldg(dl)) * scale;
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = convert::To<T>(convert::To<AccT>(dx[i]) * val);
  }
}

template <typename T, typename AccT>
__global__ void
_ReduceLossGrad(const int N, const T* normalizer, const T* dl, T* dx) {
  const AccT val = convert::To<AccT>(__ldg(dl)) /
      max(convert::To<AccT>(__ldg(normalizer)), AccT(1));
  CUDA_1D_KERNEL_LOOP(i, N) {
    dx[i] = convert::To<T>(convert::To<AccT>(dx[i]) * val);
  }
}

template <typename T, typename AccT>
__global__ void _BroadcastLossGrad(
    const int NxCxS,
    const int CxS,
    const int S,
    const T* dl,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, NxCxS) {
    const AccT val = convert::To<AccT>(__ldg(dl + i / CxS * S + i % S));
    dx[i] = convert::To<T>(convert::To<AccT>(dx[i]) * val);
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                               \
  template <>                                                   \
  void ReduceLoss<T, CUDAContext>(                              \
      const int N,                                              \
      const int num_masks,                                      \
      const float normalizer,                                   \
      const T* x,                                               \
      const T* mask,                                            \
      T* y,                                                     \
      CUDAContext* ctx) {                                       \
    if (num_masks > 0 && normalizer < 0.f) {                    \
      auto* num_valid = const_cast<T*>(mask + num_masks);       \
      math::Sum(num_masks, 1.f, mask, num_valid, ctx);          \
      math::Sum(N, 1.f, x, y, ctx);                             \
      math::Div(1, y, num_valid, y, ctx);                       \
    } else {                                                    \
      math::Sum(N, 1.f / std::max(1.f, normalizer), x, y, ctx); \
    }                                                           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                 \
  template <>                                                          \
  void ReduceLossGrad<T, CUDAContext>(                                 \
      const int N,                                                     \
      const int num_masks,                                             \
      const float normalizer,                                          \
      const T* dl,                                                     \
      const T* mask,                                                   \
      T* dx,                                                           \
      CUDAContext* ctx) {                                              \
    using ScalarT = math::ScalarType<T>::type;                         \
    using AccT = math::AccumulatorType<T>::type;                       \
    if (num_masks > 0 && normalizer < 0.f) {                           \
      auto* num_valid = const_cast<T*>(mask + num_masks);              \
      math::Sum(num_masks, 1.f, mask, num_valid, ctx);                 \
      _ReduceLossGrad<ScalarT, AccT>                                   \
          <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
              N,                                                       \
              reinterpret_cast<const ScalarT*>(num_valid),             \
              reinterpret_cast<const ScalarT*>(dl),                    \
              reinterpret_cast<ScalarT*>(dx));                         \
    } else {                                                           \
      _ReduceLossGrad<ScalarT, AccT>                                   \
          <<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
              N,                                                       \
              AccT(1.f / std::max(0.5f, normalizer)),                  \
              reinterpret_cast<const ScalarT*>(dl),                    \
              reinterpret_cast<ScalarT*>(dx));                         \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  void BroadcastLossGrad<T, CUDAContext>(                              \
      const int N,                                                     \
      const int S,                                                     \
      const int C,                                                     \
      const T* dl,                                                     \
      T* dx,                                                           \
      CUDAContext* ctx) {                                              \
    using ScalarT = math::ScalarType<T>::type;                         \
    using AccT = math::AccumulatorType<T>::type;                       \
    const auto CxS = C * S;                                            \
    const auto NxCxS = N * CxS;                                        \
    _BroadcastLossGrad<ScalarT, AccT>                                  \
        <<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
            NxCxS,                                                     \
            CxS,                                                       \
            S,                                                         \
            reinterpret_cast<const ScalarT*>(dl),                      \
            reinterpret_cast<ScalarT*>(dx));                           \
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
