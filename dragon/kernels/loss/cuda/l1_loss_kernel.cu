#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void _SmoothL1Loss(
    const int N,
    const T beta,
    const T* input,
    const T* target,
    T* loss) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T val = input[i] - target[i];
    const T abs_val = abs(val);
    loss[i] = abs_val < beta ? T(.5) * val * val / beta : abs_val - .5 * beta;
  }
}

template <typename T>
__global__ void _SmoothL1LossGrad(
    const int N,
    const T beta,
    const T* input,
    const T* target,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const T val = input[i] - target[i];
    const T abs_val = abs(val);
    dx[i] = abs_val < beta ? val / beta : (val > T(0)) - (val < T(0));
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const float beta,                                               \
      const T* input,                                                 \
      const T* target,                                                \
      T* loss,                                                        \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, T(beta), input, target, loss);                             \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
