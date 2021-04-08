#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
__global__ void _NLLLoss(
    const int NxS,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxS) {
    const int i = index / S;
    const int j = index % S;
    const int t = target[i * S + j];
    if (t == ignore_index) {
      loss[index] = mask[index] = InputT(0);
    } else {
      loss[index] = -input[(i * C + t) * S + j];
      mask[index] = InputT(1);
    }
  }
}

template <typename InputT, typename TargetT>
__global__ void _NLLLossGrad(
    const int NxS,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxS) {
    const int i = index / S;
    const int j = index % S;
    const int t = target[i * S + j];
    if (t == ignore_index) {
      mask[index] = InputT(0);
    } else {
      dx[(i * C + t) * S + j] = InputT(-1);
      mask[index] = InputT(1);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                   \
  template <>                                                           \
  void name<InputT, TargetT, CUDAContext>(                              \
      const int N,                                                      \
      const int S,                                                      \
      const int C,                                                      \
      const int ignore_index,                                           \
      const InputT* input,                                              \
      const TargetT* target,                                            \
      InputT* loss,                                                     \
      InputT* mask,                                                     \
      CUDAContext* ctx) {                                               \
    const auto NxS = N * S;                                             \
    _##name<<<CUDA_BLOCKS(NxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxS, S, C, ignore_index, input, target, loss, mask);            \
  }

DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(NLLLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
