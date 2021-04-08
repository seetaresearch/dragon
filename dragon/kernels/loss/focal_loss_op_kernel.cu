#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
__global__ void _SigmoidFocalLoss(
    const int NxCxS,
    const int S,
    const int C,
    const int start_index,
    const InputT pos_alpha,
    const InputT neg_alpha,
    const InputT gamma,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxCxS) {
    const int j = index % S;
    const int k = index / S % C;
    const int i = index / S / C;
    const int t = target[i * S + j];
    InputT c1 = InputT(t == (k + start_index));
    InputT c2 = InputT((t >= 0) & (t != (k + start_index)));
    InputT p = InputT(1) / (InputT(1) + exp(-input[index]));
    // (1 - p)^{gamma} * log(p)
    InputT pos_term = pow(InputT(1) - p, gamma) * log(max(p, InputT(FLT_MIN)));
    // p^{gamma} * log(1 - p)
    InputT neg_term = pow(p, gamma) *
        (-input[index] * (input[index] >= InputT(0)) -
         log(InputT(1) +
             exp(input[index] -
                 InputT(2) * input[index] * (input[index] >= InputT(0)))));
    loss[index] = InputT(0);
    loss[index] += -c1 * pos_term * pos_alpha;
    loss[index] += -c2 * neg_term * neg_alpha;
    mask[index] = c1;
  }
}

template <typename InputT, typename TargetT>
__global__ void _SigmoidFocalLossGrad(
    const int NxCxS,
    const int S,
    const int C,
    const int start_index,
    const InputT pos_alpha,
    const InputT neg_alpha,
    const InputT gamma,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxCxS) {
    const int j = index % S;
    const int k = index / S % C;
    const int i = index / S / C;
    const int t = target[i * S + j];
    InputT c1 = InputT(t == (k + start_index));
    InputT c2 = InputT((t >= 0) & (t != (k + start_index)));
    InputT p = InputT(1) / (InputT(1) + exp(-input[index]));
    // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    InputT pos_term = pow(InputT(1) - p, gamma) *
        (InputT(1) - p - gamma * p * log(max(p, InputT(FLT_MIN))));
    // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
    InputT neg_term = pow(p, gamma) *
        ((-input[index] * (input[index] >= InputT(0)) -
          log(InputT(1) +
              exp(input[index] -
                  InputT(2) * input[index] * (input[index] >= InputT(0))))) *
             (InputT(1) - p) * gamma -
         p);
    dx[index] = InputT(0);
    dx[index] += -c1 * pos_term * pos_alpha;
    dx[index] += -c2 * neg_term * neg_alpha;
    mask[index] = c1;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                     \
  template <>                                                             \
  void name<InputT, TargetT, CUDAContext>(                                \
      const int N,                                                        \
      const int S,                                                        \
      const int C,                                                        \
      const int start_index,                                              \
      const float alpha,                                                  \
      const float gamma,                                                  \
      const InputT* input,                                                \
      const TargetT* target,                                              \
      InputT* loss,                                                       \
      InputT* mask,                                                       \
      CUDAContext* ctx) {                                                 \
    const auto NxCxS = N * C * S;                                         \
    _##name<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        NxCxS,                                                            \
        S,                                                                \
        C,                                                                \
        start_index,                                                      \
        InputT(alpha),                                                    \
        InputT(1.f - alpha),                                              \
        InputT(gamma),                                                    \
        input,                                                            \
        target,                                                           \
        loss,                                                             \
        mask);                                                            \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
