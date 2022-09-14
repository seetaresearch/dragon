#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename InputT, typename TargetT>
__global__ void _SigmoidFocalLoss(
    const int NxCxS,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxCxS) {
    const int j = index % S;
    const int k = index / S % C;
    const int i = index / S / C;
    const float lgt = input[index];
    const int tgt = target[i * S + j];
    const float c1 = float(tgt == (k + start_index));
    const float c2 = float((tgt >= 0) & (tgt != (k + start_index)));
    const float p = 1.f / (1.f + exp(-lgt));
    // pos_term: (1 - p)^{gamma} * log(p)
    const float v1 = pow(1.f - p, gamma) * log(max(p, FLT_MIN));
    // neg_term: p^{gamma} * log(1 - p)
    const float v2 = pow(p, gamma) *
        (-lgt * (lgt >= 0.f) - log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f))));
    loss[index] = -(c1 * v1 * alpha + c2 * v2 * (1.f - alpha));
    mask[index] = c1;
  }
}

template <typename InputT, typename TargetT>
__global__ void _SigmoidFocalLossGrad(
    const int NxCxS,
    const int S,
    const int C,
    const int start_index,
    const float alpha,
    const float gamma,
    const InputT* input,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxCxS) {
    const int j = index % S;
    const int k = index / S % C;
    const int i = index / S / C;
    const float lgt = input[index];
    const int tgt = target[i * S + j];
    const float c1 = float(tgt == (k + start_index));
    const float c2 = float((tgt >= 0) & (tgt != (k + start_index)));
    const float p = 1.f / (1.f + exp(-lgt));
    // pos_term: (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    const float v1 =
        pow(1.f - p, gamma) * (1.f - p - gamma * p * log(max(p, FLT_MIN)));
    // neg_term: p^{gamma} * (gamma * (1 - p) * log(1 - p) - p)
    const float v2 = pow(p, gamma) *
        ((-lgt * (lgt >= 0.f) -
          log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f)))) *
             (1.f - p) * gamma -
         p);
    dx[index] = -(c1 * v1 * alpha + c2 * v2 * (1.f - alpha));
    mask[index] = c1;
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, InputT, TargetT)                       \
  template <>                                                               \
  void name<InputT, TargetT, CUDAContext>(                                  \
      const int N,                                                          \
      const int S,                                                          \
      const int C,                                                          \
      const int start_index,                                                \
      const float alpha,                                                    \
      const float gamma,                                                    \
      const InputT* input,                                                  \
      const TargetT* target,                                                \
      InputT* loss,                                                         \
      InputT* mask,                                                         \
      CUDAContext* ctx) {                                                   \
    const auto NxCxS = N * C * S;                                           \
    _##name<<<CUDA_BLOCKS(NxCxS), CUDA_THREADS, 0, ctx->cuda_stream()>>>(   \
        NxCxS, S, C, start_index, alpha, gamma, input, target, loss, mask); \
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
