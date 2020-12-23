#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename LogitT, typename TargetT>
__global__ void _SigmoidFocalLoss(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const LogitT pos_alpha,
    const LogitT neg_alpha,
    const LogitT gamma,
    const int negative_index,
    const LogitT* logit,
    const TargetT* target,
    LogitT* loss,
    LogitT* mask) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int j = yi % inner_dim;
    const int k = (yi / inner_dim) % axis_dim;
    const int i = yi / inner_dim / axis_dim;
    const int t = target[i * inner_dim + j];

    // "0" is reserved for target if negative index is zero
    LogitT c1 = (LogitT)(t == (k + (negative_index ? 0 : 1)));
    LogitT c2 = (LogitT)((t >= 0) & (t != (k + (negative_index ? 0 : 1))));
    LogitT p = LogitT(1) / (LogitT(1) + exp(-logit[yi]));

    // (1 - p)^{gamma} * log(p)
    LogitT pos_term = pow(LogitT(1) - p, gamma) * log(max(p, FLT_MIN));

    // p^{gamma} * log(1 - p)
    LogitT neg_term = pow(p, gamma) *
        (-logit[yi] * (logit[yi] >= 0) -
         log(LogitT(1) +
             exp(logit[yi] - LogitT(2) * logit[yi] * (logit[yi] >= 0))));

    loss[yi] = LogitT(0);
    loss[yi] += -c1 * pos_term * pos_alpha;
    loss[yi] += -c2 * neg_term * neg_alpha;
    mask[yi] = c1;
  }
}

template <typename LogitT, typename TargetT>
__global__ void _SigmoidFocalLossGrad(
    const int nthreads,
    const int inner_dim,
    const int axis_dim,
    const LogitT pos_alpha,
    const LogitT neg_alpha,
    const LogitT gamma,
    const int negative_index,
    const LogitT* logit,
    const TargetT* target,
    LogitT* dx,
    LogitT* mask) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int j = xi % inner_dim;
    const int k = (xi / inner_dim) % axis_dim;
    const int i = xi / inner_dim / axis_dim;
    const int t = target[i * inner_dim + j];

    // "0" is reserved for target if neg index is zero
    LogitT c1 = (LogitT)(t == (k + (negative_index ? 0 : 1)));
    LogitT c2 = (LogitT)((t >= 0) & (t != (k + (negative_index ? 0 : 1))));
    LogitT p = LogitT(1) / (LogitT(1) + exp(-logit[xi]));

    // (1 - p)^{gamma} * (1 - p - gamma * p * log(p))
    LogitT pos_term = pow(LogitT(1) - p, gamma) *
        (LogitT(1) - p - p * gamma * log(max(p, FLT_MIN)));

    // p^{gamma} * (gamma * (1 - p) * log(1-p) - p)
    LogitT neg_term = pow(p, gamma) *
        ((-logit[xi] * (logit[xi] >= 0) -
          log(LogitT(1) +
              exp(logit[xi] - LogitT(2) * logit[xi] * (logit[xi] >= 0)))) *
             (LogitT(1) - p) * gamma -
         p);

    dx[xi] = LogitT(0);
    dx[xi] += -c1 * pos_term * pos_alpha;
    dx[xi] += -c2 * neg_term * neg_alpha;
    mask[xi] = c1;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, LogitT, TargetT)                        \
  template <>                                                                \
  void name<LogitT, TargetT, CUDAContext>(                                   \
      const int outer_dim,                                                   \
      const int inner_dim,                                                   \
      const int axis_dim,                                                    \
      const float pos_alpha,                                                 \
      const float neg_alpha,                                                 \
      const float gamma,                                                     \
      const int negative_index,                                              \
      const LogitT* logit,                                                   \
      const TargetT* target,                                                 \
      LogitT* loss,                                                          \
      LogitT* mask,                                                          \
      CUDAContext* ctx) {                                                    \
    const auto nthreads = outer_dim * axis_dim * inner_dim;                  \
    _##name<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads,                                                            \
        inner_dim,                                                           \
        axis_dim,                                                            \
        (LogitT)pos_alpha,                                                   \
        (LogitT)neg_alpha,                                                   \
        (LogitT)gamma,                                                       \
        negative_index,                                                      \
        logit,                                                               \
        target,                                                              \
        loss,                                                                \
        mask);                                                               \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, float);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, double);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLoss, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, float);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, double);
DEFINE_KERNEL_LAUNCHER(SigmoidFocalLossGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
