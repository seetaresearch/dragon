#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename LogitType, typename TargetType>
__global__ void _SparseSoftmaxCrossEntropy(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
    const TargetType* target,
    LogitType* loss,
    LogitType* mask) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const int label = target[i * inner_dim + j];
    if (label == ignore_index) {
      loss[yi] = mask[yi] = LogitType(0);
    } else {
      loss[yi] = -log(max(
          prob[(i * axis_dim + label) * inner_dim + j], LogitType(FLT_MIN)));
      mask[yi] = LogitType(1);
    }
  }
}

template <typename LogitType, typename TargetType>
__global__ void _SparseSoftmaxCrossEntropyGrad(
    const int nthreads,
    const int axis_dim,
    const int inner_dim,
    const int ignore_index,
    const LogitType* prob,
    const TargetType* target,
    LogitType* dx,
    LogitType* mask) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int i = yi / inner_dim;
    const int j = yi % inner_dim;
    const int label = target[i * inner_dim + j];
    if (label == ignore_index) {
      LogitType* offset_dx = dx + i * axis_dim * inner_dim + j;
      for (int k = 0; k < axis_dim; ++k) {
        (*offset_dx) = LogitType(0);
        offset_dx += inner_dim;
      }
      mask[yi] = LogitType(0);
    } else {
      dx[(i * axis_dim + label) * inner_dim + j] -= LogitType(1);
      mask[yi] = LogitType(1);
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, LogitType, TargetType)                  \
  template <>                                                                \
  void name<LogitType, TargetType, CUDAContext>(                             \
      const int outer_dim,                                                   \
      const int axis_dim,                                                    \
      const int inner_dim,                                                   \
      const int ignore_index,                                                \
      const LogitType* prob,                                                 \
      const TargetType* target,                                              \
      LogitType* loss,                                                       \
      LogitType* mask,                                                       \
      CUDAContext* ctx) {                                                    \
    const int nthreads = outer_dim * inner_dim;                              \
    _##name<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads,                                                            \
        axis_dim,                                                            \
        inner_dim,                                                           \
        ignore_index,                                                        \
        prob,                                                                \
        target,                                                              \
        loss,                                                                \
        mask);                                                               \
  }

DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, float, float);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, double, double);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropy, double, int64_t);

DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, float, float);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, double, double);
DEFINE_KERNEL_LAUNCHER(SparseSoftmaxCrossEntropyGrad, double, int64_t);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
