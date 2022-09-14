#include "dragon/kernels/loss/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__global__ void
_CrossEntropy(const int N, const T* input, const T* target, T* loss) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    loss[i] = -target[i] * log(max(input[i], FLT_MIN));
  }
}

template <typename InputT, typename TargetT>
__global__ void _CrossEntropy(
    const int NxS,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* input,
    const TargetT* target,
    InputT* loss,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxS) {
    const int i = index / S, j = index % S;
    const int tgt = target[index];
    if (tgt == ignore_index) {
      loss[index] = mask[index] = InputT(0);
    } else {
      loss[index] = -log(max(input[(i * C + tgt) * S + j], InputT(FLT_MIN)));
      mask[index] = InputT(1);
    }
  }
}

template <typename T>
__global__ void _SigmoidCrossEntropy(
    const int N,
    const T* input,
    const T* target,
    T* loss,
    T* mask) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    if (target[i] < 0) {
      loss[i] = mask[i] = T(0);
    } else {
      const float lgt = input[i];
      loss[i] = log(1.f + exp(lgt - 2.f * lgt * (lgt >= 0.f))) +
          lgt * ((lgt >= 0.f) - float(target[i]));
      mask[i] = T(1);
    }
  }
}

template <typename T>
__global__ void _SigmoidCrossEntropyGrad(
    const int N,
    const T* input,
    const T* target,
    T* dx,
    T* mask) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    if (target[i] < 0) {
      dx[i] = mask[i] = T(0);
    } else {
      dx[i] = T(1) / (T(1) + exp(-input[i])) - target[i];
      mask[i] = T(1);
    }
  }
}

template <typename InputT, typename TargetT>
__global__ void _SoftmaxCrossEntropyGrad(
    const int NxS,
    const int S,
    const int C,
    const int ignore_index,
    const InputT* /* input */,
    const TargetT* target,
    InputT* dx,
    InputT* mask) {
  CUDA_1D_KERNEL_LOOP(index, NxS) {
    const int i = index / S, j = index % S;
    const int tgt = target[index];
    if (tgt == ignore_index) {
      InputT* offset_dx = dx + i * C * S + j;
      for (int _ = 0; _ < C; ++_, offset_dx += S) {
        offset_dx[0] = InputT(0);
      }
      mask[index] = InputT(0);
    } else {
      dx[(i * C + tgt) * S + j] -= InputT(1);
      mask[index] = InputT(1);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const T* input,                                                 \
      const T* target,                                                \
      T* loss,                                                        \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, input, target, loss);                                      \
  }

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)                               \
  template <>                                                         \
  void name<T, CUDAContext>(                                          \
      const int N,                                                    \
      const T* input,                                                 \
      const T* target,                                                \
      T* loss,                                                        \
      T* mask,                                                        \
      CUDAContext* ctx) {                                             \
    _##name<<<CUDA_BLOCKS(N), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        N, input, target, loss, mask);                                \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, double);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

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

DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, float, int64_t);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int);
DEFINE_KERNEL_LAUNCHER(CrossEntropy, double, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, float, int64_t);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int);
DEFINE_KERNEL_LAUNCHER(SoftmaxCrossEntropyGrad, double, int64_t);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
