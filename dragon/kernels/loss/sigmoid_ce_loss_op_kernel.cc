#include "dragon/utils/omp_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _SigmoidCrossEntropy(
    const int count,
    const T* logit,
    const T* target,
    T* loss,
    int* mask) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    if (target[i] < 0) {
      loss[i] = mask[i] = 0;
    } else {
      loss[i] =
          std::log(
              T(1) + std::exp(logit[i] - T(2) * logit[i] * (logit[i] >= 0))) +
          logit[i] * ((logit[i] >= 0) - target[i]);
      mask[i] = 1;
    }
  }
}

template <typename T>
void _SigmoidCrossEntropyGrad(
    const int count,
    const T* logit,
    const T* target,
    T* dx,
    int* mask) {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(OMP_THREADS(count))
#endif
  for (int i = 0; i < count; ++i) {
    if (target[i] < 0) {
      dx[i] = mask[i] = 0;
    } else {
      dx[i] = T(1) / (T(1) + std::exp(-logit[i])) - target[i];
      mask[i] = 1;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, T)        \
  template <>                                  \
  void name<T, CPUContext>(                    \
      const int count,                         \
      const T* logit,                          \
      const T* target,                         \
      T* loss,                                 \
      int* mask,                               \
      CPUContext* ctx) {                       \
    _##name(count, logit, target, loss, mask); \
  }

DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropy, double);

DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, float);
DEFINE_KERNEL_LAUNCHER(SigmoidCrossEntropyGrad, double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
