#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _GradientTwoSum(const int count, const T* dy1, const T* dy2, T* dx) {
  EigenVectorArrayMap<T>(dx, count) +=
      (ConstEigenVectorArrayMap<T>(dy1, count) +
       ConstEigenVectorArrayMap<T>(dy2, count));
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void GradientTwoSum<float16, CPUContext>(
    const int count,
    const float16* dy1,
    const float16* dy2,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // TwoSumGrad

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void GradientTwoSum<T, CPUContext>(                                        \
      const int count, const T* dy1, const T* dy2, T* dx, CPUContext* ctx) { \
    _GradientTwoSum(count, dy1, dy2, dx);                                    \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
