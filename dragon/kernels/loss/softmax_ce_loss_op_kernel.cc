#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _SoftmaxCrossEntropy(
    const int count,
    const T* prob,
    const T* target,
    T* loss) {
  EigenVectorArrayMap<T>(loss, count) =
      ConstEigenVectorArrayMap<T>(target, count) *
      ConstEigenVectorArrayMap<T>(prob, count).unaryExpr([](T a) {
        return -std::log(std::max(a, (T)FLT_MIN));
      });
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                    \
  template <>                                        \
  void SoftmaxCrossEntropy<T, CPUContext>(           \
      const int count,                               \
      const T* prob,                                 \
      const T* target,                               \
      T* loss,                                       \
      CPUContext* ctx) {                             \
    _SoftmaxCrossEntropy(count, prob, target, loss); \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
