#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SmoothL1Loss(
    const int N,
    const T beta,
    const T* input,
    const T* target,
    T* loss) {
  EigenVectorArrayMap<T>(loss, N) =
      (ConstEigenVectorArrayMap<T>(input, N) -
       ConstEigenVectorArrayMap<T>(target, N));
  EigenVectorArrayMap<T>(loss, N) =
      ConstEigenVectorArrayMap<T>(loss, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? T(.5) * a * a / beta
                                  : std::abs(a) - T(.5) * beta;
      });
}

template <typename T>
void _SmoothL1LossGrad(
    const int N,
    const T beta,
    const T* input,
    const T* target,
    T* dx) {
  EigenVectorArrayMap<T>(dx, N) =
      (ConstEigenVectorArrayMap<T>(input, N) -
       ConstEigenVectorArrayMap<T>(target, N));
  EigenVectorArrayMap<T>(dx, N) =
      ConstEigenVectorArrayMap<T>(dx, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? a / beta : (T)((a > T(0)) - (a < T(0)));
      });
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)       \
  template <>                                 \
  void name<T, CPUContext>(                   \
      const int N,                            \
      const float beta,                       \
      const T* input,                         \
      const T* target,                        \
      T* loss,                                \
      CPUContext* ctx) {                      \
    _##name(N, T(beta), input, target, loss); \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Loss, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1LossGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
