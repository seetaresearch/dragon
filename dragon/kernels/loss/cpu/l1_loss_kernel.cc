#include "dragon/kernels/loss/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SmoothL1(const int N, const T beta, const T* diff, T* loss) {
  EigenVectorArrayMap<T>(loss, N) =
      ConstEigenVectorArrayMap<T>(diff, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? T(.5) * a * a / beta
                                  : std::abs(a) - T(.5) * beta;
      });
}

template <typename T>
void _SmoothL1Grad(const int N, const T beta, const T* diff, T* dx) {
  EigenVectorArrayMap<T>(dx, N) =
      ConstEigenVectorArrayMap<T>(diff, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? a / beta : (T)((a > T(0)) - (a < T(0)));
      });
}

} // namespace

template <>
void SmoothL1<float16, CPUContext>(
    const int N,
    const float beta,
    const float16* diff,
    float16* loss,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void SmoothL1Grad<float16, CPUContext>(
    const int N,
    const float beta,
    const float16* diff,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const float beta,                 \
      const T* diff,                    \
      T* loss,                          \
      CPUContext* ctx) {                \
    _##name(N, T(beta), diff, loss);    \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
