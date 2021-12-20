#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _SmoothL1(const int N, const T beta, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, N) =
      ConstEigenVectorArrayMap<T>(x, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? T(.5) * a * a / beta
                                  : std::abs(a) - T(.5) * beta;
      });
}

template <typename T>
void _SmoothL1Grad(const int N, const T beta, const T* x, T* y) {
  EigenVectorArrayMap<T>(y, N) =
      ConstEigenVectorArrayMap<T>(x, N).unaryExpr([&](T a) {
        return std::abs(a) < beta ? a / beta : (T)((a > T(0)) - (a < T(0)));
      });
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void SmoothL1<float16, CPUContext>(
    const int N,
    const float beta,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void SmoothL1Grad<float16, CPUContext>(
    const int N,
    const float beta,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(name, T)                                   \
  template <>                                                             \
  void name<T, CPUContext>(                                               \
      const int N, const float beta, const T* x, T* y, CPUContext* ctx) { \
    _##name(N, T(beta), x, y);                                            \
  }

DEFINE_KERNEL_LAUNCHER(SmoothL1, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1, double);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, float);
DEFINE_KERNEL_LAUNCHER(SmoothL1Grad, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
