#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _CosGrad(const int N, const T* dy, const T* x, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      (-ConstEigenVectorArrayMap<T>(x, N).sin());
}

template <typename T>
void _SinGrad(const int N, const T* dy, const T* x, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      ConstEigenVectorArrayMap<T>(x, N).cos();
}

template <typename T>
void _ReciprocalGrad(const int N, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      (-ConstEigenVectorArrayMap<T>(y, N).square());
}

template <typename T>
void _RsqrtGrad(const int N, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, N) = ConstEigenVectorArrayMap<T>(dy, N) *
      (T(-0.5) * ConstEigenVectorArrayMap<T>(y, N).cube());
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name##Grad<T, CPUContext>(                                     \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    _##name##Grad(N, dy, y, dx);                                      \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name##Grad<T, CPUContext>(                                     \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    CPU_FP16_NOT_SUPPORTED;                                           \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float16);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
