#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _CosGrad(const int count, const T* dy, const T* x, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      (-ConstEigenVectorArrayMap<T>(x, count).sin());
}

template <typename T>
void _SinGrad(const int count, const T* dy, const T* x, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(x, count).cos();
}

template <typename T>
void _ReciprocalGrad(const int count, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      (-ConstEigenVectorArrayMap<T>(y, count).square());
}

template <typename T>
void _RsqrtGrad(const int count, const T* dy, const T* y, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      (T(-0.5) * ConstEigenVectorArrayMap<T>(y, count).cube());
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                              \
  template <>                                                             \
  void name##Grad<T, CPUContext>(                                         \
      const int count, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    _##name##Grad(count, dy, y, dx);                                      \
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

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                              \
  template <>                                                             \
  void name##Grad<T, CPUContext>(                                         \
      const int count, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    CPU_FP16_NOT_SUPPORTED;                                               \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float16);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
