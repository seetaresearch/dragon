#include "dragon/kernels/math/op_kernels.h"
#include "dragon/utils/math/types.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _CosGrad(const int N, const T* dy, const T* x, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T>(dx, N) = dY * (-X.sin());
}

template <typename T>
void _SinGrad(const int N, const T* dy, const T* x, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T>(dx, N) = dY * X.cos();
}

template <typename T>
void _ReciprocalGrad(const int N, const T* dy, const T* y, T* dx) {
  ConstEigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T>(dx, N) = dY * (-Y.square());
}

template <typename T>
void _RsqrtGrad(const int N, const T* dy, const T* y, T* dx) {
  ConstEigenVectorArrayMap<T> Y(y, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T>(dx, N) = dY * (T(-0.5) * Y.cube());
}

} // namespace

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                            \
  template <>                                                           \
  void name##Grad<T, CPUContext>(                                       \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) {   \
    using EigenT = math::Traits<T>::eigen_type;                         \
    _##name##Grad(N, (const EigenT*)dy, (const EigenT*)y, (EigenT*)dx); \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Cos, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Sin, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Reciprocal, double);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, float);
DEFINE_GRAD_KERNEL_LAUNCHER(Rsqrt, double);
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
