#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

#define DEFINE_KERNEL_LAUNCHER(T)                                            \
  template <>                                                                \
  void Tanh<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    using EigenT = math::Traits<T>::eigen_type;                              \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);                            \
    Y = ConstEigenVectorArrayMap<EigenT>((const EigenT*)x, N).tanh();        \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void TanhGrad<T, CPUContext>(                                       \
      const int N, const T* dy, const T* y, T* dx, CPUContext* ctx) { \
    using EigenT = math::Traits<T>::eigen_type;                       \
    EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);                   \
    ConstEigenVectorArrayMap<EigenT> Y((const EigenT*)y, N);          \
    ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);        \
    dX = dY * (EigenT(1) - Y.square());                               \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
