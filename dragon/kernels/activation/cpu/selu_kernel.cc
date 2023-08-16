#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void Selu<T, CPUContext>(                                                    \
      const int N,                                                             \
      const float alpha,                                                       \
      const float gamma,                                                       \
      const T* x,                                                              \
      T* y,                                                                    \
      CPUContext* ctx) {                                                       \
    using EigenT = math::Traits<T>::eigen_type;                                \
    EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);                              \
    ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);                   \
    const auto scale = alpha * gamma;                                          \
    Y = (X > EigenT(0))                                                        \
            .select(EigenT(gamma) * X, EigenT(scale) * (X.exp() - EigenT(1))); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                         \
  template <>                                                                  \
  void SeluGrad<T, CPUContext>(                                                \
      const int N,                                                             \
      const float alpha,                                                       \
      const float gamma,                                                       \
      const T* dy,                                                             \
      const T* y,                                                              \
      T* dx,                                                                   \
      CPUContext* ctx) {                                                       \
    using EigenT = math::Traits<T>::eigen_type;                                \
    EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);                            \
    ConstEigenVectorArrayMap<EigenT> Y((const EigenT*)y, N);                   \
    ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);                 \
    const auto scale = alpha * gamma;                                          \
    dX = (Y > EigenT(0)).select(dY * EigenT(gamma), dY * (EigenT(scale) + Y)); \
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
