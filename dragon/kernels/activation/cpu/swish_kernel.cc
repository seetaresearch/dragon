#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Silu(const int N, const T* x, T* y) {
  using EigenT = typename math::Traits<T>::eigen_type;
  EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  Y = X / (EigenT(1) + (-X).exp());
}

template <typename T>
void _HardSwish(const int N, const T* x, T* y) {
  using EigenT = typename math::Traits<T>::eigen_type;
  const auto kAlpha = EigenT(0.166667), kBeta = EigenT(0.5);
  EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  Y = X * ((X * kAlpha + kBeta).cwiseMin(EigenT(1)).cwiseMax(EigenT(0)));
}

template <typename T>
void _SiluGrad(const int N, const T* dy, const T* x, T* dx) {
  using EigenT = typename math::Traits<T>::eigen_type;
  EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);
  dX = EigenT(1) / (EigenT(1) + (-X).exp());
  dX = dY * dX * (X + EigenT(1) - X * dX);
}

template <typename T>
void _HardSwishGrad(const int N, const T* dy, const T* x, T* dx) {
  using EigenT = typename math::Traits<T>::eigen_type;
  EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);
  // clang-format off
  dX = (X < EigenT(-3)).select(EigenT(0),
       (X < EigenT(3)).select(dY * (X * EigenT(0.333333) + EigenT(0.5)), dY));
  // clang-format on
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)                                      \
  template <>                                                                \
  void name<T, CPUContext>(const int N, const T* x, T* y, CPUContext* ctx) { \
    _##name(N, x, y);                                                        \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                          \
  template <>                                                         \
  void name<T, CPUContext>(                                           \
      const int N, const T* dy, const T* x, T* dx, CPUContext* ctx) { \
    _##name(N, dy, x, dx);                                            \
  }

DEFINE_KERNEL_LAUNCHER(Silu, float16);
DEFINE_KERNEL_LAUNCHER(Silu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Silu, float);
DEFINE_KERNEL_LAUNCHER(Silu, double);
DEFINE_KERNEL_LAUNCHER(HardSwish, float16);
DEFINE_KERNEL_LAUNCHER(HardSwish, bfloat16);
DEFINE_KERNEL_LAUNCHER(HardSwish, float);
DEFINE_KERNEL_LAUNCHER(HardSwish, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SiluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(HardSwishGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
