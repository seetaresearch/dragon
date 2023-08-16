#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Gelu(const int N, const T* x, T* y) {
  using AccT = typename math::Traits<T>::accumulator_type;
  const auto kRsqrt2 = AccT(0.7071067811865475);
  for (int i = 0; i < N; ++i) {
    const AccT val = convert::To<AccT>(x[i]);
    y[i] = convert::To<T>(val * (AccT(1) + erf(val * kRsqrt2)) * AccT(0.5));
  }
}

template <typename T>
void _ApproxGelu(const int N, const T* x, T* y) {
  using EigenT = typename math::Traits<T>::eigen_type;
  const auto kAlpha = EigenT(0.7978845608028654); // Sqrt(2/Pi)
  const auto kBeta = EigenT(0.035677408136300125); // Sqrt(2/Pi) * 0.044715
  EigenVectorArrayMap<EigenT> Y((EigenT*)y, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  Y = X * ((X * kAlpha + X.cube() * kBeta).tanh() + EigenT(1)) * EigenT(0.5);
}

template <typename T>
void _GeluGrad(const int N, const T* dy, const T* x, T* dx) {
  using EigenT = typename math::Traits<T>::eigen_type;
  using AccT = typename math::Traits<T>::accumulator_type;
  const auto kRsqrt2 = AccT(0.7071067811865475);
  const auto kAlpha = EigenT(0.3989422804014327); // 0.5 * Sqrt(2/Pi)
  EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);
  for (int i = 0; i < N; ++i) {
    dx[i] = convert::To<T>((1 + erf(convert::To<AccT>(x[i]) * kRsqrt2)) * 0.5);
  }
  dX = dY * (dX + X * ((EigenT(-0.5) * X.square()).exp() * kAlpha));
}

template <typename T>
void _ApproxGeluGrad(const int N, const T* dy, const T* x, T* dx) {
  using EigenT = typename math::Traits<T>::eigen_type;
  const auto kAlpha = EigenT(0.7978845608028654); // Sqrt(2/Pi)
  const auto kBeta = EigenT(0.035677408136300125); // Sqrt(2/Pi) * 0.044715
  const auto kGamma = EigenT(0.10703222440890037); // Sqrt(2/Pi) * 0.044715 * 3
  ConstEigenVectorArrayMap<EigenT> dY((const EigenT*)dy, N);
  ConstEigenVectorArrayMap<EigenT> X((const EigenT*)x, N);
  EigenVectorArrayMap<EigenT> Y((EigenT*)dx, N);
  EigenVectorArrayMap<EigenT> dX((EigenT*)dx, N);
  Y = (X * kAlpha + X.cube() * kBeta).tanh();
  dX = EigenT(1) + Y + (X - X * Y.square()) * (kGamma * X.square() + kAlpha);
  dX *= EigenT(0.5) * dY;
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

DEFINE_KERNEL_LAUNCHER(Gelu, float16);
DEFINE_KERNEL_LAUNCHER(Gelu, bfloat16);
DEFINE_KERNEL_LAUNCHER(Gelu, float);
DEFINE_KERNEL_LAUNCHER(Gelu, double);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, bfloat16);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, float);
DEFINE_KERNEL_LAUNCHER(ApproxGelu, double);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(GeluGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(ApproxGeluGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
