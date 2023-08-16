#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Softmax(const int N, const int S, const int C, const T* x, T* y) {
  if (S == 1) {
    EigenArrayMap<T> Y(y, C, N);
    ConstEigenArrayMap<T> X(x, C, N);
    Y = (X.rowwise() - X.colwise().maxCoeff()).exp();
    Y = Y.rowwise() / Y.colwise().sum();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      const auto ptr_offset = offset + j;
      const auto ptr_stride = EigenInnerStride(S);
      EigenStridedVectorArrayMap<T> Y(y + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> X(x + ptr_offset, 1, C, ptr_stride);
      Y = (X - X.maxCoeff()).exp();
      Y /= Y.sum();
    }
  }
}

template <typename T>
void _LogSoftmax(const int N, const int S, const int C, const T* x, T* y) {
  if (S == 1) {
    EigenArrayMap<T> Y(y, C, N);
    ConstEigenArrayMap<T> X(x, C, N);
    Y = X.rowwise() - X.colwise().maxCoeff();
    Y = Y.rowwise() - Y.exp().colwise().sum().log();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      const auto ptr_offset = offset + j;
      const auto ptr_stride = EigenInnerStride(S);
      EigenStridedVectorArrayMap<T> Y(y + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> X(x + ptr_offset, 1, C, ptr_stride);
      Y = X - X.maxCoeff();
      Y -= T(std::log(float(Y.exp().sum())));
    }
  }
}

template <typename T>
void _SoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  if (S == 1) {
    EigenArrayMap<T> dX(dx, C, N);
    ConstEigenArrayMap<T> Y(y, C, N);
    ConstEigenArrayMap<T> dY(dy, C, N);
    dX = (dY.rowwise() - (dY * Y).colwise().sum()) * Y;
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      const auto ptr_offset = offset + j;
      const auto ptr_stride = EigenInnerStride(S);
      EigenStridedVectorArrayMap<T> dX(dx + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> dY(dy + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> Y(y + ptr_offset, 1, C, ptr_stride);
      dX = (dY - (dY * Y).sum()) * Y;
    }
  }
}

template <typename T>
void _LogSoftmaxGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* y,
    T* dx) {
  if (S == 1) {
    EigenArrayMap<T> dX(dx, C, N);
    ConstEigenArrayMap<T> Y(y, C, N);
    ConstEigenArrayMap<T> dY(dy, C, N);
    dX = dY - Y.exp().rowwise() * dY.colwise().sum();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      const auto ptr_offset = offset + j;
      const auto ptr_stride = EigenInnerStride(S);
      EigenStridedVectorArrayMap<T> dX(dx + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> dY(dy + ptr_offset, 1, C, ptr_stride);
      ConstEigenStridedVectorArrayMap<T> Y(y + ptr_offset, 1, C, ptr_stride);
      dX = dY - Y.exp() * dY.sum();
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T)             \
  template <>                                       \
  void name<T, CPUContext>(                         \
      const int N,                                  \
      const int S,                                  \
      const int C,                                  \
      const T* x,                                   \
      T* y,                                         \
      CPUContext* ctx) {                            \
    using EigenT = math::Traits<T>::eigen_type;     \
    _##name(N, S, C, (const EigenT*)x, (EigenT*)y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T)                            \
  template <>                                                           \
  void name<T, CPUContext>(                                             \
      const int N,                                                      \
      const int S,                                                      \
      const int C,                                                      \
      const T* dy,                                                      \
      const T* y,                                                       \
      T* dx,                                                            \
      CPUContext* ctx) {                                                \
    using EigenT = math::Traits<T>::eigen_type;                         \
    _##name(N, S, C, (const EigenT*)dy, (const EigenT*)y, (EigenT*)dx); \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float16);
DEFINE_KERNEL_LAUNCHER(Softmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(Softmax, float);
DEFINE_KERNEL_LAUNCHER(Softmax, double);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, bfloat16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, bfloat16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
