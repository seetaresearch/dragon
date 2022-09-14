#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Softmax(const int N, const int S, const int C, const T* x, T* y) {
  if (S == 1) {
    ConstEigenArrayMap<T> X(x, C, N);
    EigenArrayMap<T> Y(y, C, N);
    Y = (X.rowwise() - X.colwise().maxCoeff()).exp();
    Y = Y.rowwise() / Y.colwise().sum();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      ConstEigenStridedVectorArrayMap<T> X_vec(
          x + offset + j, 1, C, EigenInnerStride(S));
      EigenStridedVectorArrayMap<T> Y_vec(
          y + offset + j, 1, C, EigenInnerStride(S));
      Y_vec = (X_vec - X_vec.maxCoeff()).exp();
      Y_vec /= Y_vec.sum();
    }
  }
}

template <typename T>
void _LogSoftmax(const int N, const int S, const int C, const T* x, T* y) {
  if (S == 1) {
    ConstEigenArrayMap<T> X(x, C, N);
    EigenArrayMap<T> Y(y, C, N);
    Y = X.rowwise() - X.colwise().maxCoeff();
    Y = Y.rowwise() - Y.exp().colwise().sum().log();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      ConstEigenStridedVectorArrayMap<T> X_vec(
          x + offset + j, 1, C, EigenInnerStride(S));
      EigenStridedVectorArrayMap<T> Y_vec(
          y + offset + j, 1, C, EigenInnerStride(S));
      Y_vec = X_vec - X_vec.maxCoeff();
      Y_vec -= std::log(Y_vec.exp().sum());
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
    ConstEigenArrayMap<T> dY(dy, C, N);
    ConstEigenArrayMap<T> Y(y, C, N);
    EigenArrayMap<T> dX(dx, C, N);
    dX = (dY.rowwise() - (dY * Y).colwise().sum()) * Y;
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      ConstEigenStridedVectorArrayMap<T> dY_vec(
          dy + offset + j, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorArrayMap<T> Y_vec(
          y + offset + j, 1, C, EigenInnerStride(S));
      EigenStridedVectorArrayMap<T> dX_vec(
          dx + offset + j, 1, C, EigenInnerStride(S));
      dX_vec = (dY_vec - (dY_vec * Y_vec).sum()) * Y_vec;
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
    ConstEigenArrayMap<T> dY(dy, C, N);
    ConstEigenArrayMap<T> Y(y, C, N);
    EigenArrayMap<T> dX(dx, C, N);
    dX = dY - Y.exp().rowwise() * dY.colwise().sum();
    return;
  }
  for (int i = 0; i < N; ++i) {
    const auto offset = i * C * S;
    for (int j = 0; j < S; ++j) {
      ConstEigenStridedVectorArrayMap<T> dY_vec(
          dy + offset + j, 1, C, EigenInnerStride(S));
      ConstEigenStridedVectorArrayMap<T> Y_vec(
          y + offset + j, 1, C, EigenInnerStride(S));
      EigenStridedVectorArrayMap<T> dX_vec(
          dx + offset + j, 1, C, EigenInnerStride(S));
      dX_vec = dY_vec - Y_vec.exp() * dY_vec.sum();
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const int S,                      \
      const int C,                      \
      const T* x,                       \
      T* y,                             \
      CPUContext* ctx) {                \
    _##name(N, S, C, x, y);             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T) \
  template <>                                \
  void name<T, CPUContext>(                  \
      const int N,                           \
      const int S,                           \
      const int C,                           \
      const T* dy,                           \
      const T* y,                            \
      T* dx,                                 \
      CPUContext* ctx) {                     \
    _##name(N, S, C, dy, y, dx);             \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float);
DEFINE_KERNEL_LAUNCHER(Softmax, double);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, double);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, double);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const int S,                      \
      const int C,                      \
      const T* x,                       \
      T* y,                             \
      CPUContext* ctx) {                \
    CPU_FP16_NOT_SUPPORTED;             \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(name, T) \
  template <>                                \
  void name<T, CPUContext>(                  \
      const int N,                           \
      const int S,                           \
      const int C,                           \
      const T* dy,                           \
      const T* y,                            \
      T* dx,                                 \
      CPUContext* ctx) {                     \
    CPU_FP16_NOT_SUPPORTED;                  \
  }

DEFINE_KERNEL_LAUNCHER(Softmax, float16);
DEFINE_KERNEL_LAUNCHER(LogSoftmax, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(SoftmaxGrad, float16);
DEFINE_GRAD_KERNEL_LAUNCHER(LogSoftmaxGrad, float16);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
