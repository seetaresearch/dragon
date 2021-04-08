#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

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

template <>
void _Softmax<float16>(
    const int N,
    const int S,
    const int C,
    const float16* x,
    float16* y) {
  CPU_FP16_NOT_SUPPORTED;
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

template <>
void _SoftmaxGrad<float16>(
    const int N,
    const int S,
    const int C,
    const float16* dy,
    const float16* y,
    float16* dx) {
  CPU_FP16_NOT_SUPPORTED;
} // SoftmaxGrad

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T) \
  template <>                     \
  void Softmax<T, CPUContext>(    \
      const int N,                \
      const int S,                \
      const int C,                \
      const T* x,                 \
      T* y,                       \
      CPUContext* ctx) {          \
    _Softmax(N, S, C, x, y);      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T) \
  template <>                          \
  void SoftmaxGrad<T, CPUContext>(     \
      const int N,                     \
      const int S,                     \
      const int C,                     \
      const T* dy,                     \
      const T* y,                      \
      T* dx,                           \
      CPUContext* ctx) {               \
    _SoftmaxGrad(N, S, C, dy, y, dx);  \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
