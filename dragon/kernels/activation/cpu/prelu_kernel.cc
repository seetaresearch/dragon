#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _PRelu(const int N, const T alpha, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) = X.cwiseMax(T(0)) + X.cwiseMin(T(0)) * alpha;
}

template <typename T>
void _PRelu(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* x,
    const T* w,
    T* y) {
  if (data_format == "NHWC") {
    ConstEigenArrayMap<T> X(x, C, N * S);
    ConstEigenVectorArrayMap<T> W(w, C);
    EigenArrayMap<T>(y, C, N * S) = (X > T(0)).select(X, X.colwise() * W);
  } else {
    for (int i = 0; i < N * C; ++i) {
      _PRelu(S, w[i % C], x + i * S, y + i * S);
    }
  }
}

template <typename T>
void _PReluGrad(const int N, const T alpha, const T* dy, const T* x, T* dx) {
  ConstEigenVectorArrayMap<T> X(x, N);
  ConstEigenVectorArrayMap<T> dY(dy, N);
  EigenVectorArrayMap<T>(dx, N) = (X > T(0)).select(dY, dY * alpha);
}

template <typename T>
void _PReluGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  if (data_format == "NHWC") {
    ConstEigenArrayMap<T> dY(dy, C, N * S);
    ConstEigenArrayMap<T> X(x, C, N * S);
    ConstEigenVectorArrayMap<T> W(w, C);
    EigenArrayMap<T>(dx, C, N * S) = (X > T(0)).select(dY, dY.colwise() * W);
  } else {
    for (int i = 0; i < N * C; ++i) {
      _PReluGrad(S, w[i % C], dy + i * S, x + i * S, dx + i * S);
    }
  }
}

template <typename T>
T _PReluWGrad(const int N, const T* dy, const T* x, T* dw = nullptr) {
  ConstEigenVectorArrayMap<T> dY(dy, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  T ret = (X > T(0)).select(T(0), dY * X).sum();
  if (dw != nullptr) dw[0] = ret;
  return ret;
}

template <typename T>
void _PReluWGrad(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dw) {
  if (data_format == "NHWC") {
    ConstEigenArrayMap<T> dY(dy, C, N * S);
    ConstEigenArrayMap<T> X(x, C, N * S);
    EigenVectorArrayMap<T> dW(dw, C);
    dW = (X > T(0)).select(T(0), dY * X).rowwise().sum();
  } else {
    std::memset(dw, 0, sizeof(T) * C);
    for (int i = 0; i < N * C; ++i) {
      dw[i % C] += _PReluWGrad(S, dy + i * S, x + i * S);
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void PRelu<T, CPUContext>(                                                  \
      const int N,                                                            \
      const int S,                                                            \
      const int C,                                                            \
      const string& data_format,                                              \
      const T* x,                                                             \
      const T* w,                                                             \
      T* y,                                                                   \
      CPUContext* ctx) {                                                      \
    using PtrT = math::Traits<T>::eigen_type;                                 \
    if (C > 1) {                                                              \
      _PRelu(N, S, C, data_format, (const PtrT*)x, (const PtrT*)w, (PtrT*)y); \
    } else {                                                                  \
      _PRelu(N, ((const PtrT*)w)[0], (const PtrT*)x, (PtrT*)y);               \
    }                                                                         \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void PReluGrad<T, CPUContext>(                                      \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const string& data_format,                                      \
      const T* dy,                                                    \
      const T* x,                                                     \
      const T* w,                                                     \
      T* dx,                                                          \
      CPUContext* ctx) {                                              \
    if (C > 1) {                                                      \
      _PReluGrad(                                                     \
          N,                                                          \
          S,                                                          \
          C,                                                          \
          data_format,                                                \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(dy),   \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(x),    \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(w),    \
          reinterpret_cast<math::Traits<T>::eigen_type*>(dx));        \
    } else {                                                          \
      _PReluGrad(                                                     \
          N,                                                          \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(w)[0], \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(dy),   \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(x),    \
          reinterpret_cast<math::Traits<T>::eigen_type*>(dx));        \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  void PReluWGrad<T, CPUContext>(                                     \
      const int N,                                                    \
      const int S,                                                    \
      const int C,                                                    \
      const string& data_format,                                      \
      const T* dy,                                                    \
      const T* x,                                                     \
      T* dw,                                                          \
      CPUContext* ctx) {                                              \
    if (C > 1) {                                                      \
      _PReluWGrad(                                                    \
          N,                                                          \
          S,                                                          \
          C,                                                          \
          data_format,                                                \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(dy),   \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(x),    \
          reinterpret_cast<math::Traits<T>::eigen_type*>(dw));        \
    } else {                                                          \
      _PReluWGrad(                                                    \
          N,                                                          \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(dy),   \
          reinterpret_cast<const math::Traits<T>::eigen_type*>(x),    \
          reinterpret_cast<math::Traits<T>::eigen_type*>(dw));        \
    }                                                                 \
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
