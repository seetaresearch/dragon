#include "dragon/utils/device/common_eigen.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _PRelu(const int N, const T alpha, const T* x, T* y) {
  ConstEigenVectorArrayMap<T> X(x, N);
  EigenVectorArrayMap<T>(y, N) = X.cwiseMax(T(0)) + X.cwiseMin(T(0)) * alpha;
}

template <typename T, StorageOrder kOrder>
void _PRelu(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* w,
    T* y) {
  if (kOrder == StorageOrder::NHWC) {
    ConstEigenArrayMap<T> X(x, C, N * S);
    ConstEigenVectorArrayMap<T> W(w, C);
    EigenArrayMap<T>(y, C, N * S) = (X > T(0)).select(X, X.colwise() * W);
    return;
  }
  int offset = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PRelu(S, w[j], x + offset, y + offset);
      offset += S;
    }
  }
}

template <typename T>
void _PReluGrad(const int N, const T alpha, const T* dy, const T* x, T* dx) {
  for (int i = 0; i < N; ++i) {
    dx[i] = dy[i] * (x[i] > T(0) ? T(1) : alpha);
  }
}

template <typename T, StorageOrder kOrder>
void _PReluGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  if (kOrder == StorageOrder::NHWC) {
    ConstEigenArrayMap<T> dY(dy, C, N * S);
    ConstEigenArrayMap<T> X(x, C, N * S);
    ConstEigenVectorArrayMap<T> W(w, C);
    EigenArrayMap<T>(dx, C, N * S) = (X > T(0)).select(dY, dY.colwise() * W);
    return;
  }
  int offset = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PReluGrad(S, w[j], dy + offset, x + offset, dx + offset);
      offset += S;
    }
  }
}

template <typename T>
void _PReluWGrad(const int N, const T* dy, const T* x, T* dw) {
  ConstEigenVectorArrayMap<T> dY(dy, N);
  ConstEigenVectorArrayMap<T> X(x, N);
  dw[0] = (X > T(0)).select(T(0), dY * X).sum();
}

template <typename T, StorageOrder kOrder>
void _PReluWGrad(
    const int N,
    const int S,
    const int C,
    const T* dy,
    const T* x,
    T* dw) {
  if (kOrder == StorageOrder::NHWC) {
    ConstEigenArrayMap<T> dY(dy, C, N * S);
    ConstEigenArrayMap<T> X(x, C, N * S);
    EigenVectorArrayMap<T>(dw, C) =
        (X > T(0)).select(T(0), dY * X).rowwise().sum();
    return;
  }
  T val;
  int offset = 0;
  std::memset(dw, 0, sizeof(T) * C);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PReluWGrad(S, dy + offset, x + offset, &val);
      dw[j] += val;
      offset += S;
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void PRelu<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const float16* x,
    const float16* w,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void PReluGrad<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const float16* dy,
    const float16* x,
    const float16* w,
    float16* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void PReluWGrad<float16, CPUContext>(
    const int N,
    const int S,
    const int C,
    const string& data_format,
    const float16* dy,
    const float16* x,
    float16* dw,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DISPATCH_CWISE_PRELU_KERNEL(name, T, ...)        \
  if (data_format == "NCHW") {                           \
    name<T, StorageOrder::NCHW>(__VA_ARGS__);            \
  } else if (data_format == "NHWC") {                    \
    name<T, StorageOrder::NHWC>(__VA_ARGS__);            \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(T)                               \
  template <>                                                   \
  void PRelu<T, CPUContext>(                                    \
      const int N,                                              \
      const int S,                                              \
      const int C,                                              \
      const string& data_format,                                \
      const T* x,                                               \
      const T* w,                                               \
      T* y,                                                     \
      CPUContext* ctx) {                                        \
    if (C > 1) {                                                \
      DISPATCH_CWISE_PRELU_KERNEL(_PRelu, T, N, S, C, x, w, y); \
    } else {                                                    \
      _PRelu(N, w[0], x, y);                                    \
    }                                                           \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                   \
  template <>                                                            \
  void PReluGrad<T, CPUContext>(                                         \
      const int N,                                                       \
      const int S,                                                       \
      const int C,                                                       \
      const string& data_format,                                         \
      const T* dy,                                                       \
      const T* x,                                                        \
      const T* w,                                                        \
      T* dx,                                                             \
      CPUContext* ctx) {                                                 \
    if (C > 1) {                                                         \
      DISPATCH_CWISE_PRELU_KERNEL(_PReluGrad, T, N, S, C, dy, x, w, dx); \
    } else {                                                             \
      _PReluGrad(N, w[0], dy, x, dx);                                    \
    }                                                                    \
  }                                                                      \
  template <>                                                            \
  void PReluWGrad<T, CPUContext>(                                        \
      const int N,                                                       \
      const int S,                                                       \
      const int C,                                                       \
      const string& data_format,                                         \
      const T* dy,                                                       \
      const T* x,                                                        \
      T* dw,                                                             \
      CPUContext* ctx) {                                                 \
    if (C > 1) {                                                         \
      DISPATCH_CWISE_PRELU_KERNEL(_PReluWGrad, T, N, S, C, dy, x, dw);   \
    } else {                                                             \
      _PReluWGrad(N, dy, x, dw);                                         \
    }                                                                    \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_CWISE_PRELU_KERNEL

} // namespace kernels

} // namespace dragon
