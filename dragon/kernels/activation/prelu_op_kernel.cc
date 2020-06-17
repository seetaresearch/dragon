#include "dragon/utils/eigen_utils.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _PRelu(const int count, const T* x, const T w, T* y) {
  EigenVectorArrayMap<T>(y, count) =
      ConstEigenVectorArrayMap<T>(x, count).unaryExpr(
          [&](T a) { return a > T(0) ? a : w * a; });
}

template <typename T>
void _PReluNCHW(
    const int N,
    const int C,
    const int S,
    const T* x,
    const T* w,
    T* y) {
  int k = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PRelu(S, x + k, w[j], y + k);
      k += S;
    }
  }
}

template <typename T>
void _PReluNHWC(
    const int N,
    const int C,
    const int S,
    const T* x,
    const T* w,
    T* y) {
  ConstEigenArrayMap<T> X(x, C, N * S);
  ConstEigenVectorArrayMap<T> W(w, C);
  EigenArrayMap<T>(y, C, N * S) = (X > T(0)).select(X, X.colwise() * W);
}

template <typename T>
void _PReluGrad(const int count, const T* dy, const T* x, const T w, T* dx) {
  EigenVectorArrayMap<T>(dx, count) = ConstEigenVectorArrayMap<T>(dy, count) *
      ConstEigenVectorArrayMap<T>(x, count).unaryExpr(
          [&](T a) { return a > T(0) ? T(1) : w; });
}

template <typename T>
void _PReluGradNCHW(
    const int N,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  int k = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PReluGrad(S, dy + k, x + k, w[j], dx + k);
      k += S;
    }
  }
}

template <typename T>
void _PReluGradNHWC(
    const int N,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    const T* w,
    T* dx) {
  ConstEigenArrayMap<T> dY(dy, C, N * S);
  ConstEigenArrayMap<T> X(x, C, N * S);
  ConstEigenVectorArrayMap<T> W(w, C);
  EigenArrayMap<T>(dx, C, N * S) = (X > T(0)).select(dY, dY.colwise() * W);
}

template <typename T>
void _PReluWGrad(const int count, const T* dy, const T* x, T* dw) {
  ConstEigenVectorArrayMap<T> dY(dy, count);
  ConstEigenVectorArrayMap<T> X(x, count);
  dw[0] = (X > T(0)).select(T(0), dY * X).sum();
}

template <typename T>
void _PReluWGradNCHW(
    const int N,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    T* dw) {
  T dwi;
  int k = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      _PReluWGrad(S, dy + k, x + k, &dwi);
      dw[j] += dwi;
      k += S;
    }
  }
}

template <typename T>
void _PReluWGradNHWC(
    const int N,
    const int C,
    const int S,
    const T* dy,
    const T* x,
    T* dw) {
  ConstEigenArrayMap<T> dY(dy, C, N * S);
  ConstEigenArrayMap<T> X(x, C, N * S);
  EigenVectorArrayMap<T>(dw, C) =
      (X > T(0)).select(T(0), dY * X).rowwise().sum();
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void PRelu<float16, CPUContext>(
    const int N,
    const int C,
    const int S,
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
    const int C,
    const int S,
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
    const int C,
    const int S,
    const string& data_format,
    const float16* dy,
    const float16* x,
    float16* dw,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                             \
  template <>                                                 \
  void PRelu<T, CPUContext>(                                  \
      const int N,                                            \
      const int C,                                            \
      const int S,                                            \
      const string& data_format,                              \
      const T* x,                                             \
      const T* w,                                             \
      T* y,                                                   \
      CPUContext* ctx) {                                      \
    if (C > 1) {                                              \
      if (data_format == "NCHW") {                            \
        _PReluNCHW(N, C, S, x, w, y);                         \
      } else if (data_format == "NHWC") {                     \
        _PReluNHWC(N, C, S, x, w, y);                         \
      } else {                                                \
        LOG(FATAL) << "Unknown data format: " << data_format; \
      }                                                       \
    } else {                                                  \
      _PRelu(N, x, w[0], y);                                  \
    }                                                         \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                        \
  template <>                                                 \
  void PReluGrad<T, CPUContext>(                              \
      const int N,                                            \
      const int C,                                            \
      const int S,                                            \
      const string& data_format,                              \
      const T* dy,                                            \
      const T* x,                                             \
      const T* w,                                             \
      T* dx,                                                  \
      CPUContext* ctx) {                                      \
    if (C > 1) {                                              \
      if (data_format == "NCHW") {                            \
        _PReluGradNCHW(N, C, S, dy, x, w, dx);                \
      } else if (data_format == "NHWC") {                     \
        _PReluGradNHWC(N, C, S, dy, x, w, dx);                \
      } else {                                                \
        LOG(FATAL) << "Unknown data format: " << data_format; \
      }                                                       \
    } else {                                                  \
      _PReluGrad(N, dy, x, w[0], dx);                         \
    }                                                         \
  }                                                           \
  template <>                                                 \
  void PReluWGrad<T, CPUContext>(                             \
      const int N,                                            \
      const int C,                                            \
      const int S,                                            \
      const string& data_format,                              \
      const T* dy,                                            \
      const T* x,                                             \
      T* dw,                                                  \
      CPUContext* ctx) {                                      \
    if (C > 1) {                                              \
      if (data_format == "NCHW") {                            \
        math::Set(C, T(0), dw, ctx);                          \
        _PReluWGradNCHW(N, C, S, dy, x, dw);                  \
      } else if (data_format == "NHWC") {                     \
        _PReluWGradNHWC(N, C, S, dy, x, dw);                  \
      } else {                                                \
        LOG(FATAL) << "Unknown data format: " << data_format; \
      }                                                       \
    } else {                                                  \
      _PReluWGrad(N, dy, x, dw);                              \
    }                                                         \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
