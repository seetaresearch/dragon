#include "dragon/kernels/array/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Range(const int N, const double start, const double delta, T* y) {
  for (int i = 0; i < N; ++i) {
    y[i] = convert::To<T>(start + double(i) * delta);
  }
}

template <typename T>
void _RowwiseLinSpace(
    const int N,
    const int C,
    const double* starts,
    const double* stops,
    T* y) {
  for (int i = 0; i < C; ++i) {
    const auto delta = (stops[i] - starts[i]) / double(N - 1);
    y[i] = convert::To<T>(starts[i]);
    if (N > 1) {
      y[i + (N - 1) * C] = convert::To<T>(stops[i]);
    }
    for (int j = 1; j < N - 1; ++j) {
      y[i + j * C] = convert::To<T>(starts[i] + double(j) * delta);
    }
  }
}

template <typename T>
void _ColwiseLinSpace(
    const int N,
    const int C,
    const double* starts,
    const double* stops,
    T* y) {
  for (int i = 0; i < N; ++i) {
    const auto delta = (stops[i] - starts[i]) / double(C - 1);
    auto* offset_y = y + i * C;
    offset_y[0] = convert::To<T>(starts[i]);
    if (C > 1) {
      offset_y[C - 1] = convert::To<T>(stops[i]);
    }
    for (int j = 1; j < C - 1; ++j) {
      offset_y[j] = convert::To<T>(starts[i] + double(j) * delta);
    }
  }
}

template <typename T>
void _SwapByKey(const int N, const uint32_t* r, T* y) {
  for (int i = 0; i < N; ++i) {
    std::swap(y[i], y[i + (r[i] % (N - i))]);
  }
}

template <typename T>
void _SetEyeUpper(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y) {
  const auto MxN = M * N;
  const int imax = std::min(M, N - k);
  if (imax <= 0) return;
  for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
    for (int i = 0; i < imax; ++i) {
      y[i * N + k + i] = convert::To<T>(1.f);
    }
    y += MxN;
  }
}

template <typename T>
void _SetEyeLower(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y) {
  const auto MxN = M * N;
  const int imax = std::min(M + k, N);
  if (imax <= 0) return;
  for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
    T* offset_y = y - k * N;
    for (int i = 0; i < imax; ++i) {
      offset_y[i * N + i] = convert::To<T>(1.f);
    }
    y += MxN;
  }
}

template <typename T>
void _SetTriluUpper(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y) {
  for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
    for (int i = 0; i < M; ++i) {
      const int jmax = std::min(i + k, N);
      for (int j = 0; j < jmax; ++j) {
        y[j] = convert::To<T>(0.f);
      }
      y += N;
    }
  }
}

template <typename T>
void _SetTriluLower(
    const int batch_size,
    const int M,
    const int N,
    const int k,
    T* y) {
  for (int batch_ind = 0; batch_ind < batch_size; ++batch_ind) {
    for (int i = 0; i < M; ++i) {
      const int jmin = std::max(i + k + 1, 0);
      for (int j = jmin; j < N; ++j) {
        y[j] = convert::To<T>(0.f);
      }
      y += N;
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T) \
  template <>                     \
  void Range<T, CPUContext>(      \
      const int N,                \
      const double start,         \
      const double delta,         \
      T* y,                       \
      CPUContext* ctx) {          \
    _Range(N, start, delta, y);   \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)               \
  template <>                                   \
  void LinSpace<T, CPUContext>(                 \
      const int N,                              \
      const int C,                              \
      const int axis,                           \
      const double* starts,                     \
      const double* stops,                      \
      T* y,                                     \
      CPUContext* ctx) {                        \
    if (axis == 0) {                            \
      _RowwiseLinSpace(N, C, starts, stops, y); \
    } else {                                    \
      _ColwiseLinSpace(N, C, starts, stops, y); \
    }                                           \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                              \
  template <>                                                  \
  void Permutation<T, CPUContext>(                             \
      const int N, const uint32_t* r, T* y, CPUContext* ctx) { \
    kernels::Range(N, 0.f, 1.f, y, ctx);                       \
    _SwapByKey(N, r, y);                                       \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                 \
  template <>                                                     \
  void SetEye<T, CPUContext>(                                     \
      const int batch_size,                                       \
      const int M,                                                \
      const int N,                                                \
      const int k,                                                \
      T* y,                                                       \
      CPUContext* ctx) {                                          \
    math::Set((batch_size * M * N), convert::To<T>(0.f), y, ctx); \
    if (k > 0) {                                                  \
      _SetEyeUpper(batch_size, M, N, k, y);                       \
    } else {                                                      \
      _SetEyeLower(batch_size, M, N, k, y);                       \
    }                                                             \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                \
  template <>                                    \
  void SetTrilu<T, CPUContext>(                  \
      const int batch_size,                      \
      const int M,                               \
      const int N,                               \
      const int k,                               \
      const int upper,                           \
      const T* x,                                \
      T* y,                                      \
      CPUContext* ctx) {                         \
    math::Copy((batch_size * M * N), x, y, ctx); \
    if (upper > 0) {                             \
      _SetTriluUpper(batch_size, M, N, k, y);    \
    } else {                                     \
      _SetTriluLower(batch_size, M, N, k, y);    \
    }                                            \
  }

DEFINE_KERNEL_LAUNCHER(bool);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
