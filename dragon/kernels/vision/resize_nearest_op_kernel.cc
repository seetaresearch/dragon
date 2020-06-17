#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _ResizeNearestNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  int h_in, w_in, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = std::min(int(idx[2] * scale_h), h_max);
    w_in = std::min(int(idx[3] * scale_w), w_max);
    y[i] = x[(((idx[0] * C) + idx[1]) * H + h_in) * W + w_in];
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeNearestNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  auto count = N * out_h * out_w;
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {N, out_h, out_w};
  int h_in, w_in, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = std::min(int(idx[1] * scale_h), h_max);
    w_in = std::min(int(idx[2] * scale_w), w_max);
    memcpy(
        y + i * C, x + (((idx[0] * H) + h_in) * W + w_in) * C, C * sizeof(T));
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeNearestGradNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  int h_in, w_in, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = std::min(int(idx[2] * scale_h), h_max);
    w_in = std::min(int(idx[3] * scale_w), w_max);
    dx[(((idx[0] * C) + idx[1]) * H + h_in) * W + w_in] += (float)dy[i];
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeNearestGradNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  int h_in, w_in, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = std::min(int(idx[1] * scale_h), h_max);
    w_in = std::min(int(idx[2] * scale_w), w_max);
    dx[(((idx[0] * H) + h_in) * W + w_in) * C + idx[3]] += (float)dy[i];
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ResizeNearestGrad<float16, CPUContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const float16* dy,
    float* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // ResizeNearestGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void ResizeNearest<T, CPUContext>(                                        \
      const int N,                                                          \
      const int C,                                                          \
      const int H,                                                          \
      const int W,                                                          \
      const int out_h,                                                      \
      const int out_w,                                                      \
      const string& data_format,                                            \
      const T* x,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    auto scale_h = (float)H / (float)out_h;                                 \
    auto scale_w = (float)W / (float)out_w;                                 \
    if (data_format == "NCHW") {                                            \
      _ResizeNearestNCHW(N, C, H, W, out_h, out_w, scale_h, scale_w, x, y); \
    } else if (data_format == "NHWC") {                                     \
      _ResizeNearestNHWC(N, C, H, W, out_h, out_w, scale_h, scale_w, x, y); \
    } else {                                                                \
      LOG(FATAL) << "Unknown data format: " << data_format;                 \
    }                                                                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                         \
  template <>                                                  \
  void ResizeNearestGrad<T, CPUContext>(                       \
      const int N,                                             \
      const int C,                                             \
      const int H,                                             \
      const int W,                                             \
      const int out_h,                                         \
      const int out_w,                                         \
      const string& data_format,                               \
      const T* dy,                                             \
      float* dx,                                               \
      CPUContext* ctx) {                                       \
    auto scale_h = (float)H / (float)out_h;                    \
    auto scale_w = (float)W / (float)out_w;                    \
    math::Set(N* C* H* W, 0.f, dx, ctx);                       \
    if (data_format == "NCHW") {                               \
      _ResizeNearestGradNCHW(                                  \
          N, C, H, W, out_h, out_w, scale_h, scale_w, dy, dx); \
    } else if (data_format == "NHWC") {                        \
      _ResizeNearestGradNHWC(                                  \
          N, C, H, W, out_h, out_w, scale_h, scale_w, dy, dx); \
    } else {                                                   \
      LOG(FATAL) << "Unknown data format: " << data_format;    \
    }                                                          \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
