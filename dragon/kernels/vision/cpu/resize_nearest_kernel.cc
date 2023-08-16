#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _ResizeNearest2dNCHW(
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
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const int h = std::min(int(index[2] * scale_h), h_max);
    const int w = std::min(int(index[3] * scale_w), w_max);
    y[i] = x[(((index[0] * C) + index[1]) * H + h) * W + w];
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest2dNHWC(
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
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxHoxWo = N * out_h * out_w;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, out_h, out_w};
  for (int i = 0; i < NxHoxWo; ++i) {
    const int h = std::min(int(index[1] * scale_h), h_max);
    const int w = std::min(int(index[2] * scale_w), w_max);
    memcpy(y + i * C, x + (((index[0] * H) + h) * W + w) * C, C * sizeof(T));
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest2dGradNCHW(
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
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const int h = std::min(int(index[2] * scale_h), h_max);
    const int w = std::min(int(index[3] * scale_w), w_max);
    dx[(((index[0] * C) + index[1]) * H + h) * W + w] +=
        convert::To<float>(dy[i]);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest2dGradNHWC(
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
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  for (int i = 0; i < NxHoxWoxC; ++i) {
    const int h = std::min(int(index[1] * scale_h), h_max);
    const int w = std::min(int(index[2] * scale_w), w_max);
    dx[(((index[0] * H) + h) * W + w) * C + index[3]] +=
        convert::To<float>(dy[i]);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest3dNCHW(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  const auto d_max = D - 1, h_max = H - 1, w_max = W - 1;
  const auto NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, C, out_d, out_h, out_w};
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    const int d = std::min(int(index[2] * scale_d), d_max);
    const int h = std::min(int(index[3] * scale_h), h_max);
    const int w = std::min(int(index[4] * scale_w), w_max);
    y[i] = x[((((index[0] * C) + index[1]) * D + d) * H + h) * W + w];
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest3dNHWC(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  const auto d_max = D - 1, h_max = H - 1, w_max = W - 1;
  const auto NxDoxHoxWo = N * out_d * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_d, out_h, out_w};
  for (int i = 0; i < NxDoxHoxWo; ++i) {
    const int d = std::min(int(index[1] * scale_d), d_max);
    const int h = std::min(int(index[2] * scale_h), h_max);
    const int w = std::min(int(index[3] * scale_w), w_max);
    memcpy(
        y + i * C,
        x + ((((index[0] * D + d) * H) + h) * W + w) * C,
        C * sizeof(T));
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest3dGradNCHW(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  const auto d_max = D - 1, h_max = H - 1, w_max = W - 1;
  const auto NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, C, out_d, out_h, out_w};
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    const int d = std::min(int(index[2] * scale_d), d_max);
    const int h = std::min(int(index[3] * scale_h), h_max);
    const int w = std::min(int(index[4] * scale_w), w_max);
    dx[((((index[0] * C) + index[1]) * D + d) * H + h) * W + w] +=
        convert::To<float>(dy[i]);
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeNearest3dGradNHWC(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  const auto d_max = D - 1, h_max = H - 1, w_max = W - 1;
  const auto NxDoxHoxWoxC = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, out_d, out_h, out_w, C};
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    const int d = std::min(int(index[1] * scale_d), d_max);
    const int h = std::min(int(index[2] * scale_h), h_max);
    const int w = std::min(int(index[3] * scale_w), w_max);
    dx[((((index[0] * D) + d) * H + h) * W + w) * C + index[3]] +=
        convert::To<float>(dy[i]);
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

} // namespace

#define DISPATCH_RESIZE_KERNEL(name, ...)                \
  if (data_format == "NCHW") {                           \
    name##NCHW(__VA_ARGS__);                             \
  } else if (data_format == "NHWC") {                    \
    name##NHWC(__VA_ARGS__);                             \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kBackward, InputT, OutputT) \
  template <>                                                    \
  void name<InputT, CPUContext>(                                 \
      const int N,                                               \
      const int C,                                               \
      const int H,                                               \
      const int W,                                               \
      const int out_h,                                           \
      const int out_w,                                           \
      const string& data_format,                                 \
      const InputT* x,                                           \
      OutputT* y,                                                \
      CPUContext* ctx) {                                         \
    if (kBackward) {                                             \
      math::Set(N* C* H* W, convert::To<OutputT>(0.f), y, ctx);  \
    }                                                            \
    DISPATCH_RESIZE_KERNEL(                                      \
        _##name,                                                 \
        N,                                                       \
        C,                                                       \
        H,                                                       \
        W,                                                       \
        out_h,                                                   \
        out_w,                                                   \
        (float)H / (float)out_h,                                 \
        (float)W / (float)out_w,                                 \
        x,                                                       \
        y);                                                      \
  }

DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, bfloat16, bfloat16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, float16, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, bfloat16, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, double, float);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, kBackward, InputT, OutputT)   \
  template <>                                                      \
  void name<InputT, CPUContext>(                                   \
      const int N,                                                 \
      const int C,                                                 \
      const int D,                                                 \
      const int H,                                                 \
      const int W,                                                 \
      const int out_d,                                             \
      const int out_h,                                             \
      const int out_w,                                             \
      const string& data_format,                                   \
      const InputT* x,                                             \
      OutputT* y,                                                  \
      CPUContext* ctx) {                                           \
    if (kBackward) {                                               \
      math::Set(N* C* D* H* W, convert::To<OutputT>(0.f), y, ctx); \
    }                                                              \
    DISPATCH_RESIZE_KERNEL(                                        \
        _##name,                                                   \
        N,                                                         \
        C,                                                         \
        D,                                                         \
        H,                                                         \
        W,                                                         \
        out_d,                                                     \
        out_h,                                                     \
        out_w,                                                     \
        (float)D / (float)out_d,                                   \
        (float)H / (float)out_h,                                   \
        (float)W / (float)out_w,                                   \
        x,                                                         \
        y);                                                        \
  }

DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, bfloat16, bfloat16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, float16, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, bfloat16, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, double, float);
#undef DISPATCH_RESIZE_KERNEL

} // namespace kernels

} // namespace dragon
