#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
float ComputeScale(T in_size, T out_size, bool align_corners) {
  if (align_corners) {
    return (float)(in_size - T(1)) / (float)(out_size - T(1));
  } else {
    return (float)in_size / (float)out_size;
  }
}

template <typename T>
float TransformCoordinate(
    const T coord_resized,
    const float scale,
    const bool align_corners) {
  if (align_corners) {
    return coord_resized * scale;
  } else {
    float coord_original = (coord_resized + 0.5f) * scale - 0.5f;
    return coord_original < 0.f ? 0.f : coord_original;
  }
}

template <typename T>
void _ResizeLinear2dNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* x,
    T* y) {
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const float h = TransformCoordinate(index[2], scale_h, align_corners);
    const float w = TransformCoordinate(index[3], scale_w, align_corners);
    const int ti = std::floor(h);
    const int li = std::floor(w);
    const int bi = (h < h_max) ? std::ceil(h) : h_max;
    const int ri = (w < w_max) ? std::ceil(w) : w_max;
    const float v = h - ti;
    const float u = w - li;
    const int offset = (index[0] * C + index[1]) * H;
    const float tl = convert::To<float>(x[(offset + ti) * W + li]);
    const float tr = convert::To<float>(x[(offset + ti) * W + ri]);
    const float bl = convert::To<float>(x[(offset + bi) * W + li]);
    const float br = convert::To<float>(x[(offset + bi) * W + ri]);
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[i] = convert::To<T>(t + (b - t) * v);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeLinear2dNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* x,
    T* y) {
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  for (int i = 0; i < NxHoxWoxC; ++i) {
    const float h = TransformCoordinate(index[1], scale_h, align_corners);
    const float w = TransformCoordinate(index[2], scale_w, align_corners);
    const int ti = std::floor(h);
    const int li = std::floor(w);
    const int bi = (h < h_max) ? std::ceil(h) : h_max;
    const int ri = (w < w_max) ? std::ceil(w) : w_max;
    const float v = h - ti;
    const float u = w - li;
    const int offset = index[0] * H * W * C + index[3];
    const float tl = convert::To<float>(x[offset + (ti * W + li) * C]);
    const float tr = convert::To<float>(x[offset + (ti * W + ri) * C]);
    const float bl = convert::To<float>(x[offset + (bi * W + li) * C]);
    const float br = convert::To<float>(x[offset + (bi * W + ri) * C]);
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[i] = convert::To<T>(t + (b - t) * v);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeLinear2dGradNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* dy,
    float* dx) {
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const float h = TransformCoordinate(index[2], scale_h, align_corners);
    const float w = TransformCoordinate(index[3], scale_w, align_corners);
    const int ti = std::floor(h);
    const int li = std::floor(w);
    const int bi = (h < h_max) ? std::ceil(h) : h_max;
    const int ri = (w < w_max) ? std::ceil(w) : w_max;
    const float v = h - ti;
    const float u = w - li;
    const int offset = (index[0] * C + index[1]) * H;
    const float dt = (1.f - v) * convert::To<float>(dy[i]);
    const float db = v * convert::To<float>(dy[i]);
    dx[(offset + ti) * W + li] += (1.f - u) * dt;
    dx[(offset + ti) * W + ri] += u * dt;
    dx[(offset + bi) * W + li] += (1.f - u) * db;
    dx[(offset + bi) * W + ri] += u * db;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
void _ResizeLinear2dGradNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* dy,
    float* dx) {
  const auto h_max = H - 1, w_max = W - 1;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  for (int i = 0; i < NxHoxWoxC; ++i) {
    const float h = TransformCoordinate(index[1], scale_h, align_corners);
    const float w = TransformCoordinate(index[2], scale_w, align_corners);
    const int ti = std::floor(h);
    const int li = std::floor(w);
    const int bi = (h < h_max) ? std::ceil(h) : h_max;
    const int ri = (w < w_max) ? std::ceil(w) : w_max;
    const float v = h - ti;
    const float u = w - li;
    const int offset = index[0] * H * W * C + index[3];
    const float dt = (1.f - v) * convert::To<float>(dy[i]);
    const float db = v * convert::To<float>(dy[i]);
    dx[offset + (ti * W + li) * C] += (1.f - u) * dt;
    dx[offset + (ti * W + ri) * C] += u * dt;
    dx[offset + (bi * W + li) * C] += (1.f - u) * db;
    dx[offset + (bi * W + ri) * C] += u * db;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

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
      const bool align_corners,                                  \
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
        ComputeScale(H, out_h, align_corners),                   \
        ComputeScale(W, out_w, align_corners),                   \
        align_corners,                                           \
        x,                                                       \
        y);                                                      \
  }

DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, float16, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, float, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, double, float); // Grad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_RESIZE_KERNEL

} // namespace kernels

} // namespace dragon
