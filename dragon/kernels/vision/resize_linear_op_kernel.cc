#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

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
void _ResizeLinearNCHW(
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
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  float h_in, w_in, u, v, t, b, tl, tr, bl, br;
  int ti, bi, li, ri, offset, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = TransformCoordinate(idx[2], scale_h, align_corners);
    w_in = TransformCoordinate(idx[3], scale_w, align_corners);
    ti = std::floor(h_in), li = std::floor(w_in);
    bi = (h_in < h_max) ? std::ceil(h_in) : h_max;
    ri = (w_in < w_max) ? std::ceil(w_in) : w_max;
    v = h_in - ti, u = w_in - li;
    offset = (idx[0] * C + idx[1]) * H;
    tl = (float)x[(offset + ti) * W + li];
    tr = (float)x[(offset + ti) * W + ri];
    bl = (float)x[(offset + bi) * W + li];
    br = (float)x[(offset + bi) * W + ri];
    t = tl + (tr - tl) * u;
    b = bl + (br - bl) * u;
    y[i] = static_cast<T>(t + (b - t) * v);
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeLinearNHWC(
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
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  float h_in, w_in, u, v, t, b, tl, tr, bl, br;
  int ti, bi, li, ri, offset, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = TransformCoordinate(idx[1], scale_h, align_corners);
    w_in = TransformCoordinate(idx[2], scale_w, align_corners);
    ti = std::floor(h_in), li = std::floor(w_in);
    bi = (h_in < h_max) ? std::ceil(h_in) : h_max;
    ri = (w_in < w_max) ? std::ceil(w_in) : w_max;
    v = h_in - ti, u = w_in - li;
    offset = idx[0] * H;
    tl = (float)x[((offset + ti) * W + li) * C + idx[3]];
    tr = (float)x[((offset + ti) * W + ri) * C + idx[3]];
    bl = (float)x[((offset + bi) * W + li) * C + idx[3]];
    br = (float)x[((offset + bi) * W + ri) * C + idx[3]];
    t = tl + (tr - tl) * u;
    b = bl + (br - bl) * u;
    y[i] = static_cast<T>(t + (b - t) * v);
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeLinearGradNCHW(
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
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  float h_in, w_in, u, v, dt, db, tl, tr, bl, br;
  int ti, bi, li, ri, offset, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = TransformCoordinate(idx[2], scale_h, align_corners);
    w_in = TransformCoordinate(idx[3], scale_w, align_corners);
    ti = std::floor(h_in), li = std::floor(w_in);
    bi = (h_in < h_max) ? std::ceil(h_in) : h_max;
    ri = (w_in < w_max) ? std::ceil(w_in) : w_max;
    v = h_in - ti, u = w_in - li;
    offset = (idx[0] * C + idx[1]) * H;
    dt = (1.f - v) * static_cast<float>(dy[i]);
    db = v * static_cast<float>(dy[i]);
    dx[(offset + ti) * W + li] += (1.f - u) * dt; // tl
    dx[(offset + ti) * W + ri] += u * dt; // tr
    dx[(offset + bi) * W + li] += (1.f - u) * db; // bl
    dx[(offset + bi) * W + ri] += u * db; // br
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _ResizeLinearGradNHWC(
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
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  float h_in, w_in, u, v, dt, db, tl, tr, bl, br;
  int ti, bi, li, ri, offset, h_max = H - 1, w_max = W - 1;
  for (int i = 0; i < count; ++i) {
    h_in = TransformCoordinate(idx[1], scale_h, align_corners);
    w_in = TransformCoordinate(idx[2], scale_w, align_corners);
    ti = std::floor(h_in), li = std::floor(w_in);
    bi = (h_in < h_max) ? std::ceil(h_in) : h_max;
    ri = (w_in < w_max) ? std::ceil(w_in) : w_max;
    v = h_in - ti, u = w_in - li;
    offset = idx[0] * H;
    dt = (1.f - v) * static_cast<float>(dy[i]);
    db = v * static_cast<float>(dy[i]);
    dx[((offset + ti) * W + li) * C + idx[3]] += (1.f - u) * dt; // tl
    dx[((offset + ti) * W + ri) * C + idx[3]] += u * dt; // tr
    dx[((offset + bi) * W + li) * C + idx[3]] += (1.f - u) * db; // bl
    dx[((offset + bi) * W + ri) * C + idx[3]] += u * db; // br
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ResizeLinear<float16, CPUContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const float16* x,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void ResizeLinearGrad<float16, CPUContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const float16* dy,
    float* dx,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
} // ResizeLinearGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void ResizeLinear<T, CPUContext>(                                         \
      const int N,                                                          \
      const int C,                                                          \
      const int H,                                                          \
      const int W,                                                          \
      const int out_h,                                                      \
      const int out_w,                                                      \
      const bool align_corners,                                             \
      const string& data_format,                                            \
      const T* x,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    auto scale_h = ComputeScale(H, out_h, align_corners);                   \
    auto scale_w = ComputeScale(W, out_w, align_corners);                   \
    if (data_format == "NCHW") {                                            \
      _ResizeLinearNCHW(                                                    \
          N, C, H, W, out_h, out_w, scale_h, scale_w, align_corners, x, y); \
    } else if (data_format == "NHWC") {                                     \
      _ResizeLinearNHWC(                                                    \
          N, C, H, W, out_h, out_w, scale_h, scale_w, align_corners, x, y); \
    } else {                                                                \
      LOG(FATAL) << "Unknown data format: " << data_format;                 \
    }                                                                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                        \
  template <>                                                                 \
  void ResizeLinearGrad<T, CPUContext>(                                       \
      const int N,                                                            \
      const int C,                                                            \
      const int H,                                                            \
      const int W,                                                            \
      const int out_h,                                                        \
      const int out_w,                                                        \
      const bool align_corners,                                               \
      const string& data_format,                                              \
      const T* dy,                                                            \
      float* dx,                                                              \
      CPUContext* ctx) {                                                      \
    auto scale_h = ComputeScale(H, out_h, align_corners);                     \
    auto scale_w = ComputeScale(W, out_w, align_corners);                     \
    math::Set(N* C* H* W, 0.f, dx, ctx);                                      \
    if (data_format == "NCHW") {                                              \
      _ResizeLinearGradNCHW(                                                  \
          N, C, H, W, out_h, out_w, scale_h, scale_w, align_corners, dy, dx); \
    } else if (data_format == "NHWC") {                                       \
      _ResizeLinearGradNHWC(                                                  \
          N, C, H, W, out_h, out_w, scale_h, scale_w, align_corners, dy, dx); \
    } else {                                                                  \
      LOG(FATAL) << "Unknown data format: " << data_format;                   \
    }                                                                         \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
