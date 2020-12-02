#include "dragon/utils/conversions.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _AvgPool2dNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const T* x,
    T* y) {
  const int HW = H * W;
  const int CHW = C * HW;
  const int count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T val, area;
  int hstart, hend, wstart, wend;
  for (int i = 0; i < count; ++i) {
    hstart = idx[2] * stride_h - pad_h;
    wstart = idx[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    val = T(0);
    const T* offset_x = x + idx[0] * CHW + idx[1] * HW;
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        val += offset_x[h * W + w];
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _AvgPool2dNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const T* x,
    T* y) {
  const int HWC = H * W * C;
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T val, area;
  int hstart, hend, wstart, wend;
  for (int i = 0; i < count; ++i) {
    hstart = idx[1] * stride_h - pad_h;
    wstart = idx[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + idx[0] * HWC + idx[3];
    val = T(0);
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        val += offset_x[(h * W + w) * C];
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _AvgPool2dGradNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const T* dy,
    T* dx) {
  const int HW = H * W;
  const int CHW = C * HW;
  const int count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T area;
  int hstart, hend, wstart, wend, xi;
  for (int i = 0; i < count; ++i) {
    hstart = idx[2] * stride_h - pad_h;
    wstart = idx[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + idx[0] * CHW + idx[1] * HW;
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        offset_dx[h * W + w] += dy[i] / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _AvgPool2dGradNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const T* dy,
    T* dx) {
  const int HWC = H * W * C;
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T area;
  int hstart, hend, wstart, wend, xi;
  for (int i = 0; i < count; ++i) {
    hstart = idx[1] * stride_h - pad_h;
    wstart = idx[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + idx[0] * HWC + idx[3];
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        offset_dx[(h * W + w) * C] += dy[i] / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void AvgPool2d<T, CPUContext>(                           \
      const int N,                                         \
      const int C,                                         \
      const int H,                                         \
      const int W,                                         \
      const int out_h,                                     \
      const int out_w,                                     \
      const int kernel_h,                                  \
      const int kernel_w,                                  \
      const int stride_h,                                  \
      const int stride_w,                                  \
      const int pad_h,                                     \
      const int pad_w,                                     \
      const string& data_format,                           \
      const T* x,                                          \
      T* y,                                                \
      CPUContext* ctx) {                                   \
    if (data_format == "NCHW") {                           \
      _AvgPool2dNCHW(                                      \
          N,                                               \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          x,                                               \
          y);                                              \
    } else if (data_format == "NHWC") {                    \
      _AvgPool2dNHWC(                                      \
          N,                                               \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          x,                                               \
          y);                                              \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                     \
  template <>                                              \
  void AvgPool2dGrad<T, CPUContext>(                       \
      const int N,                                         \
      const int C,                                         \
      const int H,                                         \
      const int W,                                         \
      const int out_h,                                     \
      const int out_w,                                     \
      const int kernel_h,                                  \
      const int kernel_w,                                  \
      const int stride_h,                                  \
      const int stride_w,                                  \
      const int pad_h,                                     \
      const int pad_w,                                     \
      const string& data_format,                           \
      const T* dy,                                         \
      T* dx,                                               \
      CPUContext* ctx) {                                   \
    math::Set(N* C* H* W, convert::To<T>(0.f), dx, ctx);   \
    if (data_format == "NCHW") {                           \
      _AvgPool2dGradNCHW(                                  \
          N,                                               \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          dy,                                              \
          dx);                                             \
    } else if (data_format == "NHWC") {                    \
      _AvgPool2dGradNHWC(                                  \
          N,                                               \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          dy,                                              \
          dx);                                             \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
