#include "dragon/utils/cast.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _MaxPool2dNCHW(
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
    int* mask,
    T* y) {
  const int HW = H * W;
  const int CHW = C * HW;
  const int count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T val;
  int hstart, hend, wstart, wend, xi, mxi;
  for (int i = 0; i < count; ++i) {
    hstart = idx[2] * stride_h - pad_h;
    wstart = idx[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + idx[0] * CHW + idx[1] * HW;
    mxi = -1;
    val = T(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        xi = h * W + w;
        if (offset_x[xi] > val) {
          val = offset_x[mxi = xi];
        }
      }
    }
    y[i] = val;
    mask[i] = mxi;
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _MaxPool2dNHWC(
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
    int* mask,
    T* y) {
  const int HWC = H * W * C;
  auto count = N * C * out_h * out_w;
  std::array<int, 4> idx = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T val;
  int hstart, hend, wstart, wend, xi, mxi;
  for (int i = 0; i < count; ++i) {
    hstart = idx[1] * stride_h - pad_h;
    wstart = idx[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + idx[0] * HWC;
    mxi = -1;
    val = T(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        xi = (h * W + w) * C + idx[3];
        if (offset_x[xi] > val) {
          val = offset_x[mxi = xi];
        }
      }
    }
    y[i] = val;
    mask[i] = mxi;
    utils::math::IncreaseIndexInDims(4, dims.data(), idx.data());
  }
}

template <typename T>
void _MaxPool2dGradNCHW(
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
    const int* mask,
    T* dx) {
  const int HW = H * W;
  const int CHW = C * HW;
  const int count = N * C * out_h * out_w;
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {N, C, out_h * out_w};
  for (int i = 0; i < count; ++i) {
    if (mask[i] != -1) {
      dx[idx[0] * CHW + idx[1] * HW + mask[i]] += dy[i];
    }
    utils::math::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

template <typename T>
void _MaxPool2dGradNHWC(
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
    const int* mask,
    T* dx) {
  const int HWC = H * W * C;
  auto count = N * C * out_h * out_w;
  std::array<int, 2> idx = {0, 0};
  std::array<int, 2> dims = {N, out_h * out_w * C};
  for (int i = 0; i < count; ++i) {
    if (mask[i] != -1) {
      dx[idx[0] * HWC + mask[i]] += dy[i];
    }
    utils::math::IncreaseIndexInDims(2, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void MaxPool2d<T, CPUContext>(                           \
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
      int* mask,                                           \
      T* y,                                                \
      CPUContext* ctx) {                                   \
    if (data_format == "NCHW") {                           \
      _MaxPool2dNCHW(                                      \
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
          mask,                                            \
          y);                                              \
    } else if (data_format == "NHWC") {                    \
      _MaxPool2dNHWC(                                      \
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
          mask,                                            \
          y);                                              \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                     \
  template <>                                              \
  void MaxPool2dGrad<T, CPUContext>(                       \
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
      const int* mask,                                     \
      T* dx,                                               \
      CPUContext* ctx) {                                   \
    math::Set(N* C* H* W, cast::to<T>(0.f), dx, ctx);      \
    if (data_format == "NCHW") {                           \
      _MaxPool2dGradNCHW(                                  \
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
          mask,                                            \
          dx);                                             \
    } else if (data_format == "NHWC") {                    \
      _MaxPool2dGradNHWC(                                  \
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
          mask,                                            \
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
