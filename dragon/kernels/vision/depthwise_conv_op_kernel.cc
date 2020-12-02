#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
void _DepthwiseConv2dNCHW(
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
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* w,
    T* y) {
  T sum_val;
  int ih, iw, xi, wi;
  int yc_offset, xc_start, yc_start;
  int ih_start, yh_start, iw_start;
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      yc_offset = n * C + c;
      xc_start = yc_offset * H * W;
      yc_start = yc_offset * out_h;
      for (int oh = 0; oh < out_h; ++oh) {
        ih_start = oh * stride_h - pad_h;
        yh_start = (yc_start + oh) * out_w;
        for (int ow = 0; ow < out_w; ++ow) {
          sum_val = T(0);
          wi = c * kernel_h * kernel_w;
          iw_start = ow * stride_w - pad_w;
          for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
              ih = ih_start + kh * dilation_h;
              iw = iw_start + kw * dilation_w;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                xi = xc_start + ih * W + iw;
                sum_val += x[xi] * w[wi];
              }
              ++wi;
            } // End kw
          } // End kh
          y[yh_start + ow] = sum_val;
        } // End ow
      } // End oh
    }
  } // End c && n
}

template <typename T>
void _DepthwiseConv2dNHWC(
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
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* w,
    T* y) {
  T sum_val;
  int ih, iw, xi, wi;
  int xn_start, yn_start;
  int ih_start, yh_start;
  int iw_start, yw_start;
  for (int n = 0; n < N; ++n) {
    xn_start = n * H;
    yn_start = n * out_h;
    for (int oh = 0; oh < out_h; ++oh) {
      ih_start = oh * stride_h - pad_h;
      yh_start = (yn_start + oh) * out_w;
      for (int ow = 0; ow < out_w; ++ow) {
        iw_start = ow * stride_w - pad_w;
        yw_start = (yh_start + ow) * C;
        for (int c = 0; c < C; ++c) {
          sum_val = T(0);
          wi = c * kernel_h * kernel_w;
          for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
              ih = ih_start + kh * dilation_h;
              iw = iw_start + kw * dilation_w;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                xi = ((xn_start + ih) * W + iw) * C + c;
                sum_val += x[xi] * w[wi];
              }
              ++wi;
            } // End kw
          } // End kh
          y[yw_start + c] = sum_val;
        } // End c
      } // End ow
    } // End oh
  } // End n
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_DATA_KERNEL(name, ...)                  \
  if (data_format == "NCHW") {                           \
    name##NCHW(__VA_ARGS__);                             \
  } else if (data_format == "NHWC") {                    \
    name##NHWC(__VA_ARGS__);                             \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

template <>
void DepthwiseConv2d<float16, CPUContext>(
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
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const float16* x,
    const float16* w,
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

template <>
void DepthwiseConv2d<float, CPUContext>(
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
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const float* x,
    const float* w,
    float* y,
    CPUContext* ctx) {
  DISPATCH_DATA_KERNEL(
      _DepthwiseConv2d,
      N,
      C,
      H,
      W,
      out_h,
      out_w,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      x,
      w,
      y);
}

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)      \
  template <>                               \
  void DepthwiseConv2dGrad<T, CPUContext>(  \
      const int N,                          \
      const int C,                          \
      const int H,                          \
      const int W,                          \
      const int out_h,                      \
      const int out_w,                      \
      const int kernel_h,                   \
      const int kernel_w,                   \
      const int stride_h,                   \
      const int stride_w,                   \
      const int pad_h,                      \
      const int pad_w,                      \
      const int dilation_h,                 \
      const int dilation_w,                 \
      const string& data_format,            \
      const T* dy,                          \
      const T* w,                           \
      T* dx,                                \
      CPUContext* ctx) {                    \
    NOT_IMPLEMENTED;                        \
  }                                         \
  template <>                               \
  void DepthwiseConv2dWGrad<T, CPUContext>( \
      const int N,                          \
      const int C,                          \
      const int H,                          \
      const int W,                          \
      const int out_h,                      \
      const int out_w,                      \
      const int kernel_h,                   \
      const int kernel_w,                   \
      const int stride_h,                   \
      const int stride_w,                   \
      const int pad_h,                      \
      const int pad_w,                      \
      const int dilation_h,                 \
      const int dilation_w,                 \
      const string& data_format,            \
      const T* dy,                          \
      const T* x,                           \
      T* dw,                                \
      CPUContext* ctx) {                    \
    NOT_IMPLEMENTED;                        \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_DATA_KERNEL

} // namespace kernel

} // namespace dragon
