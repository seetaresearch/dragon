#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

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
    const T* filter,
    T* y) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const int base_offset = n * C + c;
      const int x_offset = base_offset * H * W;
      const int y_offset = base_offset * out_h * out_w;
      for (int h_out = 0; h_out < out_h; ++h_out) {
        const int hstart = h_out * stride_h - pad_h;
        for (int w_out = 0; w_out < out_w; ++w_out) {
          T val = T(0);
          int fi = c * kernel_h * kernel_w;
          const int wstart = w_out * stride_w - pad_w;
          for (int h_k = 0; h_k < kernel_h; ++h_k) {
            for (int w_k = 0; w_k < kernel_w; ++w_k) {
              const int h = hstart + h_k * dilation_h;
              const int w = wstart + w_k * dilation_w;
              if (math::utils::IsAGeZeroAndALtB(h, H) &&
                  math::utils::IsAGeZeroAndALtB(w, W)) {
                const int xi = x_offset + h * W + w;
                val += x[xi] * filter[fi];
              }
              ++fi;
            } // End w_k
          } // End h_k
          y[y_offset + h_out * out_w + w_out] = val;
        } // End w_out
      } // End h_out
    } // End c
  } // End n
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
    const T* filter,
    T* y) {
  for (int n = 0; n < N; ++n) {
    const int x_offset = n * H * W * C;
    const int y_offset = n * out_h * out_w * C;
    for (int h_out = 0; h_out < out_h; ++h_out) {
      const int hstart = h_out * stride_h - pad_h;
      for (int w_out = 0; w_out < out_w; ++w_out) {
        const int wstart = w_out * stride_w - pad_w;
        for (int c = 0; c < C; ++c) {
          T val = T(0);
          int fi = c * kernel_h * kernel_w;
          for (int h_k = 0; h_k < kernel_h; ++h_k) {
            for (int w_k = 0; w_k < kernel_w; ++w_k) {
              const int h = hstart + h_k * dilation_h;
              const int w = wstart + w_k * dilation_w;
              if (math::utils::IsAGeZeroAndALtB(h, H) &&
                  math::utils::IsAGeZeroAndALtB(w, W)) {
                const int xi = x_offset + (h * W + w) * C + c;
                val += x[xi] * filter[fi];
              }
              ++fi;
            } // End w_k
          } // End h_k
          y[y_offset + ((h_out * out_w) + w_out) * C + c] = val;
        } // End c
      } // End w_out
    } // End h_out
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
    const float16* filter,
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
    const float* filter,
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
      filter,
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
      const T* filter,                      \
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
      T* dfilter,                           \
      CPUContext* ctx) {                    \
    NOT_IMPLEMENTED;                        \
  }

DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_DATA_KERNEL

} // namespace kernels

} // namespace dragon
