#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _Im2Col2dNCHW(
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
    const T* im,
    T* col) {
  const auto HxW = H * W;
  for (int c = 0; c < C; ++c, im += HxW) {
    for (int h_k = 0; h_k < kernel_h; ++h_k) {
      for (int w_k = 0; w_k < kernel_w; ++w_k) {
        int h = -pad_h + h_k * dilation_h;
        for (int h_out = 0; h_out < out_h; ++h_out) {
          if (!math::utils::IsAGeZeroAndALtB(h, H)) {
            memset(col, 0, out_w * sizeof(T));
            col += out_w;
          } else {
            int w = -pad_w + w_k * dilation_w;
            for (int w_out = 0; w_out < out_w; ++w_out) {
              *(col++) = !math::utils::IsAGeZeroAndALtB(w, W)
                  ? convert::To<T>(0.f)
                  : im[h * W + w];
              w += stride_w;
            }
          }
          h += stride_h;
        }
      }
    }
  }
}

template <typename T>
void _Im2Col2dNHWC(
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
    const T* im,
    T* col) {
  for (int h_out = 0; h_out < out_h; ++h_out) {
    const int hstart = h_out * stride_h - pad_h;
    for (int w_out = 0; w_out < out_w; ++w_out) {
      const int wstart = w_out * stride_w - pad_w;
      for (int h_k = 0; h_k < kernel_h; ++h_k) {
        const int h = hstart + h_k * dilation_h;
        if (!math::utils::IsAGeZeroAndALtB(h, H)) {
          memset(col, 0, kernel_w * C * sizeof(T));
          col += kernel_w * C;
        } else {
          for (int w_k = 0; w_k < kernel_w; ++w_k) {
            const int w = wstart + w_k * dilation_w;
            if (!math::utils::IsAGeZeroAndALtB(w, W)) {
              memset(col, 0, C * sizeof(T));
              col += C;
            } else {
              const T* offset_im = im + (h * W + w) * C;
              for (int c = 0; c < C; ++c) {
                *(col++) = offset_im[c];
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void _Col2Im2dNCHW(
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
    const T* col,
    T* im) {
  const auto HxW = H * W;
  for (int c = 0; c < C; ++c, im += HxW) {
    for (int h_k = 0; h_k < kernel_h; ++h_k) {
      for (int w_k = 0; w_k < kernel_w; ++w_k) {
        int h = -pad_h + h_k * dilation_h;
        for (int h_out = 0; h_out < out_h; ++h_out) {
          if (!math::utils::IsAGeZeroAndALtB(h, H)) {
            col += out_w;
          } else {
            int w = -pad_w + w_k * dilation_w;
            for (int w_out = 0; w_out < out_w; ++w_out) {
              if (math::utils::IsAGeZeroAndALtB(w, W)) {
                im[h * W + w] += *col;
              }
              ++col;
              w += stride_w;
            }
          }
          h += stride_h;
        }
      }
    }
  }
}

template <typename T>
void _Col2Im2dNHWC(
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
    const T* col,
    T* im) {
  for (int h_out = 0; h_out < out_h; ++h_out) {
    const int hstart = -pad_h + stride_h * h_out;
    for (int w_out = 0; w_out < out_w; ++w_out) {
      const int wstart = -pad_w + stride_w * w_out;
      for (int h_k = 0; h_k < kernel_h; ++h_k) {
        const int h = hstart + h_k * dilation_h;
        if (!math::utils::IsAGeZeroAndALtB(h, H)) {
          col += kernel_w * C;
        } else {
          for (int w_k = 0; w_k < kernel_w; ++w_k) {
            const int w = wstart + w_k * dilation_w;
            if (!math::utils::IsAGeZeroAndALtB(w, W)) {
              col += C;
            } else {
              auto* offset_im = im + (h * W + w) * C;
              for (int c = 0; c < C; ++c) {
                offset_im[c] += *(col++);
              }
            }
          }
        }
      }
    }
  }
}

template <typename T, bool kTransposed>
void _Im2ColNdNCHW(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const T* x,
    T* y) {
  vec32_t col_dims = {channels}, in_strides(num_dims);
  col_dims.insert(col_dims.end(), kshape, kshape + num_dims);
  col_dims.insert(col_dims.end(), out_shape, out_shape + num_dims);
  math::utils::ComputeStrides(num_dims, in_shape, in_strides.data());
  const auto N = math::utils::Prod(col_dims);
  const auto S = math::utils::Prod(num_dims, in_shape);
  vec32_t index(col_dims.size(), 0);
  int32_t im_idx, r;
  for (int col_idx = 0; col_idx < N; ++col_idx) {
    bool is_padding = false;
    im_idx = index[0] * S;
    for (int d = num_dims - 1; d >= 0; --d) {
      r = -pads[d] + index[d + 1] * dilations[d];
      r += index[d + 1 + num_dims] * strides[d];
      if (!math::utils::IsAGeZeroAndALtB(r, in_shape[d])) {
        is_padding = true;
        break;
      }
      im_idx += r * in_strides[d];
    }
    if (!kTransposed) {
      y[col_idx] = is_padding ? T(0) : x[im_idx];
    } else if (!is_padding) {
      y[im_idx] += x[col_idx];
    }
    math::utils::IncreaseIndexInDims(
        col_dims.size(), col_dims.data(), index.data());
  }
}

template <typename T, bool kTransposed>
void _Im2ColNdNHWC(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const T* x,
    T* y) {
  vec32_t col_dims = {channels}, in_strides(num_dims);
  col_dims.insert(col_dims.begin(), kshape, kshape + num_dims);
  col_dims.insert(col_dims.begin(), out_shape, out_shape + num_dims);
  math::utils::ComputeStrides(num_dims, in_shape, in_strides.data());
  const auto N = math::utils::Prod(col_dims);
  vec32_t index(col_dims.size(), 0);
  int32_t im_idx, r;
  for (int col_idx = 0; col_idx < N; ++col_idx) {
    bool is_padding = false;
    im_idx = 0;
    for (int d = num_dims - 1; d >= 0; --d) {
      r = -pads[d] + index[d + num_dims] * dilations[d];
      r += index[d] * strides[d];
      if (!math::utils::IsAGeZeroAndALtB(r, in_shape[d])) {
        is_padding = true;
        break;
      }
      im_idx += r * in_strides[d];
    }
    im_idx = im_idx * channels + index.back();
    if (!kTransposed) {
      y[col_idx] = is_padding ? T(0) : x[im_idx];
    } else if (!is_padding) {
      y[im_idx] += x[col_idx];
    }
    math::utils::IncreaseIndexInDims(
        col_dims.size(), col_dims.data(), index.data());
  }
}

} // namespace

#define DISPATCH_CONV_KERNEL(name, ...)                  \
  if (data_format == "NCHW") {                           \
    name##NCHW(__VA_ARGS__);                             \
  } else if (data_format == "NHWC") {                    \
    name##NHWC(__VA_ARGS__);                             \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kTransposed, T)   \
  template <>                                          \
  void name<T, CPUContext>(                            \
      const int C,                                     \
      const int H,                                     \
      const int W,                                     \
      const int out_h,                                 \
      const int out_w,                                 \
      const int kernel_h,                              \
      const int kernel_w,                              \
      const int stride_h,                              \
      const int stride_w,                              \
      const int pad_h,                                 \
      const int pad_w,                                 \
      const int dilation_h,                            \
      const int dilation_w,                            \
      const string& data_format,                       \
      const T* x,                                      \
      T* y,                                            \
      CPUContext* ctx) {                               \
    if (kTransposed) {                                 \
      math::Set(C* H* W, convert::To<T>(0.f), y, ctx); \
    }                                                  \
    DISPATCH_CONV_KERNEL(                              \
        _##name,                                       \
        C,                                             \
        H,                                             \
        W,                                             \
        out_h,                                         \
        out_w,                                         \
        kernel_h,                                      \
        kernel_w,                                      \
        stride_h,                                      \
        stride_w,                                      \
        pad_h,                                         \
        pad_w,                                         \
        dilation_h,                                    \
        dilation_w,                                    \
        x,                                             \
        y);                                            \
  }

template <>
void Col2Im2d<float16, CPUContext>(
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
    float16* y,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, float16);
DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, float);
DEFINE_KERNEL_LAUNCHER(Im2Col2d, false, double);
DEFINE_KERNEL_LAUNCHER(Col2Im2d, true, float);
DEFINE_KERNEL_LAUNCHER(Col2Im2d, true, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_CONV_KERNEL

#define DISPATCH_CONV_KERNEL(name, kTransposed, T, ...)  \
  if (data_format == "NCHW") {                           \
    name##NCHW<T, kTransposed>(__VA_ARGS__);             \
  } else if (data_format == "NHWC") {                    \
    name##NHWC<T, kTransposed>(__VA_ARGS__);             \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kTransposed, T)             \
  template <>                                                    \
  void name<T, CPUContext>(                                      \
      const int num_dims,                                        \
      const int channels,                                        \
      const int* in_shape,                                       \
      const int* out_shape,                                      \
      const int* kshape,                                         \
      const int* strides,                                        \
      const int* pads,                                           \
      const int* dilations,                                      \
      const string& data_format,                                 \
      const T* x,                                                \
      T* y,                                                      \
      CPUContext* ctx) {                                         \
    if (kTransposed) {                                           \
      const auto in_dim = math::utils::Prod(num_dims, in_shape); \
      math::Set(channels* in_dim, T(0), y, ctx);                 \
    }                                                            \
    DISPATCH_CONV_KERNEL(                                        \
        _Im2ColNd,                                               \
        kTransposed,                                             \
        T,                                                       \
        num_dims,                                                \
        channels,                                                \
        in_shape,                                                \
        out_shape,                                               \
        kshape,                                                  \
        strides,                                                 \
        pads,                                                    \
        dilations,                                               \
        x,                                                       \
        y);                                                      \
  }

DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, float);
DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, double);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, float);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_CONV_KERNEL

#define DEFINE_KERNEL_LAUNCHER(name, kTransposed, T) \
  template <>                                        \
  void name<T, CPUContext>(                          \
      const int num_dims,                            \
      const int channels,                            \
      const int* in_shape,                           \
      const int* out_shape,                          \
      const int* kshape,                             \
      const int* strides,                            \
      const int* pads,                               \
      const int* dilations,                          \
      const string& data_format,                     \
      const T* x,                                    \
      T* y,                                          \
      CPUContext* ctx) {                             \
    CPU_FP16_NOT_SUPPORTED;                          \
  }

DEFINE_KERNEL_LAUNCHER(Im2ColNd, false, float16);
DEFINE_KERNEL_LAUNCHER(Col2ImNd, true, float16);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
