#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const int hstart = index[2] * stride_h - pad_h;
    const int wstart = index[3] * stride_w - pad_w;
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H), wend = std::min(wend, W);
    AccT val = AccT(0);
    const T* offset_x = x + index[0] * CxHxW + index[1] * HxW;
    for (int h = std::max(hstart, 0); h < hend; ++h) {
      for (int w = std::max(wstart, 0); w < wend; ++w) {
        val += convert::To<AccT>(offset_x[h * W + w]);
      }
    }
    y[i] = convert::To<T>(val / area);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  for (int i = 0; i < NxHoxWoxC; ++i) {
    const int hstart = index[1] * stride_h - pad_h;
    const int wstart = index[2] * stride_w - pad_w;
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H), wend = std::min(wend, W);
    const T* offset_x = x + index[0] * HxWxC + index[3];
    AccT val = AccT(0);
    for (int h = std::max(hstart, 0); h < hend; ++h) {
      for (int w = std::max(wstart, 0); w < wend; ++w) {
        val += convert::To<AccT>(offset_x[(h * W + w) * C]);
      }
    }
    y[i] = convert::To<T>(val / area);
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  memset(dx, 0, sizeof(T) * N * CxHxW);
  for (int i = 0; i < NxCxHoxWo; ++i) {
    const int hstart = index[2] * stride_h - pad_h;
    const int wstart = index[3] * stride_w - pad_w;
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H), wend = std::min(wend, W);
    const AccT val = convert::To<AccT>(dy[i]) / area;
    T* offset_dx = dx + index[0] * CxHxW + index[1] * HxW;
    for (int h = std::max(hstart, 0); h < hend; ++h) {
      for (int w = std::max(wstart, 0); w < wend; ++w) {
        const auto xi = h * W + w;
        offset_dx[xi] = convert::To<T>(val + convert::To<AccT>(offset_dx[xi]));
      }
    }
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  memset(dx, 0, sizeof(T) * N * HxWxC);
  for (int i = 0; i < NxHoxWoxC; ++i) {
    const int hstart = index[1] * stride_h - pad_h;
    const int wstart = index[2] * stride_w - pad_w;
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H), wend = std::min(wend, W);
    const AccT val = convert::To<AccT>(dy[i]) / area;
    T* offset_dx = dx + index[0] * HxWxC + index[3];
    for (int h = std::max(hstart, 0); h < hend; ++h) {
      for (int w = std::max(wstart, 0); w < wend; ++w) {
        const auto xi = (h * W + w) * C;
        offset_dx[xi] = convert::To<T>(val + convert::To<AccT>(offset_dx[xi]));
      }
    }
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
void _AvgPool3dNCHW(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const T* x,
    T* y) {
  const auto DxHxW = D * H * W;
  const auto CxDxHxW = C * DxHxW;
  const auto NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, C, out_d, out_h, out_w};
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    const int dstart = index[2] * stride_d - pad_d;
    const int hstart = index[3] * stride_h - pad_h;
    const int wstart = index[4] * stride_w - pad_w;
    int dend = std::min(dstart + kernel_d, D + pad_d);
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H), wend = std::min(wend, W);
    AccT val = AccT(0);
    const T* offset_x = x + index[0] * CxDxHxW + index[1] * DxHxW;
    for (int d = std::max(dstart, 0); d < dend; ++d) {
      for (int h = std::max(hstart, 0); h < hend; ++h) {
        for (int w = std::max(wstart, 0); w < wend; ++w) {
          val += convert::To<AccT>(offset_x[(d * H + h) * W + w]);
        }
      }
    }
    y[i] = convert::To<T>(val / area);
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
void _AvgPool3dNHWC(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const T* x,
    T* y) {
  const auto DxHxWxC = D * H * W * C;
  const auto NxDoxHoxWoxC = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, out_d, out_h, out_w, C};
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    const int dstart = index[1] * stride_d - pad_d;
    const int hstart = index[2] * stride_h - pad_h;
    const int wstart = index[3] * stride_w - pad_w;
    int dend = std::min(dstart + kernel_d, D + pad_d);
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H), wend = std::min(wend, W);
    const T* offset_x = x + index[0] * DxHxWxC + index[4];
    AccT val = AccT(0);
    for (int d = std::max(dstart, 0); d < dend; ++d) {
      for (int h = std::max(hstart, 0); h < hend; ++h) {
        for (int w = std::max(wstart, 0); w < wend; ++w) {
          val += convert::To<AccT>(offset_x[((d * H + h) * W + w) * C]);
        }
      }
    }
    y[i] = convert::To<T>(val / area);
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
void _AvgPool3dGradNCHW(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const T* dy,
    T* dx) {
  const auto DxHxW = D * H * W;
  const auto CxDxHxW = C * DxHxW;
  const auto NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, C, out_d, out_h, out_w};
  memset(dx, 0, sizeof(T) * N * CxDxHxW);
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    const int dstart = index[2] * stride_d - pad_d;
    const int hstart = index[3] * stride_h - pad_h;
    const int wstart = index[4] * stride_w - pad_w;
    int dend = std::min(dstart + kernel_d, D + pad_d);
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H), wend = std::min(wend, W);
    const AccT val = convert::To<AccT>(dy[i]) / area;
    T* offset_dx = dx + index[0] * CxDxHxW + index[1] * DxHxW;
    for (int d = std::max(dstart, 0); d < dend; ++d) {
      for (int h = std::max(hstart, 0); h < hend; ++h) {
        for (int w = std::max(wstart, 0); w < wend; ++w) {
          const auto xi = ((d * H) + h) * W + w;
          offset_dx[xi] =
              convert::To<T>(val + convert::To<AccT>(offset_dx[xi]));
        }
      }
    }
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T, typename AccT>
void _AvgPool3dGradNHWC(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const T* dy,
    T* dx) {
  const auto DxHxWxC = D * H * W * C;
  const auto NxDoxHoxWoxC = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, out_d, out_h, out_w, C};
  memset(dx, 0, sizeof(T) * N * DxHxWxC);
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    const int dstart = index[1] * stride_d - pad_d;
    const int hstart = index[2] * stride_h - pad_h;
    const int wstart = index[3] * stride_w - pad_w;
    int dend = std::min(dstart + kernel_d, D + pad_d);
    int hend = std::min(hstart + kernel_h, H + pad_h);
    int wend = std::min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    const AccT val = convert::To<AccT>(dy[i]) / area;
    T* offset_dx = dx + index[0] * DxHxWxC + index[4];
    for (int d = std::max(dstart, 0); d < dend; ++d) {
      for (int h = std::max(hstart, 0); h < hend; ++h) {
        for (int w = std::max(wstart, 0); w < wend; ++w) {
          const auto xi = ((d * H + h) * W + w) * C;
          offset_dx[xi] =
              convert::To<T>(val + convert::To<AccT>(offset_dx[xi]));
        }
      }
    }
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

} // namespace

#define DISPATCH_POOL_KERNEL(name, T, AccT, ...)         \
  if (data_format == "NCHW") {                           \
    name##NCHW<T, AccT>(__VA_ARGS__);                    \
  } else if (data_format == "NHWC") {                    \
    name##NHWC<T, AccT>(__VA_ARGS__);                    \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(name, T)    \
  template <>                              \
  void name<T, CPUContext>(                \
      const int N,                         \
      const int C,                         \
      const int H,                         \
      const int W,                         \
      const int out_h,                     \
      const int out_w,                     \
      const int kernel_h,                  \
      const int kernel_w,                  \
      const int stride_h,                  \
      const int stride_w,                  \
      const int pad_h,                     \
      const int pad_w,                     \
      const string& data_format,           \
      const T* x,                          \
      T* y,                                \
      CPUContext* ctx) {                   \
    DISPATCH_POOL_KERNEL(                  \
        _##name,                           \
        math::Traits<T>::scalar_type,      \
        math::Traits<T>::accumulator_type, \
        N,                                 \
        C,                                 \
        H,                                 \
        W,                                 \
        out_h,                             \
        out_w,                             \
        kernel_h,                          \
        kernel_w,                          \
        stride_h,                          \
        stride_w,                          \
        pad_h,                             \
        pad_w,                             \
        x,                                 \
        y);                                \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool2d, float16);
DEFINE_KERNEL_LAUNCHER(AvgPool2d, bfloat16);
DEFINE_KERNEL_LAUNCHER(AvgPool2d, float);
DEFINE_KERNEL_LAUNCHER(AvgPool2d, double);
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, float16);
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, bfloat16);
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, float);
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, double);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T)    \
  template <>                              \
  void name<T, CPUContext>(                \
      const int N,                         \
      const int C,                         \
      const int D,                         \
      const int H,                         \
      const int W,                         \
      const int out_d,                     \
      const int out_h,                     \
      const int out_w,                     \
      const int kernel_d,                  \
      const int kernel_h,                  \
      const int kernel_w,                  \
      const int stride_d,                  \
      const int stride_h,                  \
      const int stride_w,                  \
      const int pad_d,                     \
      const int pad_h,                     \
      const int pad_w,                     \
      const string& data_format,           \
      const T* x,                          \
      T* y,                                \
      CPUContext* ctx) {                   \
    DISPATCH_POOL_KERNEL(                  \
        _##name,                           \
        math::Traits<T>::scalar_type,      \
        math::Traits<T>::accumulator_type, \
        N,                                 \
        C,                                 \
        D,                                 \
        H,                                 \
        W,                                 \
        out_d,                             \
        out_h,                             \
        out_w,                             \
        kernel_d,                          \
        kernel_h,                          \
        kernel_w,                          \
        stride_d,                          \
        stride_h,                          \
        stride_w,                          \
        pad_d,                             \
        pad_h,                             \
        pad_w,                             \
        x,                                 \
        y);                                \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool3d, float16);
DEFINE_KERNEL_LAUNCHER(AvgPool3d, bfloat16);
DEFINE_KERNEL_LAUNCHER(AvgPool3d, float);
DEFINE_KERNEL_LAUNCHER(AvgPool3d, double);
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, float16);
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, bfloat16);
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, float);
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_POOL_KERNEL

} // namespace kernels

} // namespace dragon
