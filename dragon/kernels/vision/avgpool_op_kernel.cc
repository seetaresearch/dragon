#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T val, area;
  int hstart, hend, wstart, wend;
  for (int i = 0; i < NxCxHoxWo; ++i) {
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    val = T(0);
    const T* offset_x = x + index[0] * CxHxW + index[1] * HxW;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        val += offset_x[h * W + w];
      }
    }
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T val, area;
  int hstart, hend, wstart, wend;
  for (int i = 0; i < NxHoxWoxC; ++i) {
    hstart = index[1] * stride_h - pad_h;
    wstart = index[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * HxWxC + index[3];
    val = T(0);
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        val += offset_x[(h * W + w) * C];
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T area;
  int hstart, hend, wstart, wend, xi;
  memset(dx, 0, sizeof(T) * N * CxHxW);
  for (int i = 0; i < NxCxHoxWo; ++i) {
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + index[0] * CxHxW + index[1] * HxW;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        offset_dx[h * W + w] += dy[i] / area;
      }
    }
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T area;
  int hstart, hend, wstart, wend, xi;
  memset(dx, 0, sizeof(T) * N * HxWxC);
  for (int i = 0; i < NxHoxWoxC; ++i) {
    hstart = index[1] * stride_h - pad_h;
    wstart = index[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (hend - hstart) * (wend - wstart);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + index[0] * HxWxC + index[3];
    for (int h = hstart; h < hend; ++h)
      for (int w = wstart; w < wend; ++w)
        offset_dx[(h * W + w) * C] += dy[i] / area;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
  }
}

template <typename T>
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
  T val, area;
  int dstart, dend, hstart, hend, wstart, wend;
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    dstart = index[2] * stride_d - pad_d;
    hstart = index[3] * stride_h - pad_h;
    wstart = index[4] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D + pad_d);
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    val = T(0);
    const T* offset_x = x + index[0] * CxDxHxW + index[1] * DxHxW;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          val += offset_x[(d * H + h) * W + w];
        }
      }
    }
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
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
  T val, area;
  int dstart, dend, hstart, hend, wstart, wend;
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    dstart = index[1] * stride_d - pad_d;
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D + pad_d);
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * DxHxWxC + index[4];
    val = T(0);
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          val += offset_x[((d * H + h) * W + w) * C];
        }
      }
    }
    y[i] = val / area;
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
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
  T area;
  int dstart, dend, hstart, hend, wstart, wend, xi;
  memset(dx, 0, sizeof(T) * N * CxDxHxW);
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    dstart = index[2] * stride_d - pad_d;
    hstart = index[3] * stride_h - pad_h;
    wstart = index[4] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D + pad_d);
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + index[0] * CxDxHxW + index[1] * DxHxW;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          offset_dx[((d * H) + h) * W + w] += dy[i] / area;
        }
      }
    }
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
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
  T area;
  int dstart, dend, hstart, hend, wstart, wend, xi;
  memset(dx, 0, sizeof(T) * N * DxHxWxC);
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    dstart = index[1] * stride_d - pad_d;
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D + pad_d);
    hend = std::min(hstart + kernel_h, H + pad_h);
    wend = std::min(wstart + kernel_w, W + pad_w);
    area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = std::min(dend, D);
    hend = std::min(hend, H);
    wend = std::min(wend, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    T* offset_dx = dx + index[0] * DxHxWxC + index[4];
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          offset_dx[((d * H + h) * W + w) * C] += dy[i] / area;
        }
      }
    }
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_POOL_KERNEL(name, ...)                  \
  if (data_format == "NCHW") {                           \
    name##NCHW(__VA_ARGS__);                             \
  } else if (data_format == "NHWC") {                    \
    name##NHWC(__VA_ARGS__);                             \
  } else {                                               \
    LOG(FATAL) << "Unknown DataFormat: " << data_format; \
  }

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const int C,                      \
      const int H,                      \
      const int W,                      \
      const int out_h,                  \
      const int out_w,                  \
      const int kernel_h,               \
      const int kernel_w,               \
      const int stride_h,               \
      const int stride_w,               \
      const int pad_h,                  \
      const int pad_w,                  \
      const string& data_format,        \
      const T* x,                       \
      T* y,                             \
      CPUContext* ctx) {                \
    DISPATCH_POOL_KERNEL(               \
        _##name,                        \
        N,                              \
        C,                              \
        H,                              \
        W,                              \
        out_h,                          \
        out_w,                          \
        kernel_h,                       \
        kernel_w,                       \
        stride_h,                       \
        stride_w,                       \
        pad_h,                          \
        pad_w,                          \
        x,                              \
        y);                             \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool2d, float);
DEFINE_KERNEL_LAUNCHER(AvgPool2d, double);
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, float); // AvgPool2dGrad
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, double); // AvgPool2dGrad
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T) \
  template <>                           \
  void name<T, CPUContext>(             \
      const int N,                      \
      const int C,                      \
      const int D,                      \
      const int H,                      \
      const int W,                      \
      const int out_d,                  \
      const int out_h,                  \
      const int out_w,                  \
      const int kernel_d,               \
      const int kernel_h,               \
      const int kernel_w,               \
      const int stride_d,               \
      const int stride_h,               \
      const int stride_w,               \
      const int pad_d,                  \
      const int pad_h,                  \
      const int pad_w,                  \
      const string& data_format,        \
      const T* x,                       \
      T* y,                             \
      CPUContext* ctx) {                \
    DISPATCH_POOL_KERNEL(               \
        _##name,                        \
        N,                              \
        C,                              \
        D,                              \
        H,                              \
        W,                              \
        out_d,                          \
        out_h,                          \
        out_w,                          \
        kernel_d,                       \
        kernel_h,                       \
        kernel_w,                       \
        stride_d,                       \
        stride_h,                       \
        stride_w,                       \
        pad_d,                          \
        pad_h,                          \
        pad_w,                          \
        x,                              \
        y);                             \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool3d, float);
DEFINE_KERNEL_LAUNCHER(AvgPool3d, double);
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, float); // AvgPool3dGrad
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, double); // AvgPool3dGrad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_POOL_KERNEL

} // namespace kernels

} // namespace dragon
