#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, C, out_h, out_w};
  T val;
  int hstart, hend, wstart, wend, xi, mask_val;
  for (int i = 0; i < NxCxHoxWo; ++i) {
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * CxHxW + index[1] * HxW;
    mask_val = -1;
    val = T(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        xi = h * W + w;
        if (offset_x[xi] > val) {
          val = offset_x[mask_val = xi];
        }
      }
    }
    y[i] = val;
    mask[i] = mask_val;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 4> index = {0, 0, 0, 0};
  std::array<int, 4> dims = {N, out_h, out_w, C};
  T val;
  int hstart, hend, wstart, wend, xi, mask_val;
  for (int i = 0; i < NxHoxWoxC; ++i) {
    hstart = index[1] * stride_h - pad_h;
    wstart = index[2] * stride_w - pad_w;
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * HxWxC;
    mask_val = -1;
    val = T(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        xi = (h * W + w) * C + index[3];
        if (offset_x[xi] > val) {
          val = offset_x[mask_val = xi];
        }
      }
    }
    y[i] = val;
    mask[i] = mask_val;
    math::utils::IncreaseIndexInDims(4, dims.data(), index.data());
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
  const auto HxW = H * W;
  const auto CxHxW = C * HxW;
  const auto NxCxHoxWo = N * C * out_h * out_w;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, out_h * out_w};
  memset(dx, 0, sizeof(T) * N * CxHxW);
  for (int i = 0; i < NxCxHoxWo; ++i) {
    if (mask[i] != -1) {
      dx[index[0] * CxHxW + index[1] * HxW + mask[i]] += dy[i];
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
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
  const auto HxWxC = H * W * C;
  const auto NxHoxWoxC = N * C * out_h * out_w;
  std::array<int, 2> index = {0, 0};
  std::array<int, 2> dims = {N, out_h * out_w * C};
  memset(dx, 0, sizeof(T) * N * HxWxC);
  for (int i = 0; i < NxHoxWoxC; ++i) {
    if (mask[i] != -1) {
      dx[index[0] * HxWxC + mask[i]] += dy[i];
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
  }
}

template <typename T>
void _MaxPool3dNCHW(
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
    int* mask,
    T* y) {
  const auto DxHxW = D * H * W;
  const auto CxDxHxW = C * DxHxW;
  const auto NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, C, out_d, out_h, out_w};
  T val;
  int dstart, dend, hstart, hend, wstart, wend, xi, mask_val;
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    dstart = index[2] * stride_d - pad_d;
    hstart = index[3] * stride_h - pad_h;
    wstart = index[4] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D);
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * CxDxHxW + index[1] * DxHxW;
    mask_val = -1;
    val = T(-FLT_MAX);
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          xi = (d * H + h) * W + w;
          if (offset_x[xi] > val) {
            val = offset_x[mask_val = xi];
          }
        }
      }
    }
    y[i] = val;
    mask[i] = mask_val;
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
void _MaxPool3dNHWC(
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
    int* mask,
    T* y) {
  const auto DxHxWxC = D * H * W * C;
  const auto NxDoxHoxWoxC = N * C * out_d * out_h * out_w;
  std::array<int, 5> index = {0, 0, 0, 0, 0};
  std::array<int, 5> dims = {N, out_d, out_h, out_w, C};
  T val;
  int dstart, dend, hstart, hend, wstart, wend, xi, mask_val;
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    dstart = index[1] * stride_d - pad_d;
    hstart = index[2] * stride_h - pad_h;
    wstart = index[3] * stride_w - pad_w;
    dend = std::min(dstart + kernel_d, D);
    hend = std::min(hstart + kernel_h, H);
    wend = std::min(wstart + kernel_w, W);
    dstart = std::max(dstart, 0);
    hstart = std::max(hstart, 0);
    wstart = std::max(wstart, 0);
    const T* offset_x = x + index[0] * DxHxWxC;
    mask_val = -1;
    val = T(-FLT_MAX);
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          xi = ((d * H + h) * W + w) * C + index[4];
          if (offset_x[xi] > val) {
            val = offset_x[mask_val = xi];
          }
        }
      }
    }
    y[i] = val;
    mask[i] = mask_val;
    math::utils::IncreaseIndexInDims(5, dims.data(), index.data());
  }
}

template <typename T>
void _MaxPool3dGradNCHW(
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
    const int* mask,
    T* dx) {
  const int DxHxW = D * H * W;
  const int CxDxHxW = C * DxHxW;
  const int NxCxDoxHoxWo = N * C * out_d * out_h * out_w;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, C, out_d * out_h * out_w};
  memset(dx, 0, sizeof(T) * N * CxDxHxW);
  for (int i = 0; i < NxCxDoxHoxWo; ++i) {
    if (mask[i] != -1) {
      dx[index[0] * CxDxHxW + index[1] * DxHxW + mask[i]] += dy[i];
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

template <typename T>
void _MaxPool3dGradNHWC(
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
    const int* mask,
    T* dx) {
  const int DxHxWxC = D * H * W * C;
  const auto NxDoxHoxWoxC = N * C * out_d * out_h * out_w;
  std::array<int, 2> index = {0, 0};
  std::array<int, 2> dims = {N, out_d * out_h * out_w * C};
  memset(dx, 0, sizeof(T) * N * DxHxWxC);
  for (int i = 0; i < NxDoxHoxWoxC; ++i) {
    if (mask[i] != -1) {
      dx[index[0] * DxHxWxC + mask[i]] += dy[i];
    }
    math::utils::IncreaseIndexInDims(2, dims.data(), index.data());
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
      int* mask,                        \
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
        mask,                           \
        y);                             \
  }

DEFINE_KERNEL_LAUNCHER(MaxPool2d, float);
DEFINE_KERNEL_LAUNCHER(MaxPool2d, double);
DEFINE_KERNEL_LAUNCHER(MaxPool2dGrad, float); // MaxPool2dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool2dGrad, double); // MaxPool2dGrad
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
      int* mask,                        \
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
        mask,                           \
        y);                             \
  }

DEFINE_KERNEL_LAUNCHER(MaxPool3d, float);
DEFINE_KERNEL_LAUNCHER(MaxPool3d, double);
DEFINE_KERNEL_LAUNCHER(MaxPool3dGrad, float); // MaxPool3dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool3dGrad, double); // MaxPool3dGrad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_POOL_KERNEL

} // namespace kernels

} // namespace dragon
