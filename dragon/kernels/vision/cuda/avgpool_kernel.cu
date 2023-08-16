#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T, typename AccT>
__global__ void _AvgPool2dNCHW(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int wout = yi % out_w;
    const int hout = yi / out_w % out_h;
    const int c = yi / out_w / out_h % C;
    const int n = yi / out_w / out_h / C;
    const int hstart = hout * stride_h - pad_h;
    const int wstart = wout * stride_w - pad_w;
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = min(hend, H), wend = min(wend, W);
    const T* offset_x = x + (n * C + c) * H * W;
    AccT val = AccT(0);
    for (int h = max(hstart, 0); h < hend; ++h) {
      for (int w = max(wstart, 0); w < wend; ++w) {
        val += math::utils::LDGC<AccT>(offset_x + h * W + w);
      }
    }
    y[yi] = val / area;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool2dNHWC(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int wout = yi / C % out_w;
    const int hout = yi / C / out_w % out_h;
    const int n = yi / C / out_w / out_h;
    const int hstart = hout * stride_h - pad_h;
    const int wstart = wout * stride_w - pad_w;
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const AccT area = (hend - hstart) * (wend - wstart);
    hend = min(hend, H), wend = min(wend, W);
    const T* offset_x = x + n * H * W * C + c;
    AccT val = AccT(0);
    for (int h = max(hstart, 0); h < hend; ++h) {
      for (int w = max(wstart, 0); w < wend; ++w) {
        val += math::utils::LDGC<AccT>(offset_x + (h * W + w) * C);
      }
    }
    y[yi] = val / area;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool2dGradNCHW(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int w = xi % W + pad_w;
    const int h = xi / W % H + pad_h;
    const int c = xi / W / H % C;
    const int n = xi / W / H / C;
    const int out_hstart = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
    const int out_wstart = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);
    const T* offset_dy = dy + (n * C + c) * out_h * out_w;
    AccT val = AccT(0);
    for (int hout = out_hstart; hout < out_hend; ++hout) {
      const int hstart = hout * stride_h - pad_h;
      const int hend = min(hstart + kernel_h, H + pad_h);
      for (int wout = out_wstart; wout < out_wend; ++wout) {
        const int wstart = wout * stride_w - pad_w;
        const int wend = min(wstart + kernel_w, W + pad_w);
        const AccT area = (hend - hstart) * (wend - wstart);
        val += math::utils::LDGC<AccT>(offset_dy + hout * out_w + wout) / area;
      }
    }
    dx[xi] = val;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool2dGradNHWC(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int c = xi % C;
    const int w = xi / C % W + pad_w;
    const int h = xi / C / W % H + pad_h;
    const int n = xi / C / W / H;
    const int out_hstart = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
    const int out_wstart = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);
    const T* offset_dy = dy + n * out_h * out_w * C + c;
    AccT val = AccT(0);
    for (int hout = out_hstart; hout < out_hend; ++hout) {
      const int hstart = hout * stride_h - pad_h;
      const int hend = min(hstart + kernel_h, H + pad_h);
      for (int wout = out_wstart; wout < out_wend; ++wout) {
        const int wstart = wout * stride_w - pad_w;
        const int wend = min(wstart + kernel_w, W + pad_w);
        const AccT area = (hend - hstart) * (wend - wstart); // clang-format off
        val += math::utils::LDGC<AccT>(offset_dy + (hout * out_w + wout) * C) / area;
      } // clang-format on
    }
    dx[xi] = val;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool3dNCHW(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / out_w;
    const int wout = yi % out_w;
    const int hout = tmp % out_h;
    tmp /= out_h;
    const int dout = tmp % out_d;
    tmp /= out_d;
    const int c = tmp % C;
    const int n = tmp / C;
    const int dstart = dout * stride_d - pad_d;
    const int hstart = hout * stride_h - pad_h;
    const int wstart = wout * stride_w - pad_w;
    int dend = min(dstart + kernel_d, D + pad_d);
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = min(dend, D), hend = min(hend, H), wend = min(wend, W);
    const T* offset_x = x + (n * C + c) * D * H * W;
    AccT val = AccT(0);
    for (int d = max(dstart, 0); d < dend; ++d) {
      for (int h = max(hstart, 0); h < hend; ++h) {
        for (int w = max(wstart, 0); w < wend; ++w) {
          val += math::utils::LDGC<AccT>(offset_x + (d * H + h) * W + w);
        }
      }
    }
    y[yi] = val / area;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool3dNHWC(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / C;
    const int c = yi % C;
    const int wout = tmp % out_w;
    tmp /= out_w;
    const int hout = tmp % out_h;
    tmp /= out_h;
    const int dout = tmp % out_d;
    const int n = tmp / out_d;
    const int dstart = dout * stride_d - pad_d;
    const int hstart = hout * stride_h - pad_h;
    const int wstart = wout * stride_w - pad_w;
    int dend = min(dstart + kernel_d, D + pad_d);
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
    dend = min(dend, D), hend = min(hend, H), wend = min(wend, W);
    const T* offset_x = x + n * D * H * W * C + c;
    AccT val = AccT(0);
    for (int d = max(dstart, 0); d < dend; ++d) {
      for (int h = max(hstart, 0); h < hend; ++h) {
        for (int w = max(wstart, 0); w < wend; ++w) {
          val += math::utils::LDGC<AccT>(offset_x + ((d * H + h) * W + w) * C);
        }
      }
    }
    y[yi] = val / area;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool3dGradNCHW(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int tmp = xi / W;
    const int w = xi % W + pad_w;
    const int h = tmp % H + pad_h;
    tmp /= H;
    const int d = tmp % D + pad_d;
    tmp /= D;
    const int c = tmp % C;
    const int n = tmp / C;
    const int out_dstart = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
    const int out_hstart = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
    const int out_wstart = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int out_dend = min(d / stride_d + 1, out_d);
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);
    const T* offset_dy = dy + (n * C + c) * out_d * out_h * out_w;
    AccT val = AccT(0);
    for (int dout = out_dstart; dout < out_dend; ++dout) {
      const int dstart = dout * stride_d - pad_d;
      const int dend = min(dstart + kernel_d, D + pad_d);
      for (int hout = out_hstart; hout < out_hend; ++hout) {
        const int hstart = hout * stride_h - pad_h;
        const int hend = min(hstart + kernel_h, H + pad_h);
        for (int wout = out_wstart; wout < out_wend; ++wout) {
          const int wstart = wout * stride_w - pad_w; // clang-format off
          const int wend = min(wstart + kernel_w, W + pad_w);
          const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
          val += math::utils::LDGC<AccT>(offset_dy + (dout * out_h + hout) * out_w + wout) / area;
        } // clang-format on
      }
    }
    dx[xi] = val;
  }
}

template <typename T, typename AccT>
__global__ void _AvgPool3dGradNHWC(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int tmp = xi / C;
    const int c = xi % C;
    const int w = tmp % W + pad_w;
    tmp /= W;
    const int h = tmp % H + pad_h;
    tmp /= H;
    const int d = tmp % D + pad_d;
    const int n = tmp / D;
    const int out_dstart = d < kernel_d ? 0 : (d - kernel_d) / stride_d + 1;
    const int out_hstart = h < kernel_h ? 0 : (h - kernel_h) / stride_h + 1;
    const int out_wstart = w < kernel_w ? 0 : (w - kernel_w) / stride_w + 1;
    const int out_dend = min(d / stride_d + 1, out_d);
    const int out_hend = min(h / stride_h + 1, out_h);
    const int out_wend = min(w / stride_w + 1, out_w);
    const T* offset_dy = dy + n * out_d * out_h * out_w * C + c;
    AccT val = AccT(0);
    for (int dout = out_dstart; dout < out_dend; ++dout) {
      const int dstart = dout * stride_d - pad_d;
      const int dend = min(dstart + kernel_d, D + pad_d);
      for (int hout = out_hstart; hout < out_hend; ++hout) {
        const int hstart = hout * stride_h - pad_h;
        const int hend = min(hstart + kernel_h, H + pad_h);
        for (int wout = out_wstart; wout < out_wend; ++wout) {
          const int wstart = wout * stride_w - pad_w; // clang-format off
          const int wend = min(wstart + kernel_w, W + pad_w);
          const AccT area = (dend - dstart) * (hend - hstart) * (wend - wstart);
          val += math::utils::LDGC<AccT>(offset_dy + ((dout * out_h + hout) * out_w + wout) * C) / area;
        } // clang-format on
      }
    }
    dx[xi] = val;
  }
}

} // namespace

#define DISPATCH_POOL_KERNEL(name, T, AccT, kBlocks, kThreads, ...)  \
  if (data_format == "NCHW") {                                       \
    name##NCHW<T, AccT>                                              \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else if (data_format == "NHWC") {                                \
    name##NHWC<T, AccT>                                              \
        <<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else {                                                           \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;             \
  }

#define DEFINE_KERNEL_LAUNCHER(name, T, out_dim)                  \
  template <>                                                     \
  void name<T, CUDAContext>(                                      \
      const int N,                                                \
      const int C,                                                \
      const int H,                                                \
      const int W,                                                \
      const int out_h,                                            \
      const int out_w,                                            \
      const int kernel_h,                                         \
      const int kernel_w,                                         \
      const int stride_h,                                         \
      const int stride_w,                                         \
      const int pad_h,                                            \
      const int pad_w,                                            \
      const string& data_format,                                  \
      const T* x,                                                 \
      T* y,                                                       \
      CUDAContext* ctx) {                                         \
    const int nthreads = N * C * out_dim;                         \
    DISPATCH_POOL_KERNEL(                                         \
        _##name,                                                  \
        math::Traits<T>::scalar_type,                             \
        math::Traits<T>::accumulator_type,                        \
        CUDA_BLOCKS(nthreads),                                    \
        CUDA_THREADS,                                             \
        nthreads,                                                 \
        C,                                                        \
        H,                                                        \
        W,                                                        \
        out_h,                                                    \
        out_w,                                                    \
        kernel_h,                                                 \
        kernel_w,                                                 \
        stride_h,                                                 \
        stride_w,                                                 \
        pad_h,                                                    \
        pad_w,                                                    \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));      \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool2d, float16, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool2d, bfloat16, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool2d, float, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool2d, double, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, float16, (H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, bfloat16, (H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, float, (H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool2dGrad, double, (H * W));
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T, out_dim)                  \
  template <>                                                     \
  void name<T, CUDAContext>(                                      \
      const int N,                                                \
      const int C,                                                \
      const int D,                                                \
      const int H,                                                \
      const int W,                                                \
      const int out_d,                                            \
      const int out_h,                                            \
      const int out_w,                                            \
      const int kernel_d,                                         \
      const int kernel_h,                                         \
      const int kernel_w,                                         \
      const int stride_d,                                         \
      const int stride_h,                                         \
      const int stride_w,                                         \
      const int pad_d,                                            \
      const int pad_h,                                            \
      const int pad_w,                                            \
      const string& data_format,                                  \
      const T* x,                                                 \
      T* y,                                                       \
      CUDAContext* ctx) {                                         \
    const int nthreads = N * C * out_dim;                         \
    DISPATCH_POOL_KERNEL(                                         \
        _##name,                                                  \
        math::Traits<T>::scalar_type,                             \
        math::Traits<T>::accumulator_type,                        \
        CUDA_BLOCKS(nthreads),                                    \
        CUDA_THREADS,                                             \
        nthreads,                                                 \
        C,                                                        \
        D,                                                        \
        H,                                                        \
        W,                                                        \
        out_d,                                                    \
        out_h,                                                    \
        out_w,                                                    \
        kernel_d,                                                 \
        kernel_h,                                                 \
        kernel_w,                                                 \
        stride_d,                                                 \
        stride_h,                                                 \
        stride_w,                                                 \
        pad_d,                                                    \
        pad_h,                                                    \
        pad_w,                                                    \
        reinterpret_cast<const math::Traits<T>::scalar_type*>(x), \
        reinterpret_cast<math::Traits<T>::scalar_type*>(y));      \
  }

DEFINE_KERNEL_LAUNCHER(AvgPool3d, float16, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool3d, bfloat16, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool3d, float, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool3d, double, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, float16, (D * H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, bfloat16, (D * H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, float, (D * H * W));
DEFINE_KERNEL_LAUNCHER(AvgPool3dGrad, double, (D * H * W));
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_POOL_KERNEL

} // namespace kernels

} // namespace dragon
