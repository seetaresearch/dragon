#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) convert::To<AccT>(__ldg(x + i))

template <typename T, typename AccT>
__global__ void _MaxPool2dNCHW(
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
    int* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const T* offset_x = x + (n * C + c) * H * W;
    int mask_val = -1;
    AccT val = AccT(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (LDG(offset_x, h * W + w) > val) {
          mask_val = h * W + w;
          val = LDG(offset_x, mask_val);
        }
      }
    }
    y[yi] = convert::To<T>(val);
    mask[yi] = mask_val;
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool2dNHWC(
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
    int* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const int x_offset = n * H * W * C + c;
    int mask_val = -1;
    AccT val = T(-FLT_MAX);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = x_offset + (h * W + w) * C;
        if (LDG(x, xi) > val) {
          mask_val = xi;
          val = LDG(x, xi);
        }
      }
    }
    y[yi] = convert::To<T>(val);
    mask[yi] = mask_val;
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool2dGradNCHW(
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
    const int* mask,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int w = xi % W;
    const int h = (xi / W) % H;
    const int c = (xi / W / H) % C;
    const int n = xi / W / H / C;

    const int out_hstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int out_wstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int out_hend = min((h + pad_h) / stride_h + 1, out_h);
    const int out_wend = min((w + pad_w) / stride_w + 1, out_w);

    const int y_offset = (n * C + c) * out_h * out_w;
    AccT val = AccT(0);
    for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
      for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
        const int yi = y_offset + h_out * out_w + w_out;
        if (mask[yi] == (h * W + w)) {
          val += LDG(dy, yi);
        }
      }
    }
    dx[xi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool2dGradNHWC(
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
    const int* mask,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int c = xi % C;
    const int w = (xi / C) % W;
    const int h = (xi / C / W) % H;
    const int n = xi / C / W / H;

    const int out_hstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int out_wstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int out_hend = min((h + pad_h) / stride_h + 1, out_h);
    const int out_wend = min((w + pad_w) / stride_w + 1, out_w);

    const int y_offset = n * out_h * out_w * C + c;
    AccT val = AccT(0);
    for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
      for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
        const int yi = y_offset + (h_out * out_w + w_out) * C;
        if (mask[yi] == xi) {
          val += LDG(dy, yi);
        }
      }
    }
    dx[xi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool3dNCHW(
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
    int* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / out_w;
    const int w_out = yi % out_w;
    const int h_out = tmp % out_h;
    tmp /= out_h;
    const int d_out = tmp % out_d;
    tmp /= out_d;
    const int c = tmp % C;
    const int n = tmp / C;

    int dstart = d_out * stride_d - pad_d;
    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    const int dend = min(dstart + kernel_d, D);
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const T* offset_x = x + (n * C + c) * D * H * W;
    int mask_val = -1;
    AccT val = AccT(-FLT_MAX);
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          tmp = (d * H + h) * W + w;
          if (LDG(offset_x, tmp) > val) {
            mask_val = tmp;
            val = LDG(offset_x, mask_val);
          }
        }
      }
    }
    y[yi] = convert::To<T>(val);
    mask[yi] = mask_val;
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool3dNHWC(
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
    int* mask,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / C;
    const int c = yi % C;
    const int w_out = tmp % out_w;
    tmp /= out_w;
    const int h_out = tmp % out_h;
    tmp /= out_h;
    const int d_out = tmp % out_d;
    const int n = tmp / out_d;

    int dstart = d_out * stride_d - pad_d;
    int hstart = h_out * stride_h - pad_h;
    int wstart = w_out * stride_w - pad_w;
    const int dend = min(dstart + kernel_d, D);
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const int x_offset = n * D * H * W * C + c;
    int mask_val = -1;
    AccT val = AccT(-FLT_MAX);
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          tmp = x_offset + ((d * H + h) * W + w) * C;
          if (LDG(x, tmp) > val) {
            mask_val = tmp;
            val = LDG(x, tmp);
          }
        }
      }
    }
    y[yi] = convert::To<T>(val);
    mask[yi] = mask_val;
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool3dGradNCHW(
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
    const int* mask,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int tmp = xi / W;
    const int w = xi % W;
    const int h = tmp % H;
    tmp /= H;
    const int d = tmp % D;
    tmp /= D;
    const int c = tmp % C;
    const int n = tmp / C;

    const int out_dstart =
        (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    const int out_hstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int out_wstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int out_dend = min((d + pad_d) / stride_d + 1, out_d);
    const int out_hend = min((h + pad_h) / stride_h + 1, out_h);
    const int out_wend = min((w + pad_w) / stride_w + 1, out_w);

    const int y_offset = (n * C + c) * out_d * out_h * out_w;
    AccT val = AccT(0);
    for (int d_out = out_dstart; d_out < out_dend; ++d_out) {
      for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
        for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
          tmp = y_offset + (d_out * out_h + h_out) * out_w + w_out;
          if (mask[tmp] == ((d * H + h) * W + w)) {
            val += LDG(dy, tmp);
          }
        }
      }
    }
    dx[xi] = convert::To<T>(val);
  }
}

template <typename T, typename AccT>
__global__ void _MaxPool3dGradNHWC(
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
    const int* mask,
    T* dx) {
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    int tmp = xi / C;
    const int c = xi % C;
    const int w = tmp % W;
    tmp /= W;
    const int h = tmp % H;
    tmp /= H;
    const int d = tmp % D;
    const int n = tmp / D;

    const int out_dstart =
        (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
    const int out_hstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int out_wstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int out_dend = min((d + pad_d) / stride_d + 1, out_d);
    const int out_hend = min((h + pad_h) / stride_h + 1, out_h);
    const int out_wend = min((w + pad_w) / stride_w + 1, out_w);

    const int y_offset = n * out_d * out_h * out_w * C + c;
    AccT val = AccT(0);
    for (int d_out = out_dstart; d_out < out_dend; ++d_out) {
      for (int h_out = out_hstart; h_out < out_hend; ++h_out) {
        for (int w_out = out_wstart; w_out < out_wend; ++w_out) {
          tmp = y_offset + ((d_out * out_h + h_out) * out_w + w_out) * C;
          if (mask[tmp] == xi) {
            val += LDG(dy, tmp);
          }
        }
      }
    }
    dx[xi] = convert::To<T>(val);
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

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

#define DEFINE_KERNEL_LAUNCHER(name, T, out_dim)               \
  template <>                                                  \
  void name<T, CUDAContext>(                                   \
      const int N,                                             \
      const int C,                                             \
      const int H,                                             \
      const int W,                                             \
      const int out_h,                                         \
      const int out_w,                                         \
      const int kernel_h,                                      \
      const int kernel_w,                                      \
      const int stride_h,                                      \
      const int stride_w,                                      \
      const int pad_h,                                         \
      const int pad_w,                                         \
      const string& data_format,                               \
      const T* x,                                              \
      int* mask,                                               \
      T* y,                                                    \
      CUDAContext* ctx) {                                      \
    const int nthreads = N * C * out_dim;                      \
    DISPATCH_POOL_KERNEL(                                      \
        _##name,                                               \
        math::ScalarType<T>::type,                             \
        math::AccumulatorType<T>::type,                        \
        CUDA_BLOCKS(nthreads),                                 \
        CUDA_THREADS,                                          \
        nthreads,                                              \
        C,                                                     \
        H,                                                     \
        W,                                                     \
        out_h,                                                 \
        out_w,                                                 \
        kernel_h,                                              \
        kernel_w,                                              \
        stride_h,                                              \
        stride_w,                                              \
        pad_h,                                                 \
        pad_w,                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(x), \
        mask,                                                  \
        reinterpret_cast<math::ScalarType<T>::type*>(y));      \
  }

DEFINE_KERNEL_LAUNCHER(MaxPool2d, float16, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool2d, float, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool2d, double, (out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool2dGrad, float16, (H * W)); // MaxPool2dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool2dGrad, float, (H * W)); // MaxPool2dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool2dGrad, double, (H * W)); // MaxPool2dGrad
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, T, out_dim)               \
  template <>                                                  \
  void name<T, CUDAContext>(                                   \
      const int N,                                             \
      const int C,                                             \
      const int D,                                             \
      const int H,                                             \
      const int W,                                             \
      const int out_d,                                         \
      const int out_h,                                         \
      const int out_w,                                         \
      const int kernel_d,                                      \
      const int kernel_h,                                      \
      const int kernel_w,                                      \
      const int stride_d,                                      \
      const int stride_h,                                      \
      const int stride_w,                                      \
      const int pad_d,                                         \
      const int pad_h,                                         \
      const int pad_w,                                         \
      const string& data_format,                               \
      const T* x,                                              \
      int* mask,                                               \
      T* y,                                                    \
      CUDAContext* ctx) {                                      \
    const int nthreads = N * C * out_dim;                      \
    DISPATCH_POOL_KERNEL(                                      \
        _##name,                                               \
        math::ScalarType<T>::type,                             \
        math::AccumulatorType<T>::type,                        \
        CUDA_BLOCKS(nthreads),                                 \
        CUDA_THREADS,                                          \
        nthreads,                                              \
        C,                                                     \
        D,                                                     \
        H,                                                     \
        W,                                                     \
        out_d,                                                 \
        out_h,                                                 \
        out_w,                                                 \
        kernel_d,                                              \
        kernel_h,                                              \
        kernel_w,                                              \
        stride_d,                                              \
        stride_h,                                              \
        stride_w,                                              \
        pad_d,                                                 \
        pad_h,                                                 \
        pad_w,                                                 \
        reinterpret_cast<const math::ScalarType<T>::type*>(x), \
        mask,                                                  \
        reinterpret_cast<math::ScalarType<T>::type*>(y));      \
  }

DEFINE_KERNEL_LAUNCHER(MaxPool3d, float16, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool3d, float, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool3d, double, (out_d * out_h * out_w));
DEFINE_KERNEL_LAUNCHER(MaxPool3dGrad, float16, (D * H * W)); // MaxPool3dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool3dGrad, float, (D * H * W)); // MaxPool3dGrad
DEFINE_KERNEL_LAUNCHER(MaxPool3dGrad, double, (D * H * W)); // MaxPool3dGrad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_POOL_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
