#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#else
#define LDG(x, i) x[i]
#endif

template <typename T, typename AccT, int KKH, int KKW>
__global__ void _DepthwiseConv2dNCHW(
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
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* w,
    T* y) {
  const int KH = KKH < 0 ? kernel_h : KKH;
  const int KW = KKW < 0 ? kernel_w : KKW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;
    const int x_start = (n * C + c) * H * W;

    int ih, iw, xi, wi = c * KH * KW;
    AccT sum_val = AccT(0);

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          xi = x_start + ih * W + iw;
          sum_val += convert::To<AccT>(Multiplies(LDG(x, xi), LDG(w, wi)));
        }
        ++wi;
      } // End kw
    } // End kh
    y[yi] = convert::To<T>(sum_val);
  }
}

template <typename T, typename AccT, int KKH, int KKW>
__global__ void _DepthwiseConv2dNHWC(
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
    const int dilation_h,
    const int dilation_w,
    const T* x,
    const T* w,
    T* y) {
  const int KH = KKH < 0 ? kernel_h : KKH;
  const int KW = KKW < 0 ? kernel_w : KKW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int ow = (yi / C) % out_w;
    const int oh = (yi / C / out_w) % out_h;
    const int n = yi / C / out_h / out_h;

    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;
    const int x_start = n * H;

    int ih, iw, xi, wi = c * KH * KW;
    AccT sum_val = AccT(0);

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          xi = ((x_start + ih) * W + iw) * C + c;
          sum_val += convert::To<AccT>(Multiplies(LDG(x, xi), LDG(w, wi)));
        }
        ++wi;
      } // End kw
    } // End kh
    y[yi] = convert::To<T>(sum_val);
  }
}

template <typename T, typename AccT, int KKH, int KKW>
__global__ void _DepthwiseConv2dGradNCHW(
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
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* w,
    T* dx) {
  const int KH = KKH < 0 ? kernel_h : KKH;
  const int KW = KKW < 0 ? kernel_w : KKW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int iw = xi % W;
    const int ih = (xi / W) % H;
    const int c = (xi / W / H) % C;
    const int n = xi / W / H / C;

    int oh, ow, yi, wi = c * KH * KW;
    const int y_start = (n * C + c) * out_h * out_w;
    AccT sum_val = AccT(0);

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        oh = ih + pad_h - kh * dilation_h;
        ow = iw + pad_w - kw * dilation_w;
        if ((oh % stride_h == 0) && (ow % stride_w == 0)) {
          oh = oh / stride_h;
          ow = ow / stride_w;
          if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
            yi = y_start + oh * out_w + ow;
            sum_val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(w, wi)));
          }
        }
        ++wi;
      } // End kw
    } // End kh
    dx[xi] = convert::To<T>(sum_val);
  }
}

template <typename T, typename AccT, int KKH, int KKW>
__global__ void _DepthwiseConv2dGradNHWC(
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
    const int dilation_h,
    const int dilation_w,
    const T* dy,
    const T* w,
    T* dx) {
  const int KH = KKH < 0 ? kernel_h : KKH;
  const int KW = KKW < 0 ? kernel_w : KKW;
  const auto Multiplies = math::MultipliesFunctor<T>();
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int c = xi % C;
    const int iw = (xi / C) % W;
    const int ih = (xi / C / W) % H;
    const int n = xi / C / W / H;

    int oh, ow, yi, wi = c * KH * KW;
    const int y_start = n * out_h;
    AccT sum_val = AccT(0);

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        oh = ih + pad_h - kh * dilation_h;
        ow = iw + pad_w - kw * dilation_w;
        if ((oh % stride_h == 0) && (ow % stride_w == 0)) {
          oh = oh / stride_h;
          ow = ow / stride_w;
          if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
            yi = ((y_start + oh) * out_w + ow) * C + c;
            sum_val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(w, wi)));
          }
        }
        ++wi;
      } // End kw
    } // End kh
    dx[xi] = sum_val;
  }
}

template <typename T, typename AccT>
__global__ void _DepthwiseConv2dWGradNCHW(
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
    const T* dy,
    const T* x,
    T* dw) {
  const auto Multiplies = math::MultipliesFunctor<T>();
  const int block_idx = blockIdx.x;
  const int kw = block_idx % kernel_w;
  const int kh = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;

  const int ohw = out_h * out_w;
  int ih, iw, xi, yi;
  AccT sum_val = AccT(0);

  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < ohw; j += 32) {
      ih = (j / out_w) * stride_h - pad_h + kh * dilation_h;
      iw = (j % out_w) * stride_w - pad_w + kw * dilation_w;
      if (ih >= 0 && iw >= 0 && ih < H && iw < W) {
        xi = ((i * C + c) * H + ih) * W + iw;
        yi = (i * C + c) * out_h * out_w + j;
        sum_val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(x, xi)));
      }
    }
  }

  typedef cub::BlockReduce<AccT, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  sum_val = Reduce(storage).Sum(sum_val);
  if (threadIdx.x == 0) {
    dw[block_idx] = convert::To<T>(sum_val);
  }
}

template <typename T, typename AccT>
__global__ void _DepthwiseConv2dWGradNHWC(
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
    const T* dy,
    const T* x,
    T* dw) {
  const auto Multiplies = math::MultipliesFunctor<T>();
  const int block_idx = blockIdx.x;
  const int kw = block_idx % kernel_w;
  const int kh = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;

  const int ohw = out_h * out_w;
  int ih, iw, xi, yi;
  AccT sum_val = AccT(0);

  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < ohw; j += 32) {
      ih = (j / out_w) * stride_h - pad_h + kh * dilation_h;
      iw = (j % out_w) * stride_w - pad_w + kw * dilation_w;
      if (ih >= 0 && iw >= 0 && ih < H && iw < W) {
        xi = ((i * H + ih) * W + iw) * C + c;
        yi = (i * ohw + j) * C + c;
        sum_val += convert::To<AccT>(Multiplies(LDG(dy, yi), LDG(x, xi)));
      }
    }
  }

  typedef cub::BlockReduce<AccT, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  sum_val = Reduce(storage).Sum(sum_val);
  if (threadIdx.x == 0) {
    dw[block_idx] = convert::To<T>(sum_val);
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_DATA_KERNEL(name, T, AccT, nblocks, nthreads, ...)    \
  if (data_format == "NCHW") {                                         \
    if (kernel_h == 3 && kernel_w == 3) {                              \
      name##NCHW<T, AccT, 3, 3>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 5 && kernel_w == 5) {                       \
      name##NCHW<T, AccT, 5, 5>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 7 && kernel_w == 7) {                       \
      name##NCHW<T, AccT, 7, 7>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else {                                                           \
      name##NCHW<T, AccT, -1, -1>                                      \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    }                                                                  \
  } else if (data_format == "NHWC") {                                  \
    if (kernel_h == 3 && kernel_w == 3) {                              \
      name##NHWC<T, AccT, 3, 3>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 5 && kernel_w == 5) {                       \
      name##NHWC<T, AccT, 5, 5>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else if (kernel_h == 7 && kernel_w == 7) {                       \
      name##NHWC<T, AccT, 7, 7>                                        \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    } else {                                                           \
      name##NHWC<T, AccT, -1, -1>                                      \
          <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
    }                                                                  \
  } else {                                                             \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;               \
  }

#define DISPATCH_WEIGHT_KERNEL(name, T, AccT, nblocks, nthreads, ...) \
  if (data_format == "NCHW") {                                        \
    name##NCHW<T, AccT>                                               \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);  \
  } else if (data_format == "NHWC") {                                 \
    name##NHWC<T, AccT>                                               \
        <<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__);  \
  } else {                                                            \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;              \
  }

#define DEFINE_KERNEL_LAUNCHER(T, ScalarT, AccT) \
  template <>                                    \
  void DepthwiseConv2d<T, CUDAContext>(          \
      const int N,                               \
      const int C,                               \
      const int H,                               \
      const int W,                               \
      const int out_h,                           \
      const int out_w,                           \
      const int kernel_h,                        \
      const int kernel_w,                        \
      const int stride_h,                        \
      const int stride_w,                        \
      const int pad_h,                           \
      const int pad_w,                           \
      const int dilation_h,                      \
      const int dilation_w,                      \
      const string& data_format,                 \
      const T* x,                                \
      const T* w,                                \
      T* y,                                      \
      CUDAContext* ctx) {                        \
    const auto nthreads = N * C * out_h * out_w; \
    DISPATCH_DATA_KERNEL(                        \
        _DepthwiseConv2d,                        \
        ScalarT,                                 \
        AccT,                                    \
        CUDA_BLOCKS(nthreads),                   \
        CUDA_THREADS,                            \
        nthreads,                                \
        C,                                       \
        H,                                       \
        W,                                       \
        out_h,                                   \
        out_w,                                   \
        kernel_h,                                \
        kernel_w,                                \
        stride_h,                                \
        stride_w,                                \
        pad_h,                                   \
        pad_w,                                   \
        dilation_h,                              \
        dilation_w,                              \
        reinterpret_cast<const ScalarT*>(x),     \
        reinterpret_cast<const ScalarT*>(w),     \
        reinterpret_cast<ScalarT*>(y));          \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, ScalarT, AccT) \
  template <>                                         \
  void DepthwiseConv2dGrad<T, CUDAContext>(           \
      const int N,                                    \
      const int C,                                    \
      const int H,                                    \
      const int W,                                    \
      const int out_h,                                \
      const int out_w,                                \
      const int kernel_h,                             \
      const int kernel_w,                             \
      const int stride_h,                             \
      const int stride_w,                             \
      const int pad_h,                                \
      const int pad_w,                                \
      const int dilation_h,                           \
      const int dilation_w,                           \
      const string& data_format,                      \
      const T* dy,                                    \
      const T* w,                                     \
      T* dx,                                          \
      CUDAContext* ctx) {                             \
    auto nthreads = N * C * H * W;                    \
    DISPATCH_DATA_KERNEL(                             \
        _DepthwiseConv2dGrad,                         \
        ScalarT,                                      \
        AccT,                                         \
        CUDA_BLOCKS(nthreads),                        \
        CUDA_THREADS,                                 \
        nthreads,                                     \
        C,                                            \
        H,                                            \
        W,                                            \
        out_h,                                        \
        out_w,                                        \
        kernel_h,                                     \
        kernel_w,                                     \
        stride_h,                                     \
        stride_w,                                     \
        pad_h,                                        \
        pad_w,                                        \
        dilation_h,                                   \
        dilation_w,                                   \
        reinterpret_cast<const ScalarT*>(dy),         \
        reinterpret_cast<const ScalarT*>(w),          \
        reinterpret_cast<ScalarT*>(dx));              \
  }                                                   \
  template <>                                         \
  void DepthwiseConv2dWGrad<T, CUDAContext>(          \
      const int N,                                    \
      const int C,                                    \
      const int H,                                    \
      const int W,                                    \
      const int out_h,                                \
      const int out_w,                                \
      const int kernel_h,                             \
      const int kernel_w,                             \
      const int stride_h,                             \
      const int stride_w,                             \
      const int pad_h,                                \
      const int pad_w,                                \
      const int dilation_h,                           \
      const int dilation_w,                           \
      const string& data_format,                      \
      const T* dy,                                    \
      const T* x,                                     \
      T* dw,                                          \
      CUDAContext* ctx) {                             \
    const auto nblocks = C * kernel_h * kernel_w;     \
    const auto nthreads = 256;                        \
    DISPATCH_WEIGHT_KERNEL(                           \
        _DepthwiseConv2dWGrad,                        \
        ScalarT,                                      \
        AccT,                                         \
        nblocks,                                      \
        nthreads,                                     \
        N,                                            \
        C,                                            \
        H,                                            \
        W,                                            \
        out_h,                                        \
        out_w,                                        \
        kernel_h,                                     \
        kernel_w,                                     \
        stride_h,                                     \
        stride_w,                                     \
        pad_h,                                        \
        pad_w,                                        \
        dilation_h,                                   \
        dilation_w,                                   \
        reinterpret_cast<const ScalarT*>(dy),         \
        reinterpret_cast<const ScalarT*>(x),          \
        reinterpret_cast<ScalarT*>(dw));              \
  }

DEFINE_KERNEL_LAUNCHER(float16, half, float);
DEFINE_KERNEL_LAUNCHER(float, float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, half, float);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float, float);
#undef DISPATCH_DATA_KERNEL
#undef DISPATCH_WEIGHT_KERNEL
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
