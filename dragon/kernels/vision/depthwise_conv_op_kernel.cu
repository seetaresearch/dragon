#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/device/common_cub.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T, int KKH, int KKW>
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;
    const int x_start = (n * C + c) * H * W;

    int ih, iw, xi, wi = c * KH * KW;
    T sum_val = T(0);
#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          xi = x_start + ih * W + iw;
#if __CUDA_ARCH__ >= 350
          sum_val += __ldg(x + xi) * __ldg(w + wi);
#else
          sum_val += x[xi] * w[wi];
#endif
        }
        ++wi;
      } // End kw
    } // End kh
    y[yi] = sum_val;
  }
}

template <typename T, int KKH, int KKW>
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
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int ow = (yi / C) % out_w;
    const int oh = (yi / C / out_w) % out_h;
    const int n = yi / C / out_h / out_h;

    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;
    const int x_start = n * H;

    int ih, iw, xi, wi = c * KH * KW;
    T sum_val = T(0);

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          xi = ((x_start + ih) * W + iw) * C + c;
#if __CUDA_ARCH__ >= 350
          sum_val += __ldg(x + xi) * __ldg(w + wi);
#else
          sum_val += x[xi] * w[wi];
#endif
        }
        ++wi;
      } // End kw
    } // End kh
    y[yi] = sum_val;
  }
}

template <typename T, int KKH, int KKW>
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int iw = xi % W;
    const int ih = (xi / W) % H;
    const int c = (xi / W / H) % C;
    const int n = xi / W / H / C;

    int oh, ow, yi, wi = c * KH * KW;
    const int y_start = (n * C + c) * out_h * out_w;
    T sum_val = T(0);

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
#if __CUDA_ARCH__ >= 350
            sum_val += __ldg(dy + yi) * __ldg(w + wi);
#else
            sum_val += dy[yi] * w[wi];
#endif
          }
        }
        ++wi;
      } // End kw
    } // End kh
    dx[xi] = sum_val;
  }
}

template <typename T, int KKH, int KKW>
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
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int c = xi % C;
    const int iw = (xi / C) % W;
    const int ih = (xi / C / W) % H;
    const int n = xi / C / W / H;

    int oh, ow, yi, wi = c * KH * KW;
    const int y_start = n * out_h;
    T sum_val = T(0);

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
#if __CUDA_ARCH__ >= 350
            sum_val += __ldg(dy + yi) * __ldg(w + wi);
#else
            sum_val += dy[yi] * w[wi];
#endif
          }
        }
        ++wi;
      } // End kw
    } // End kh
    dx[xi] = sum_val;
  }
}

template <typename T>
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
  const int block_idx = blockIdx.x;
  const int kw = block_idx % kernel_w;
  const int kh = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;

  const int ohw = out_h * out_w;
  T grad = T(0);
  int ih, iw, xi, yi;

  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < ohw; j += 32) {
      ih = (j / out_w) * stride_h - pad_h + kh * dilation_h;
      iw = (j % out_w) * stride_w - pad_w + kw * dilation_w;
      if (ih >= 0 && iw >= 0 && ih < H && iw < W) {
        xi = ((i * C + c) * H + ih) * W + iw;
        yi = (i * C + c) * out_h * out_w + j;
#if __CUDA_ARCH__ >= 350
        grad += __ldg(dy + yi) * __ldg(x + xi);
#else
        grad += dy[yi] * x[xi];
#endif
      }
    }
  }
  typedef cub::BlockReduce<T, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  grad = Reduce(storage).Sum(grad);
  if (threadIdx.x == 0) dw[block_idx] = grad;
}

template <typename T>
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
  const int block_idx = blockIdx.x;
  const int kw = block_idx % kernel_w;
  const int kh = (block_idx / kernel_w) % kernel_h;
  const int c = block_idx / kernel_w / kernel_h;

  const int n = threadIdx.x / 32;
  const int nwarps = blockDim.x / 32;
  const int lane_idx = threadIdx.x % 32;

  const int ohw = out_h * out_w;
  T grad = T(0);
  int ih, iw, xi, yi;

  for (int i = n; i < N; i += nwarps) {
    for (int j = lane_idx; j < ohw; j += 32) {
      ih = (j / out_w) * stride_h - pad_h + kh * dilation_h;
      iw = (j % out_w) * stride_w - pad_w + kw * dilation_w;
      if (ih >= 0 && iw >= 0 && ih < H && iw < W) {
        xi = ((i * H + ih) * W + iw) * C + c;
        yi = (i * ohw + j) * C + c;
#if __CUDA_ARCH__ >= 350
        grad += __ldg(dy + yi) * __ldg(x + xi);
#else
        grad += dy[yi] * x[xi];
#endif
      }
    }
  }
  typedef cub::BlockReduce<T, 256> Reduce;
  __shared__ typename Reduce::TempStorage storage;
  grad = Reduce(storage).Sum(grad);
  if (threadIdx.x == 0) dw[block_idx] = grad;
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void DepthwiseConv2d<float, CUDAContext>(
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
    CUDAContext* ctx) {
  const auto nthreads = N * C * out_h * out_w;
  if (data_format == "NCHW") {
    if (kernel_h == 3 && kernel_w == 3) {
      _DepthwiseConv2dNCHW<float, 3, 3>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else if (kernel_h == 5 && kernel_w == 5) {
      _DepthwiseConv2dNCHW<float, 5, 5>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else if (kernel_h == 7 && kernel_w == 7) {
      _DepthwiseConv2dNCHW<float, 7, 7>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else {
      _DepthwiseConv2dNCHW<float, -1, -1>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
  } else if (data_format == "NHWC") {
    if (kernel_h == 3 && kernel_w == 3) {
      _DepthwiseConv2dNHWC<float, 3, 3>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else if (kernel_h == 5 && kernel_w == 5) {
      _DepthwiseConv2dNHWC<float, 5, 5>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else if (kernel_h == 7 && kernel_w == 7) {
      _DepthwiseConv2dNHWC<float, 7, 7>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
    } else {
      _DepthwiseConv2dNHWC<float, -1, -1>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format;
  }
}

template <>
void DepthwiseConv2dGrad<float, CUDAContext>(
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
    const float* dy,
    const float* w,
    float* dx,
    CUDAContext* ctx) {
  auto nthreads = N * C * H * W;
  if (data_format == "NCHW") {
    if (kernel_h == 3 && kernel_w == 3) {
      _DepthwiseConv2dGradNCHW<float, 3, 3>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else if (kernel_h == 5 && kernel_w == 5) {
      _DepthwiseConv2dGradNCHW<float, 5, 5>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else if (kernel_h == 7 && kernel_w == 7) {
      _DepthwiseConv2dGradNCHW<float, 7, 7>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else {
      _DepthwiseConv2dGradNCHW<float, -1, -1>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    }
  } else if (data_format == "NHWC") {
    if (kernel_h == 3 && kernel_w == 3) {
      _DepthwiseConv2dGradNHWC<float, 3, 3>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else if (kernel_h == 5 && kernel_w == 5) {
      _DepthwiseConv2dGradNHWC<float, 5, 5>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else if (kernel_h == 7 && kernel_w == 7) {
      _DepthwiseConv2dGradNHWC<float, 7, 7>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    } else {
      _DepthwiseConv2dGradNHWC<float, -1, -1>
          <<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
              nthreads,
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
              dy,
              w,
              dx);
    }
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format;
  }
} // DepthwiseConv2dGrad

template <>
void DepthwiseConv2dWGrad<float, CUDAContext>(
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
    const float* dy,
    const float* x,
    float* dw,
    CUDAContext* ctx) {
  int nthreads = 256;
  auto nblocks = C * kernel_h * kernel_w;
  if (data_format == "NCHW") {
    _DepthwiseConv2dWGradNCHW<<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(
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
        dy,
        x,
        dw);
  } else if (data_format == "NHWC") {
    _DepthwiseConv2dWGradNHWC<<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(
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
        dy,
        x,
        dw);
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format;
  }
} // DepthwiseConv2dWGrad

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
