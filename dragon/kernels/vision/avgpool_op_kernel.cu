#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
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
    const int pw = yi % out_w;
    const int ph = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const T area = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, H);
    wend = min(wend, W);

    const T* offset_x = x + (n * C + c) * H * W;

    T val = T(0);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        val += offset_x[h * W + w];
      }
    }

    y[yi] = val / area;
  }
}

template <typename T>
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
    const int pw = (yi / C) % out_w;
    const int ph = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, H + pad_h);
    int wend = min(wstart + kernel_w, W + pad_w);
    const T area = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, H);
    wend = min(wend, W);

    const T* offset_x = x + n * H * W * C + c;

    T val = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        val += offset_x[(h * W + w) * C];
      }
    }

    y[yi] = val / area;
  }
}

template <typename T>
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
    const int h = (xi / W) % H + pad_h;
    const int c = (xi / W / H) % C;
    const int n = xi / W / H / C;

    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int phend = min(h / stride_h + 1, out_h);
    const int pwend = min(w / stride_w + 1, out_w);

    const T* offset_dy = dy + (n * C + c) * out_h * out_w;

    T val = T(0);
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        const int hstart = ph * stride_h - pad_h;
        const int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, H + pad_h);
        const int wend = min(wstart + kernel_w, W + pad_w);
        const T area = (hend - hstart) * (wend - wstart);
        val += offset_dy[ph * out_w + pw] / area;
      }
    }

    dx[xi] = val;
  }
}

template <typename T>
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
    const int w = (xi / C) % W + pad_w;
    const int h = (xi / C / W) % H + pad_h;
    const int n = xi / C / W / H;

    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int phend = min(h / stride_h + 1, out_h);
    const int pwend = min(w / stride_w + 1, out_w);

    const T* offset_dy = dy + n * out_h * out_w * C + c;

    T val = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        const int hstart = ph * stride_h - pad_h;
        const int wstart = pw * stride_w - pad_w;
        const int hend = min(hstart + kernel_h, H + pad_h);
        const int wend = min(wstart + kernel_w, W + pad_w);
        const T area = (hend - hstart) * (wend - wstart);
        val += offset_dy[(ph * out_w + pw) * C] / area;
      }
    }

    dx[xi] = val;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void AvgPool2d<T, CUDAContext>(                          \
      const int N,                                         \
      const int C,                                         \
      const int H,                                         \
      const int W,                                         \
      const int out_h,                                     \
      const int out_w,                                     \
      const int kernel_h,                                  \
      const int kernel_w,                                  \
      const int stride_h,                                  \
      const int stride_w,                                  \
      const int pad_h,                                     \
      const int pad_w,                                     \
      const string& data_format,                           \
      const T* x,                                          \
      T* y,                                                \
      CUDAContext* ctx) {                                  \
    const int nthreads = N * C * out_h * out_w;            \
    if (data_format == "NCHW") {                           \
      _AvgPool2dNCHW<<<                                    \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          x,                                               \
          y);                                              \
    } else if (data_format == "NHWC") {                    \
      _AvgPool2dNHWC<<<                                    \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          x,                                               \
          y);                                              \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                     \
  template <>                                              \
  void AvgPool2dGrad<T, CUDAContext>(                      \
      const int N,                                         \
      const int C,                                         \
      const int H,                                         \
      const int W,                                         \
      const int out_h,                                     \
      const int out_w,                                     \
      const int kernel_h,                                  \
      const int kernel_w,                                  \
      const int stride_h,                                  \
      const int stride_w,                                  \
      const int pad_h,                                     \
      const int pad_w,                                     \
      const string& data_format,                           \
      const T* dy,                                         \
      T* dx,                                               \
      CUDAContext* ctx) {                                  \
    const int nthreads = N * C * H * W;                    \
    if (data_format == "NCHW") {                           \
      _AvgPool2dGradNCHW<<<                                \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          dy,                                              \
          dx);                                             \
    } else if (data_format == "NHWC") {                    \
      _AvgPool2dGradNHWC<<<                                \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
          C,                                               \
          H,                                               \
          W,                                               \
          out_h,                                           \
          out_w,                                           \
          kernel_h,                                        \
          kernel_w,                                        \
          stride_h,                                        \
          stride_w,                                        \
          pad_h,                                           \
          pad_w,                                           \
          dy,                                              \
          dx);                                             \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
