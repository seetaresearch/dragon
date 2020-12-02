#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
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
    const int pw = yi % out_w;
    const int ph = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const T* offset_x = x + (n * C + c) * H * W;
    int mxi = -1;
    T val = T(-FLT_MAX);

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (offset_x[h * W + w] > val) {
          val = offset_x[mxi = h * W + w];
        }
      }
    }

    y[yi] = val;
    mask[yi] = mxi;
  }
}

template <typename T>
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
    const int pw = (yi / C) % out_w;
    const int ph = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, H);
    const int wend = min(wstart + kernel_w, W);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const int x_offset = n * H * W * C + c;
    int mxi = -1;
    T val = T(-FLT_MAX);

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = x_offset + (h * W + w) * C;
        if (x[xi] > val) {
          val = x[mxi = xi];
        }
      }
    }

    y[yi] = val;
    mask[yi] = mxi;
  }
}

template <typename T>
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

    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int phend = min((h + pad_h) / stride_h + 1, out_h);
    const int pwend = min((w + pad_w) / stride_w + 1, out_w);

    const int y_offset = (n * C + c) * out_h * out_w;
    const T* offset_dy = dy + y_offset;
    const int* offset_mask = mask + y_offset;

    T val = T(0);
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (offset_mask[ph * out_w + pw] == (h * W + w)) {
          val += offset_dy[ph * out_w + pw];
        }
      }
    }

    dx[xi] = val;
  }
}

template <typename T>
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

    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int phend = min((h + pad_h) / stride_h + 1, out_h);
    const int pwend = min((w + pad_w) / stride_w + 1, out_w);

    T val = T(0);
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        const int yi = ((n * out_h + ph) * out_w + pw) * C + c;
        if (mask[yi] == xi) val += dy[yi];
      }
    }

    dx[xi] = val;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void MaxPool2d<T, CUDAContext>(                          \
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
      int* mask,                                           \
      T* y,                                                \
      CUDAContext* ctx) {                                  \
    const int nthreads = N * C * out_h * out_w;            \
    if (data_format == "NCHW") {                           \
      _MaxPool2dNCHW<<<                                    \
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
          mask,                                            \
          y);                                              \
    } else if (data_format == "NHWC") {                    \
      _MaxPool2dNHWC<<<                                    \
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
          mask,                                            \
          y);                                              \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                     \
  template <>                                              \
  void MaxPool2dGrad<T, CUDAContext>(                      \
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
      const int* mask,                                     \
      T* dx,                                               \
      CUDAContext* ctx) {                                  \
    const int nthreads = N * C * H * W;                    \
    if (data_format == "NCHW") {                           \
      _MaxPool2dGradNCHW<<<                                \
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
          mask,                                            \
          dx);                                             \
    } else if (data_format == "NHWC") {                    \
      _MaxPool2dGradNHWC<<<                                \
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
          mask,                                            \
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
