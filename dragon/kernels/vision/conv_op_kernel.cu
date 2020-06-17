#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _Im2Col2dNCHW(
    const int nthreads,
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
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int hi = i / out_w;
    const int ow = i % out_w;
    const int oh = hi % out_h;
    const int ic = hi / out_h;
    const int oc = ic * kernel_h * kernel_w;

    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;

    int ih, iw;
    T* y = col + ((oc * out_h + oh) * out_w + ow);
    const T* x = im + ((ic * H + ih_start) * W + iw_start);

    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
#if __CUDA_ARCH__ >= 350
        *y = (ih >= 0 && iw >= 0 && ih < H && iw < W)
            ? __ldg(x + (kh * dilation_h * W + kw * dilation_w))
            : T(0);
#else
        *y = (ih >= 0 && iw >= 0 && ih < H && iw < W)
            ? x[kh * dilation_h * W + kw * dilation_w]
            : T(0);
#endif
        y += (out_h * out_w);
      }
    }
  }
}

template <typename T>
__global__ void _Im2Col2dNHWC(
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
    const T* im,
    T* col) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    const int ic = i % C;
    const int ow = (i / C) % out_w;
    const int oh = i / C / out_w;

    int ih, iw, yi;
    const int ih_start = oh * stride_h - pad_h;
    const int iw_start = ow * stride_w - pad_w;
    const int yi_start = ((oh * out_w) + ow) * kernel_h;

    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        ih = ih_start + kh * dilation_h;
        iw = iw_start + kw * dilation_w;
        yi = (((yi_start + kh) * kernel_w + kw) * C + ic);
#if __CUDA_ARCH__ >= 350
        col[yi] = (ih >= 0 && iw >= 0 && ih < H && iw < W)
            ? __ldg(im + ((ih * W + iw) * C + ic))
            : T(0);
#else
        col[yi] = (ih >= 0 && iw >= 0 && ih < H && iw < W)
            ? im[(ih * W + iw) * C + ic]
            : T(0);
#endif
      }
    }
  }
}

template <typename T>
__global__ void _Col2Im2dNCHW(
    const int nthreads,
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
  const int DKH = (kernel_h - 1) * dilation_h + 1;
  const int DKW = (kernel_w - 1) * dilation_w + 1;
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int iw = xi % W + pad_w;
    const int ih = (xi / W) % H + pad_h;
    const int ic = xi / W / H;

    const int oh_start = (ih < DKH) ? 0 : (ih - DKH) / stride_h + 1;
    const int ow_start = (iw < DKW) ? 0 : (iw - DKW) / stride_w + 1;
    const int oh_end = min(ih / stride_h + 1, out_h);
    const int ow_end = min(iw / stride_w + 1, out_w);

    int kh, kw, yi;

    T sum_val = T(0);
    for (int oh = oh_start; oh < oh_end; ++oh) {
      for (int ow = ow_start; ow < ow_end; ++ow) {
        kh = (ih - oh * stride_h);
        kw = (iw - ow * stride_w);
        if (kh % dilation_h == 0 && kw % dilation_w == 0) {
          kh /= dilation_h;
          kw /= dilation_w;
          yi = (((ic * kernel_h + kh) * kernel_w + kw) * out_h + oh) * out_w +
              ow;
#if __CUDA_ARCH__ >= 350
          sum_val += __ldg(col + yi);
#else
          sum_val += col[yi];
#endif
        }
      } // End ow
    } // End oh

    im[xi] = sum_val;
  }
}

template <typename T>
__global__ void _Col2Im2dNHWC(
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
    const T* col,
    T* im) {
  const int DKH = (kernel_h - 1) * dilation_h + 1;
  const int DKW = (kernel_w - 1) * dilation_w + 1;
  CUDA_1D_KERNEL_LOOP(xi, nthreads) {
    const int ic = xi % C;
    const int iw = (xi / C) % W + pad_w;
    const int ih = (xi / C / W) + pad_h;

    const int oh_start = (ih < DKH) ? 0 : (ih - DKH) / stride_h + 1;
    const int ow_start = (iw < DKW) ? 0 : (iw - DKW) / stride_w + 1;
    const int oh_end = min(ih / stride_h + 1, out_h);
    const int ow_end = min(iw / stride_w + 1, out_w);

    int kh, kw, yi;

    T sum_val = T(0);
    for (int oh = oh_start; oh < oh_end; ++oh) {
      for (int ow = ow_start; ow < ow_end; ++ow) {
        kh = (ih - oh * stride_h);
        kw = (iw - ow * stride_w);
        if (kh % dilation_h == 0 && kw % dilation_w == 0) {
          kh /= dilation_h;
          kw /= dilation_w;
          yi = (((oh * out_w + ow) * kernel_h + kh) * kernel_w + kw) * C + ic;
#if __CUDA_ARCH__ >= 350
          sum_val += __ldg(col + yi);
#else
          sum_val += col[yi];
#endif
        }
      }
    }

    im[xi] = sum_val;
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                          \
  template <>                                              \
  void Im2Col2d<T, CUDAContext>(                           \
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
      const int dilation_h,                                \
      const int dilation_w,                                \
      const string& data_format,                           \
      const T* im,                                         \
      T* col,                                              \
      CUDAContext* ctx) {                                  \
    const int nthreads = C * out_h * out_w;                \
    if (data_format == "NCHW") {                           \
      _Im2Col2dNCHW<<<                                     \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
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
          dilation_h,                                      \
          dilation_w,                                      \
          im,                                              \
          col);                                            \
    } else if (data_format == "NHWC") {                    \
      _Im2Col2dNHWC<<<                                     \
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
          dilation_h,                                      \
          dilation_w,                                      \
          im,                                              \
          col);                                            \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }                                                        \
  template <>                                              \
  void Col2Im2d<T, CUDAContext>(                           \
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
      const int dilation_h,                                \
      const int dilation_w,                                \
      const string& data_format,                           \
      const T* col,                                        \
      T* im,                                               \
      CUDAContext* ctx) {                                  \
    const int nthreads = C * H * W;                        \
    if (data_format == "NCHW") {                           \
      _Col2Im2dNCHW<<<                                     \
          CUDA_BLOCKS(nthreads),                           \
          CUDA_THREADS,                                    \
          0,                                               \
          ctx->cuda_stream()>>>(                           \
          nthreads,                                        \
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
          dilation_h,                                      \
          dilation_w,                                      \
          col,                                             \
          im);                                             \
    } else if (data_format == "NHWC") {                    \
      _Col2Im2dNHWC<<<                                     \
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
          dilation_h,                                      \
          dilation_w,                                      \
          col,                                             \
          im);                                             \
    } else {                                               \
      LOG(FATAL) << "Unknown DataFormat: " << data_format; \
    }                                                      \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
