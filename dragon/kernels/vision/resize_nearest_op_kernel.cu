#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _ResizeNearestNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w = yi % out_w;
    const int h = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    y[yi] = __ldg(x + (((n * C + c) * H + h_in) * W + w_in));
#else
    y[yi] = x[((n * C + c) * H + h_in) * W + w_in];
#endif
  }
}

template <typename T>
__global__ void _ResizeNearestNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w = (yi / C) % out_w;
    const int h = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    y[yi] = __ldg(x + (((n * H + h_in) * W + w_in) * C + c));
#else
    y[yi] = x[((n * H + h_in) * W + w_in) * C + c];
#endif
  }
}

template <typename T>
__global__ void _ResizeNearestGradNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w = yi % out_w;
    const int h = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], (float)__ldg(dy + yi));
#else
    atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], (float)dy[yi]);
#endif
  }
}

template <>
__global__ void _ResizeNearestGradNCHW<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const half* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w = yi % out_w;
    const int h = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    atomicAdd(
        &dx[((n * C + c) * H + h_in) * W + w_in], __half2float(__ldg(dy + yi)));
#else
    atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], __half2float(dy[yi]));
#endif
  }
}

template <typename T>
__global__ void _ResizeNearestGradNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w = (yi / C) % out_w;
    const int h = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], (float)__ldg(dy + yi));
#else
    atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], (float)dy[yi]);
#endif
  }
}

template <>
__global__ void _ResizeNearestGradNHWC<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const half* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w = (yi / C) % out_w;
    const int h = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;
    const int h_in = min(int(h * scale_h), H - 1);
    const int w_in = min(int(w * scale_w), W - 1);
#if __CUDA_ARCH__ >= 350
    atomicAdd(
        &dx[((n * H + h_in) * W + w_in) * C + c], __half2float(__ldg(dy + yi)));
#else
    atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], __half2float(dy[yi]));
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ResizeNearest<float16, CUDAContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  auto nthreads = N * C * out_h * out_w;
  auto scale_h = (float)H / (float)out_h;
  auto scale_w = (float)W / (float)out_w;
  if (data_format == "NCHW") {
    _ResizeNearestNCHW<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads,
        C,
        H,
        W,
        out_h,
        out_w,
        scale_h,
        scale_w,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  } else if (data_format == "NHWC") {
    _ResizeNearestNHWC<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads,
        C,
        H,
        W,
        out_h,
        out_w,
        scale_h,
        scale_w,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  } else {
    LOG(FATAL) << "Unknown data format: " << data_format;
  }
}

template <>
void ResizeNearestGrad<float16, CUDAContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const float16* dy,
    float* dx,
    CUDAContext* ctx) {
  auto nthreads = N * C * out_h * out_w;
  auto scale_h = (float)H / (float)out_h;
  auto scale_w = (float)W / (float)out_w;
  math::Set(N * C * H * W, 0.f, dx, ctx);
  if (data_format == "NCHW") {
    _ResizeNearestGradNCHW<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads,
        C,
        H,
        W,
        out_h,
        out_w,
        scale_h,
        scale_w,
        reinterpret_cast<const half*>(dy),
        dx);
  } else if (data_format == "NHWC") {
    _ResizeNearestGradNHWC<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads,
        C,
        H,
        W,
        out_h,
        out_w,
        scale_h,
        scale_w,
        reinterpret_cast<const half*>(dy),
        dx);
  } else {
    LOG(FATAL) << "Unknown data format: " << data_format;
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                                   \
  template <>                                                       \
  void ResizeNearest<T, CUDAContext>(                               \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const string& data_format,                                    \
      const T* x,                                                   \
      T* y,                                                         \
      CUDAContext* ctx) {                                           \
    auto nthreads = N * C * out_h * out_w;                          \
    auto scale_h = (float)H / (float)out_h;                         \
    auto scale_w = (float)W / (float)out_w;                         \
    if (data_format == "NCHW") {                                    \
      _ResizeNearestNCHW<<<                                         \
          CUDA_BLOCKS(nthreads),                                    \
          CUDA_THREADS,                                             \
          0,                                                        \
          ctx->cuda_stream()>>>(                                    \
          nthreads, C, H, W, out_h, out_w, scale_h, scale_w, x, y); \
    } else if (data_format == "NHWC") {                             \
      _ResizeNearestNHWC<<<                                         \
          CUDA_BLOCKS(nthreads),                                    \
          CUDA_THREADS,                                             \
          0,                                                        \
          ctx->cuda_stream()>>>(                                    \
          nthreads, C, H, W, out_h, out_w, scale_h, scale_w, x, y); \
    } else {                                                        \
      LOG(FATAL) << "Unknown data format: " << data_format;         \
    }                                                               \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                \
  template <>                                                         \
  void ResizeNearestGrad<T, CUDAContext>(                             \
      const int N,                                                    \
      const int C,                                                    \
      const int H,                                                    \
      const int W,                                                    \
      const int out_h,                                                \
      const int out_w,                                                \
      const string& data_format,                                      \
      const T* dy,                                                    \
      float* dx,                                                      \
      CUDAContext* ctx) {                                             \
    auto nthreads = N * C * out_h * out_w;                            \
    auto scale_h = (float)H / (float)out_h;                           \
    auto scale_w = (float)W / (float)out_w;                           \
    math::Set(N* C* H* W, 0.f, dx, ctx);                              \
    if (data_format == "NCHW") {                                      \
      _ResizeNearestGradNCHW<<<                                       \
          CUDA_BLOCKS(nthreads),                                      \
          CUDA_THREADS,                                               \
          0,                                                          \
          ctx->cuda_stream()>>>(                                      \
          nthreads, C, H, W, out_h, out_w, scale_h, scale_w, dy, dx); \
    } else if (data_format == "NHWC") {                               \
      _ResizeNearestGradNHWC<<<                                       \
          CUDA_BLOCKS(nthreads),                                      \
          CUDA_THREADS,                                               \
          0,                                                          \
          ctx->cuda_stream()>>>(                                      \
          nthreads, C, H, W, out_h, out_w, scale_h, scale_w, dy, dx); \
    } else {                                                          \
      LOG(FATAL) << "Unknown data format: " << data_format;           \
    }                                                                 \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);

DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);

#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
