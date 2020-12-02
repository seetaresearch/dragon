#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<float>(__ldg(x + i))
#else
#define LDG(x, i) x[i]
#define LDG2(x, i) convert::To<float>(x[i])
#endif

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
    y[yi] = LDG(x, (((n * C + c) * H + h_in) * W + w_in));
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
    y[yi] = LDG(x, (((n * H + h_in) * W + w_in) * C + c));
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
    atomicAdd(&dx[((n * C + c) * H + h_in) * W + w_in], LDG2(dy, yi));
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
    atomicAdd(&dx[((n * H + h_in) * W + w_in) * C + c], LDG2(dy, yi));
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_RESIZE_KERNEL(name, T, nblocks, nthreads, ...)            \
  if (data_format == "NCHW") {                                             \
    name##NCHW<<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else if (data_format == "NHWC") {                                      \
    name##NHWC<<<nblocks, nthreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else {                                                                 \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                   \
  }

#define DEFINE_KERNEL_LAUNCHER(T, ScalarT)   \
  template <>                                \
  void ResizeNearest<T, CUDAContext>(        \
      const int N,                           \
      const int C,                           \
      const int H,                           \
      const int W,                           \
      const int out_h,                       \
      const int out_w,                       \
      const string& data_format,             \
      const T* x,                            \
      T* y,                                  \
      CUDAContext* ctx) {                    \
    auto nthreads = N * C * out_h * out_w;   \
    auto scale_h = (float)H / (float)out_h;  \
    auto scale_w = (float)W / (float)out_w;  \
    DISPATCH_RESIZE_KERNEL(                  \
        _ResizeNearest,                      \
        ScalarT,                             \
        CUDA_BLOCKS(nthreads),               \
        CUDA_THREADS,                        \
        nthreads,                            \
        C,                                   \
        H,                                   \
        W,                                   \
        out_h,                               \
        out_w,                               \
        scale_h,                             \
        scale_w,                             \
        reinterpret_cast<const ScalarT*>(x), \
        reinterpret_cast<ScalarT*>(y));      \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, ScalarT) \
  template <>                                   \
  void ResizeNearestGrad<T, CUDAContext>(       \
      const int N,                              \
      const int C,                              \
      const int H,                              \
      const int W,                              \
      const int out_h,                          \
      const int out_w,                          \
      const string& data_format,                \
      const T* dy,                              \
      float* dx,                                \
      CUDAContext* ctx) {                       \
    auto nthreads = N * C * out_h * out_w;      \
    auto scale_h = (float)H / (float)out_h;     \
    auto scale_w = (float)W / (float)out_w;     \
    math::Set(N* C* H* W, 0.f, dx, ctx);        \
    DISPATCH_RESIZE_KERNEL(                     \
        _ResizeNearestGrad,                     \
        ScalarT,                                \
        CUDA_BLOCKS(nthreads),                  \
        CUDA_THREADS,                           \
        nthreads,                               \
        C,                                      \
        H,                                      \
        W,                                      \
        out_h,                                  \
        out_w,                                  \
        scale_h,                                \
        scale_w,                                \
        reinterpret_cast<const ScalarT*>(dy),   \
        dx);                                    \
  }

DEFINE_KERNEL_LAUNCHER(int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(int, int);
DEFINE_KERNEL_LAUNCHER(int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(float16, half);
DEFINE_KERNEL_LAUNCHER(float, float);
DEFINE_KERNEL_LAUNCHER(double, double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16, half);
DEFINE_GRAD_KERNEL_LAUNCHER(float, float);
DEFINE_GRAD_KERNEL_LAUNCHER(double, double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER
#undef DISPATCH_RESIZE_KERNEL

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
