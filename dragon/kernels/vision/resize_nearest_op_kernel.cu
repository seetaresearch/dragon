#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

#define LDG(x, i) __ldg(x + i)
#define LDG2(x, i) convert::To<float>(__ldg(x + i))

template <typename T>
__global__ void _ResizeNearest2dNCHW(
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
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    y[yi] = LDG(x, (((n * C + c) * H + h) * W + w));
  }
}

template <typename T>
__global__ void _ResizeNearest2dNHWC(
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
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    y[yi] = LDG(x, (((n * H + h) * W + w) * C + c));
  }
}

template <typename T>
__global__ void _ResizeNearest2dGradNCHW(
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
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    math::utils::AtomicAdd(&dx[((n * C + c) * H + h) * W + w], LDG2(dy, yi));
  }
}

template <typename T>
__global__ void _ResizeNearest2dGradNHWC(
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
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    math::utils::AtomicAdd(&dx[((n * H + h) * W + w) * C + c], LDG2(dy, yi));
  }
}

template <typename T>
__global__ void _ResizeNearest3dNCHW(
    const int nthreads,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* x,
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
    const int d = min(int(d_out * scale_d), D - 1);
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    y[yi] = LDG(x, (((n * C + c) * D + d) * H + h) * W + w);
  }
}

template <typename T>
__global__ void _ResizeNearest3dNHWC(
    const int nthreads,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* x,
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
    const int d = min(int(d_out * scale_d), D - 1);
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    y[yi] = LDG(x, (((n * D + d) * H + h) * W + w) * C + c);
  }
}

template <typename T>
__global__ void _ResizeNearest3dGradNCHW(
    const int nthreads,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / out_w;
    const int w_out = yi % out_w;
    const int h_out = tmp % out_h;
    tmp /= out_h;
    const int d_out = tmp % out_d;
    tmp /= out_d;
    const int c = tmp % C;
    const int n = tmp / C;
    const int d = min(int(d_out * scale_d), D - 1);
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    math::utils::AtomicAdd(
        &dx[(((n * C + c) * D + d) * H + h) * W + w], LDG2(dy, yi));
  }
}

template <typename T>
__global__ void _ResizeNearest3dGradNHWC(
    const int nthreads,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const float scale_d,
    const float scale_h,
    const float scale_w,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    int tmp = yi / C;
    const int c = yi % C;
    const int w_out = tmp % out_w;
    tmp /= out_w;
    const int h_out = tmp % out_h;
    tmp /= out_h;
    const int d_out = tmp % out_d;
    const int n = tmp / out_d;
    const int d = min(int(d_out * scale_d), D - 1);
    const int h = min(int(h_out * scale_h), H - 1);
    const int w = min(int(w_out * scale_w), W - 1);
    math::utils::AtomicAdd(
        &dx[(((n * D + d) * H + h) * W + w) * C + c], LDG2(dy, yi));
  }
}

#undef LDG
#undef LDG2

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DISPATCH_RESIZE_KERNEL(name, T, kBlocks, kThreads, ...)            \
  if (data_format == "NCHW") {                                             \
    name##NCHW<<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else if (data_format == "NHWC") {                                      \
    name##NHWC<<<kBlocks, kThreads, 0, ctx->cuda_stream()>>>(__VA_ARGS__); \
  } else {                                                                 \
    LOG(FATAL) << "Unknown DataFormat: " << data_format;                   \
  }

#define DEFINE_KERNEL_LAUNCHER(name, kBackward, InputT, OutputT)    \
  template <>                                                       \
  void name<InputT, CUDAContext>(                                   \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const string& data_format,                                    \
      const InputT* x,                                              \
      OutputT* y,                                                   \
      CUDAContext* ctx) {                                           \
    auto nthreads = N * C * out_h * out_w;                          \
    if (kBackward) {                                                \
      math::Set(N* C* H* W, convert::To<OutputT>(0.f), y, ctx);     \
    }                                                               \
    DISPATCH_RESIZE_KERNEL(                                         \
        _##name,                                                    \
        math::ScalarType<InputT>::type,                             \
        CUDA_BLOCKS(nthreads),                                      \
        CUDA_THREADS,                                               \
        nthreads,                                                   \
        C,                                                          \
        H,                                                          \
        W,                                                          \
        out_h,                                                      \
        out_w,                                                      \
        (float)H / (float)out_h,                                    \
        (float)W / (float)out_w,                                    \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x), \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));     \
  }

DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, float16, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, float, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeNearest2dGrad, true, double, float); // Grad
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(name, kBackward, InputT, OutputT)    \
  template <>                                                       \
  void name<InputT, CUDAContext>(                                   \
      const int N,                                                  \
      const int C,                                                  \
      const int D,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_d,                                              \
      const int out_h,                                              \
      const int out_w,                                              \
      const string& data_format,                                    \
      const InputT* x,                                              \
      OutputT* y,                                                   \
      CUDAContext* ctx) {                                           \
    auto nthreads = N * C * out_d * out_h * out_w;                  \
    if (kBackward) {                                                \
      math::Set(N* C* D* H* W, convert::To<OutputT>(0.f), y, ctx);  \
    }                                                               \
    DISPATCH_RESIZE_KERNEL(                                         \
        _##name,                                                    \
        math::ScalarType<InputT>::type,                             \
        CUDA_BLOCKS(nthreads),                                      \
        CUDA_THREADS,                                               \
        nthreads,                                                   \
        C,                                                          \
        D,                                                          \
        H,                                                          \
        W,                                                          \
        out_d,                                                      \
        out_h,                                                      \
        out_w,                                                      \
        (float)D / (float)out_d,                                    \
        (float)H / (float)out_h,                                    \
        (float)W / (float)out_w,                                    \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x), \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));     \
  }

DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, float16, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, float, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeNearest3dGrad, true, double, float); // Grad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_RESIZE_KERNEL

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
