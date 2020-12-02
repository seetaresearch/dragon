#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) convert::To<float>(__ldg(x + i))
#else
#define LDG(x, i) convert::To<float>(x[i])
#endif

template <typename T>
float ComputeScale(T in_size, T out_size, bool align_corners) {
  if (align_corners) {
    return (float)(in_size - T(1)) / (float)(out_size - T(1));
  } else {
    return (float)in_size / (float)out_size;
  }
}

template <typename T>
__device__ float TransformCoordinate(
    const T coord_resized,
    const float scale,
    const bool align_corners) {
  if (align_corners) {
    return coord_resized * scale;
  } else {
    float coord_original = (coord_resized + 0.5f) * scale - 0.5f;
    return coord_original < 0.f ? 0.f : coord_original;
  }
}

template <typename T>
__global__ void _ResizeLinearNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w = yi % out_w;
    const int h = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_w / C;

    const float h_in = TransformCoordinate(h, scale_h, align_corners);
    const int ti = floorf(h_in);
    const int bi = h_in < H - 1 ? ceilf(h_in) : H - 1;
    const float v = h_in - ti;

    const float w_in = TransformCoordinate(w, scale_w, align_corners);
    const int li = floorf(w_in);
    const int ri = (w_in < W - 1) ? ceilf(w_in) : W - 1;
    const float u = w_in - li;

    const int offset = (n * C + c) * H;
    const float tl = LDG(x, ((offset + ti) * W + li));
    const float tr = LDG(x, ((offset + ti) * W + ri));
    const float bl = LDG(x, ((offset + bi) * W + li));
    const float br = LDG(x, ((offset + bi) * W + ri));
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = convert::To<T>(t + (b - t) * v);
  }
}

template <typename T>
__global__ void _ResizeLinearNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* x,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w = (yi / C) % out_w;
    const int h = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    const float h_in = TransformCoordinate(h, scale_h, align_corners);
    const int ti = floorf(h_in);
    const int bi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
    const float v = h_in - ti;

    const float w_in = TransformCoordinate(w, scale_w, align_corners);
    const int li = floorf(w_in);
    const int ri = (w_in < W - 1) ? ceilf(w_in) : W - 1;
    const float u = w_in - li;

    const int offset = n * H;
    const float tl = LDG(x, (((offset + ti) * W + li) * C + c));
    const float tr = LDG(x, (((offset + ti) * W + ri) * C + c));
    const float bl = LDG(x, (((offset + bi) * W + li) * C + c));
    const float br = LDG(x, (((offset + bi) * W + ri) * C + c));
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = convert::To<T>(t + (b - t) * v);
  }
}

template <typename T>
__global__ void _ResizeLinearGradNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w = yi % out_w;
    const int h = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_w / C;

    const float h_in = TransformCoordinate(h, scale_h, align_corners);
    const int ti = floorf(h_in);
    const int bi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
    const float v = h_in - ti;

    const float w_in = TransformCoordinate(w, scale_w, align_corners);
    const int li = floorf(w_in);
    const int ri = (w_in < W - 1) ? ceilf(w_in) : W - 1;
    const float u = w_in - li;

    const float dt = (1.f - v) * LDG(dy, yi);
    const float db = v * LDG(dy, yi);

    const int offset = (n * C + c) * H;
    atomicAdd(&dx[(offset + ti) * W + li], (1.f - u) * dt);
    atomicAdd(&dx[(offset + ti) * W + ri], u * dt);
    atomicAdd(&dx[(offset + bi) * W + li], (1.f - u) * db);
    atomicAdd(&dx[(offset + bi) * W + ri], u * db);
  }
}

template <typename T>
__global__ void _ResizeLinearGradNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const T* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi % C;
    const int w = (yi / C) % out_w;
    const int h = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    const float h_in = TransformCoordinate(h, scale_h, align_corners);
    const int ti = floorf(h_in);
    const int bi = (h_in < H - 1) ? ceilf(h_in) : H - 1;
    const float v = h_in - ti;

    const float w_in = TransformCoordinate(w, scale_w, align_corners);
    const int li = floorf(w_in);
    const int ri = (w_in < W - 1) ? ceilf(w_in) : W - 1;
    const float u = w_in - li;

    const float dt = (1.f - v) * LDG(dy, yi);
    const float db = v * LDG(dy, yi);

    const int offset = n * H;
    atomicAdd(&dx[((offset + ti) * W + li) * C + c], (1.f - u) * dt);
    atomicAdd(&dx[((offset + ti) * W + ri) * C + c], u * dt);
    atomicAdd(&dx[((offset + bi) * W + li) * C + c], (1.f - u) * db);
    atomicAdd(&dx[((offset + bi) * W + ri) * C + c], u * db);
  }
}

#undef LDG

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

#define DEFINE_KERNEL_LAUNCHER(T, ScalarT)                \
  template <>                                             \
  void ResizeLinear<T, CUDAContext>(                      \
      const int N,                                        \
      const int C,                                        \
      const int H,                                        \
      const int W,                                        \
      const int out_h,                                    \
      const int out_w,                                    \
      const bool align_corners,                           \
      const string& data_format,                          \
      const T* x,                                         \
      T* y,                                               \
      CUDAContext* ctx) {                                 \
    auto nthreads = N * C * out_h * out_w;                \
    auto scale_h = ComputeScale(H, out_h, align_corners); \
    auto scale_w = ComputeScale(W, out_w, align_corners); \
    DISPATCH_RESIZE_KERNEL(                               \
        _ResizeLinear,                                    \
        ScalarT,                                          \
        CUDA_BLOCKS(nthreads),                            \
        CUDA_THREADS,                                     \
        nthreads,                                         \
        C,                                                \
        H,                                                \
        W,                                                \
        out_h,                                            \
        out_w,                                            \
        scale_h,                                          \
        scale_w,                                          \
        align_corners,                                    \
        reinterpret_cast<const ScalarT*>(x),              \
        reinterpret_cast<ScalarT*>(y));                   \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T, ScalarT)           \
  template <>                                             \
  void ResizeLinearGrad<T, CUDAContext>(                  \
      const int N,                                        \
      const int C,                                        \
      const int H,                                        \
      const int W,                                        \
      const int out_h,                                    \
      const int out_w,                                    \
      const bool align_corners,                           \
      const string& data_format,                          \
      const T* dy,                                        \
      float* dx,                                          \
      CUDAContext* ctx) {                                 \
    auto nthreads = N * C * out_h * out_w;                \
    auto scale_h = ComputeScale(H, out_h, align_corners); \
    auto scale_w = ComputeScale(W, out_w, align_corners); \
    math::Set(N* C* H* W, 0.f, dx, ctx);                  \
    DISPATCH_RESIZE_KERNEL(                               \
        _ResizeLinearGrad,                                \
        ScalarT,                                          \
        CUDA_BLOCKS(nthreads),                            \
        CUDA_THREADS,                                     \
        nthreads,                                         \
        C,                                                \
        H,                                                \
        W,                                                \
        out_h,                                            \
        out_w,                                            \
        scale_h,                                          \
        scale_w,                                          \
        align_corners,                                    \
        reinterpret_cast<const ScalarT*>(dy),             \
        dx);                                              \
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
