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
__global__ void _ResizeLinear2dNCHW(
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
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_w / C;

    const float h = TransformCoordinate(h_out, scale_h, align_corners);
    const float w = TransformCoordinate(w_out, scale_w, align_corners);

    const int ti = floorf(h);
    const int li = floorf(w);
    const int bi = h < H - 1 ? ceilf(h) : H - 1;
    const int ri = w < W - 1 ? ceilf(w) : W - 1;

    const int offset = (n * C + c) * H;
    const float tl = LDG(x, ((offset + ti) * W + li));
    const float tr = LDG(x, ((offset + ti) * W + ri));
    const float bl = LDG(x, ((offset + bi) * W + li));
    const float br = LDG(x, ((offset + bi) * W + ri));
    const float v = h - ti;
    const float u = w - li;
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = convert::To<T>(t + (b - t) * v);
  }
}

template <typename T>
__global__ void _ResizeLinear2dNHWC(
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
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    const float h = TransformCoordinate(h_out, scale_h, align_corners);
    const float w = TransformCoordinate(w_out, scale_w, align_corners);

    const int ti = floorf(h);
    const int li = floorf(w);
    const int bi = h < H - 1 ? ceilf(h) : H - 1;
    const int ri = w < W - 1 ? ceilf(w) : W - 1;

    const int offset = n * H * W * C + c;
    const float tl = LDG(x, offset + (ti * W + li) * C);
    const float tr = LDG(x, offset + (ti * W + ri) * C);
    const float bl = LDG(x, offset + (bi * W + li) * C);
    const float br = LDG(x, offset + (bi * W + ri) * C);
    const float v = h - ti;
    const float u = w - li;
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = convert::To<T>(t + (b - t) * v);
  }
}

template <typename T>
__global__ void _ResizeLinear2dGradNCHW(
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
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_w / C;

    const float h = TransformCoordinate(h_out, scale_h, align_corners);
    const float w = TransformCoordinate(w_out, scale_w, align_corners);

    const int ti = floorf(h);
    const int li = floorf(w);
    const int bi = h < H - 1 ? ceilf(h) : H - 1;
    const int ri = w < W - 1 ? ceilf(w) : W - 1;

    const int offset = (n * C + c) * H;
    const float v = h - ti;
    const float u = w - li;
    const float dt = (1.f - v) * LDG(dy, yi);
    const float db = v * LDG(dy, yi);
    math::utils::AtomicAdd(&dx[(offset + ti) * W + li], (1.f - u) * dt);
    math::utils::AtomicAdd(&dx[(offset + ti) * W + ri], u * dt);
    math::utils::AtomicAdd(&dx[(offset + bi) * W + li], (1.f - u) * db);
    math::utils::AtomicAdd(&dx[(offset + bi) * W + ri], u * db);
  }
}

template <typename T>
__global__ void _ResizeLinear2dGradNHWC(
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
    const int w_out = (yi / C) % out_w;
    const int h_out = (yi / C / out_w) % out_h;
    const int n = yi / C / out_w / out_h;

    const float h = TransformCoordinate(h_out, scale_h, align_corners);
    const float w = TransformCoordinate(w_out, scale_w, align_corners);

    const int ti = floorf(h);
    const int li = floorf(w);
    const int bi = h < H - 1 ? ceilf(h) : H - 1;
    const int ri = w < W - 1 ? ceilf(w) : W - 1;

    const int offset = n * H * W * C + c;
    const float v = h - ti;
    const float u = w - li;
    const float dt = (1.f - v) * LDG(dy, yi);
    const float db = v * LDG(dy, yi);
    math::utils::AtomicAdd(&dx[offset + (ti * W + li) * C], (1.f - u) * dt);
    math::utils::AtomicAdd(&dx[offset + (ti * W + ri) * C], u * dt);
    math::utils::AtomicAdd(&dx[offset + (bi * W + li) * C], (1.f - u) * db);
    math::utils::AtomicAdd(&dx[offset + (bi * W + ri) * C], u * db);
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

#define DEFINE_KERNEL_LAUNCHER(name, kBackward, InputT, OutputT)    \
  template <>                                                       \
  void name<InputT, CUDAContext>(                                   \
      const int N,                                                  \
      const int C,                                                  \
      const int H,                                                  \
      const int W,                                                  \
      const int out_h,                                              \
      const int out_w,                                              \
      const bool align_corners,                                     \
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
        ComputeScale(H, out_h, align_corners),                      \
        ComputeScale(W, out_w, align_corners),                      \
        align_corners,                                              \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x), \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));     \
  }

DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int8_t, int8_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, uint8_t, uint8_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int, int);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, int64_t, int64_t);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, float16, float16);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, float, float);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2d, false, double, double);
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, float16, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, float, float); // Grad
DEFINE_KERNEL_LAUNCHER(ResizeLinear2dGrad, true, double, float); // Grad
#undef DEFINE_KERNEL_LAUNCHER
#undef DISPATCH_RESIZE_KERNEL

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
