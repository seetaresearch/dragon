#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

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
#if __CUDA_ARCH__ >= 350
    const float tl = __ldg(x + ((offset + ti) * W + li));
    const float tr = __ldg(x + ((offset + ti) * W + ri));
    const float bl = __ldg(x + ((offset + bi) * W + li));
    const float br = __ldg(x + ((offset + bi) * W + ri));
#else
    const float tl = x[(offset + ti) * W + li];
    const float tr = x[(offset + ti) * W + ri];
    const float bl = x[(offset + bi) * W + li];
    const float br = x[(offset + bi) * W + ri];
#endif
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = (T)(t + (b - t) * v);
  }
}

template <>
__global__ void _ResizeLinearNCHW<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
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
    const float tl = __half2float(__ldg(x + ((offset + ti) * W + li)));
    const float tr = __half2float(__ldg(x + ((offset + ti) * W + ri)));
    const float bl = __half2float(__ldg(x + ((offset + bi) * W + li)));
    const float br = __half2float(__ldg(x + ((offset + bi) * W + ri)));
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;

    y[yi] = __float2half(t + (b - t) * v);
#endif
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
#if __CUDA_ARCH__ >= 350
    const float tl = __ldg(x + (((offset + ti) * W + li) * C + c));
    const float tr = __ldg(x + (((offset + ti) * W + ri) * C + c));
    const float bl = __ldg(x + (((offset + bi) * W + li) * C + c));
    const float br = __ldg(x + (((offset + bi) * W + ri) * C + c));
#else
    const float tl = x[((offset + ti) * W + li) * C + c];
    const float tr = x[((offset + ti) * W + ri) * C + c];
    const float bl = x[((offset + bi) * W + li) * C + c];
    const float br = x[((offset + bi) * W + ri) * C + c];
#endif
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;
    y[yi] = (T)(t + (b - t) * v);
  }
}

template <>
__global__ void _ResizeLinearNHWC<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const half* x,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
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
    const float tl =
        __half2float(__ldg(x + (((offset + ti) * W + li) * C + c)));
    const float tr =
        __half2float(__ldg(x + (((offset + ti) * W + ri) * C + c)));
    const float bl =
        __half2float(__ldg(x + (((offset + bi) * W + li) * C + c)));
    const float br =
        __half2float(__ldg(x + (((offset + bi) * W + ri) * C + c)));
    const float t = tl + (tr - tl) * u;
    const float b = bl + (br - bl) * u;

    y[yi] = __float2half(t + (b - t) * v);
#endif
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

#if __CUDA_ARCH__ >= 350
    const float dt = (1.f - v) * ((float)__ldg(dy + yi));
    const float db = v * ((float)__ldg(dy + yi));
#else
    const float dt = (1.f - v) * ((float)dy[yi]);
    const float db = v * ((float)dy[yi]);
#endif

    const int offset = (n * C + c) * H;
    atomicAdd(&dx[(offset + ti) * W + li], (1.f - u) * dt);
    atomicAdd(&dx[(offset + ti) * W + ri], u * dt);
    atomicAdd(&dx[(offset + bi) * W + li], (1.f - u) * db);
    atomicAdd(&dx[(offset + bi) * W + ri], u * db);
  }
}

template <>
__global__ void _ResizeLinearGradNCHW<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float scale_h,
    const float scale_w,
    const bool align_corners,
    const half* dy,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
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

    const float dt = (1.f - v) * __half2float(__ldg(dy + yi));
    const float db = v * __half2float(__ldg(dy + yi));

    const int offset = (n * C + c) * H;
    atomicAdd(&dx[(offset + ti) * W + li], (1.f - u) * dt);
    atomicAdd(&dx[(offset + ti) * W + ri], u * dt);
    atomicAdd(&dx[(offset + bi) * W + li], (1.f - u) * db);
    atomicAdd(&dx[(offset + bi) * W + ri], u * db);
#endif
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

#if __CUDA_ARCH__ >= 350
    const float dt = (1.f - v) * ((float)__ldg(dy + yi));
    const float db = v * ((float)__ldg(dy + yi));
#else
    const float dt = (1.f - v) * ((float)dy[yi]);
    const float db = v * ((float)dy[yi]);
#endif

    const int offset = n * H;
    atomicAdd(&dx[((offset + ti) * W + li) * C + c], (1.f - u) * dt);
    atomicAdd(&dx[((offset + ti) * W + ri) * C + c], u * dt);
    atomicAdd(&dx[((offset + bi) * W + li) * C + c], (1.f - u) * db);
    atomicAdd(&dx[((offset + bi) * W + ri) * C + c], u * db);
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void ResizeLinear<float16, CUDAContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const float16* x,
    float16* y,
    CUDAContext* ctx) {
  auto nthreads = N * C * out_h * out_w;
  auto scale_h = ComputeScale(H, out_h, align_corners);
  auto scale_w = ComputeScale(W, out_w, align_corners);
  if (data_format == "NCHW") {
    _ResizeLinearNCHW<<<
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
        align_corners,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  } else if (data_format == "NHWC") {
    _ResizeLinearNHWC<<<
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
        align_corners,
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(y));
  } else {
    LOG(FATAL) << "Unknown data format: " << data_format;
  }
}

template <>
void ResizeLinearGrad<float16, CUDAContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const float16* dy,
    float* dx,
    CUDAContext* ctx) {
  auto nthreads = N * C * out_h * out_w;
  auto scale_h = ComputeScale(H, out_h, align_corners);
  auto scale_w = ComputeScale(W, out_w, align_corners);
  math::Set(N * C * H * W, 0.f, dx, ctx);
  if (data_format == "NCHW") {
    _ResizeLinearGradNCHW<<<
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
        align_corners,
        reinterpret_cast<const half*>(dy),
        dx);
  } else if (data_format == "NHWC") {
    _ResizeLinearGradNHWC<<<
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
        align_corners,
        reinterpret_cast<const half*>(dy),
        dx);
  } else {
    LOG(FATAL) << "Unknown data format: " << data_format;
  }
}

#define DEFINE_KERNEL_LAUNCHER(T)                           \
  template <>                                               \
  void ResizeLinear<T, CUDAContext>(                        \
      const int N,                                          \
      const int C,                                          \
      const int H,                                          \
      const int W,                                          \
      const int out_h,                                      \
      const int out_w,                                      \
      const bool align_corners,                             \
      const string& data_format,                            \
      const T* x,                                           \
      T* y,                                                 \
      CUDAContext* ctx) {                                   \
    auto nthreads = N * C * out_h * out_w;                  \
    auto scale_h = ComputeScale(H, out_h, align_corners);   \
    auto scale_w = ComputeScale(W, out_w, align_corners);   \
    if (data_format == "NCHW") {                            \
      _ResizeLinearNCHW<<<                                  \
          CUDA_BLOCKS(nthreads),                            \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          nthreads,                                         \
          C,                                                \
          H,                                                \
          W,                                                \
          out_h,                                            \
          out_w,                                            \
          scale_h,                                          \
          scale_w,                                          \
          align_corners,                                    \
          x,                                                \
          y);                                               \
    } else if (data_format == "NHWC") {                     \
      _ResizeLinearNHWC<<<                                  \
          CUDA_BLOCKS(nthreads),                            \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          nthreads,                                         \
          C,                                                \
          H,                                                \
          W,                                                \
          out_h,                                            \
          out_w,                                            \
          scale_h,                                          \
          scale_w,                                          \
          align_corners,                                    \
          x,                                                \
          y);                                               \
    } else {                                                \
      LOG(FATAL) << "Unknown data format: " << data_format; \
    }                                                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                      \
  template <>                                               \
  void ResizeLinearGrad<T, CUDAContext>(                    \
      const int N,                                          \
      const int C,                                          \
      const int H,                                          \
      const int W,                                          \
      const int out_h,                                      \
      const int out_w,                                      \
      const bool align_corners,                             \
      const string& data_format,                            \
      const T* dy,                                          \
      float* dx,                                            \
      CUDAContext* ctx) {                                   \
    auto nthreads = N * C * out_h * out_w;                  \
    auto scale_h = ComputeScale(H, out_h, align_corners);   \
    auto scale_w = ComputeScale(W, out_w, align_corners);   \
    math::Set(N* C* H* W, 0.f, dx, ctx);                    \
    if (data_format == "NCHW") {                            \
      _ResizeLinearGradNCHW<<<                              \
          CUDA_BLOCKS(nthreads),                            \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          nthreads,                                         \
          C,                                                \
          H,                                                \
          W,                                                \
          out_h,                                            \
          out_w,                                            \
          scale_h,                                          \
          scale_w,                                          \
          align_corners,                                    \
          dy,                                               \
          dx);                                              \
    } else if (data_format == "NHWC") {                     \
      _ResizeLinearGradNHWC<<<                              \
          CUDA_BLOCKS(nthreads),                            \
          CUDA_THREADS,                                     \
          0,                                                \
          ctx->cuda_stream()>>>(                            \
          nthreads,                                         \
          C,                                                \
          H,                                                \
          W,                                                \
          out_h,                                            \
          out_w,                                            \
          scale_h,                                          \
          scale_w,                                          \
          align_corners,                                    \
          dy,                                               \
          dx);                                              \
    } else {                                                \
      LOG(FATAL) << "Unknown data format: " << data_format; \
    }                                                       \
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
