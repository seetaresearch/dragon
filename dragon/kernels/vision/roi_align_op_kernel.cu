#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LOAD(x, i) __ldg(x + i)
#define LOADF(x, i) __half2float(__ldg(x + i))
#else
#define LOAD(x, i) x[i]
#define LOADF(x, i) __half2float(x[i])
#endif

template <typename T>
__device__ float
_RoiAlignIntp(const int H, const int W, float h, float w, const T* x) {
  if (h < -1.f || h > H || w < -1.f || w > W) return T(0);

  if (h <= 0.f) h = 0.f;
  if (w <= 0.f) w = 0.f;

  int ti = (int)h, bi;
  int li = (int)w, ri;

  if (ti < H - 1) {
    bi = ti + 1;
  } else {
    ti = bi = H - 1;
    h = (float)ti;
  }

  if (li < W - 1) {
    ri = li + 1;
  } else {
    ri = li = W - 1;
    w = (float)li;
  }

  const float tl = LOAD(x, (ti * W + li));
  const float tr = LOAD(x, (ti * W + ri));
  const float bl = LOAD(x, (bi * W + li));
  const float br = LOAD(x, (bi * W + ri));

  const float v = h - ti;
  const float u = w - li;
  const float t = tl + (tr - tl) * u;
  const float b = bl + (br - bl) * u;

  return t + (b - t) * v;
}

template <>
__device__ float
_RoiAlignIntp<half>(const int H, const int W, float h, float w, const half* x) {
  if (h < -1.f || h > H || w < -1.f || w > W) return 0.f;

  if (h <= 0.f) h = 0.f;
  if (w <= 0.f) w = 0.f;

  int ti = (int)h, bi;
  int li = (int)w, ri;

  if (ti < H - 1) {
    bi = ti + 1;
  } else {
    ti = bi = H - 1;
    h = (float)ti;
  }

  if (li < W - 1) {
    ri = li + 1;
  } else {
    ri = li = W - 1;
    w = (float)li;
  }

  const float tl = LOADF(x, (ti * W + li));
  const float tr = LOADF(x, (ti * W + ri));
  const float bl = LOADF(x, (bi * W + li));
  const float br = LOADF(x, (bi * W + ri));

  const float v = h - ti;
  const float u = w - li;
  const float t = tl + (tr - tl) * u;
  const float b = bl + (br - bl) * u;

  return t + (b - t) * v;
}

__device__ void _RoiAlignIntpParam(
    const int H,
    const int W,
    float h,
    float w,
    int& ti,
    int& bi,
    int& li,
    int& ri,
    float& v,
    float& u) {
  if (h < -1.f || h > H || w < -1.f || w > W) {
    li = ri = ti = bi = -1;
    return;
  }

  if (h <= 0.f) h = 0.f;
  if (w <= 0) w = 0.f;

  ti = (int)h;
  li = (int)w;

  if (ti < H - 1) {
    bi = ti + 1;
  } else {
    ti = bi = H - 1;
    h = (float)ti;
  }

  if (li < W - 1) {
    ri = li + 1;
  } else {
    ri = li = W - 1;
    w = (float)li;
  }

  v = h - ti;
  u = w - li;
}

template <typename T>
__global__ void _RoiAlign(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const int sampling_ratio,
    const T* x,
    const float* rois,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];

    if (batch_ind < 0) {
      y[yi] = T(0);
      continue;
    }

    const float roi_start_w = roi[1] * spatial_scale;
    const float roi_start_h = roi[2] * spatial_scale;
    const float roi_end_w = roi[3] * spatial_scale;
    const float roi_end_h = roi[4] * spatial_scale;

    const float roi_w = max(roi_end_w - roi_start_w, 1.f);
    const float roi_h = max(roi_end_h - roi_start_h, 1.f);
    const float bin_h = roi_h / (float)out_h;
    const float bin_w = roi_w / (float)out_w;

    const int grid_h =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / out_h);
    const int grid_w =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / out_w);

    const float hstart = roi_start_h + oh * bin_h;
    const float wstart = roi_start_w + ow * bin_w;

    const T* offset_x = x + (batch_ind * C + c) * H * W;

    T val = T(0);
    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        val += _RoiAlignIntp(H, W, h, w, offset_x);
      }
    }

    y[yi] = val / T(grid_h * grid_w);
  }
}

template <>
__global__ void _RoiAlign<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const int sampling_ratio,
    const half* x,
    const float* rois,
    half* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];

    if (batch_ind < 0) {
      y[yi] = __float2half(0.f);
      continue;
    }

    const float roi_start_w = roi[1] * spatial_scale;
    const float roi_start_h = roi[2] * spatial_scale;
    const float roi_end_w = roi[3] * spatial_scale;
    const float roi_end_h = roi[4] * spatial_scale;

    const float roi_w = max(roi_end_w - roi_start_w, 1.f);
    const float roi_h = max(roi_end_h - roi_start_h, 1.f);
    const float bin_h = roi_h / (float)out_h;
    const float bin_w = roi_w / (float)out_w;

    const int grid_h =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / out_h);
    const int grid_w =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / out_w);

    const float hstart = roi_start_h + oh * bin_h;
    const float wstart = roi_start_w + ow * bin_w;

    const half* offset_x = x + (batch_ind * C + c) * H * W;

    float val = 0.f;
    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        val += _RoiAlignIntp(H, W, h, w, offset_x);
      }
    }

    y[yi] = __float2half(val / float(grid_h * grid_w));
  }
}

template <typename T>
__global__ void _RoiAlignGrad(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const int sampling_ratio,
    const T* dy,
    const float* rois,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];

    if (batch_ind < 0) continue;

    const float roi_start_w = roi[1] * spatial_scale;
    const float roi_start_h = roi[2] * spatial_scale;
    const float roi_end_w = roi[3] * spatial_scale;
    const float roi_end_h = roi[4] * spatial_scale;

    const float roi_w = max(roi_end_w - roi_start_w, 1.f);
    const float roi_h = max(roi_end_h - roi_start_h, 1.f);
    const float bin_h = roi_h / (float)out_h;
    const float bin_w = roi_w / (float)out_w;

    const int grid_h =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / out_h);
    const int grid_w =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / out_w);

    const float hstart = roi_start_h + oh * bin_h;
    const float wstart = roi_start_w + ow * bin_w;

    const float dyi = dy[yi] / T(grid_h * grid_w);
    float* offset_dx = dx + (batch_ind * C + c) * H * W;

    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        int ti, bi, li, ri;
        float v, u;
        _RoiAlignIntpParam(H, W, h, w, ti, bi, li, ri, v, u);
        if (li >= 0 && ri >= 0 && ti >= 0 && bi >= 0) {
          const float db = dyi * v;
          const float dt = dyi * (1.f - v);
          atomicAdd(offset_dx + ti * W + li, (1.f - u) * dt);
          atomicAdd(offset_dx + ti * W + ri, u * dt);
          atomicAdd(offset_dx + bi * W + li, (1.f - u) * db);
          atomicAdd(offset_dx + bi * W + ri, u * db);
        }
      } // End i
    } // End j
  }
}

template <>
__global__ void _RoiAlignGrad<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const int sampling_ratio,
    const half* dy,
    const float* rois,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int ow = yi % out_w;
    const int oh = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];

    if (batch_ind < 0) continue;

    const float roi_start_w = roi[1] * spatial_scale;
    const float roi_start_h = roi[2] * spatial_scale;
    const float roi_end_w = roi[3] * spatial_scale;
    const float roi_end_h = roi[4] * spatial_scale;

    const float roi_w = max(roi_end_w - roi_start_w, 1.f);
    const float roi_h = max(roi_end_h - roi_start_h, 1.f);
    const float bin_h = roi_h / (float)out_h;
    const float bin_w = roi_w / (float)out_w;

    const int grid_h =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / out_h);
    const int grid_w =
        sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / out_w);

    const float hstart = roi_start_h + oh * bin_h;
    const float wstart = roi_start_w + ow * bin_w;

    const float dyi = __half2float(dy[yi]) / float(grid_h * grid_w);
    float* offset_dx = dx + (batch_ind * C + c) * H * W;

    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        int ti, bi, li, ri;
        float v, u;
        _RoiAlignIntpParam(H, W, h, w, ti, bi, li, ri, v, u);
        if (li >= 0 && ri >= 0 && ti >= 0 && bi >= 0) {
          const float db = dyi * v;
          const float dt = dyi * (1.f - v);
          atomicAdd(offset_dx + ti * W + li, (1.f - u) * dt);
          atomicAdd(offset_dx + ti * W + ri, u * dt);
          atomicAdd(offset_dx + bi * W + li, (1.f - u) * db);
          atomicAdd(offset_dx + bi * W + ri, u * db);
        }
      } // End i
    } // End j
  }
}

#undef LOAD
#undef LOADF

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void RoiAlign<float16, CUDAContext>(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const float16* x,
    const float* rois,
    float16* y,
    CUDAContext* ctx) {
  auto nthreads = num_rois * C * out_h * out_w;
  _RoiAlign<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      C,
      H,
      W,
      out_h,
      out_w,
      spatial_scale,
      sampling_ratio,
      reinterpret_cast<const half*>(x),
      rois,
      reinterpret_cast<half*>(y));
}

template <>
void RoiAlignGrad<float16, CUDAContext>(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const float16* dy,
    const float* rois,
    float* dx,
    CUDAContext* ctx) {
  auto nthreads = num_rois * C * out_h * out_w;
  _RoiAlignGrad<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      C,
      H,
      W,
      out_h,
      out_w,
      spatial_scale,
      sampling_ratio,
      reinterpret_cast<const half*>(dy),
      rois,
      dx);
}

#define DEFINE_KERNEL_LAUNCHER(T)                                              \
  template <>                                                                  \
  void RoiAlign<T, CUDAContext>(                                               \
      const int C,                                                             \
      const int H,                                                             \
      const int W,                                                             \
      const int out_h,                                                         \
      const int out_w,                                                         \
      const int num_rois,                                                      \
      const float spatial_scale,                                               \
      const int sampling_ratio,                                                \
      const T* x,                                                              \
      const float* rois,                                                       \
      T* y,                                                                    \
      CUDAContext* ctx) {                                                      \
    auto nthreads = num_rois * C * out_h * out_w;                              \
    _RoiAlign<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads,                                                              \
        C,                                                                     \
        H,                                                                     \
        W,                                                                     \
        out_h,                                                                 \
        out_w,                                                                 \
        spatial_scale,                                                         \
        sampling_ratio,                                                        \
        x,                                                                     \
        rois,                                                                  \
        y);                                                                    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)            \
  template <>                                     \
  void RoiAlignGrad<T, CUDAContext>(              \
      const int C,                                \
      const int H,                                \
      const int W,                                \
      const int out_h,                            \
      const int out_w,                            \
      const int num_rois,                         \
      const float spatial_scale,                  \
      const int sampling_ratio,                   \
      const T* dy,                                \
      const float* rois,                          \
      float* dx,                                  \
      CUDAContext* ctx) {                         \
    auto nthreads = num_rois * C * out_h * out_w; \
    _RoiAlignGrad<<<                              \
        CUDA_BLOCKS(nthreads),                    \
        CUDA_THREADS,                             \
        0,                                        \
        ctx->cuda_stream()>>>(                    \
        nthreads,                                 \
        C,                                        \
        H,                                        \
        W,                                        \
        out_h,                                    \
        out_w,                                    \
        spatial_scale,                            \
        sampling_ratio,                           \
        dy,                                       \
        rois,                                     \
        dx);                                      \
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
