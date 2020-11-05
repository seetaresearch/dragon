#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
__global__ void _RoiPool(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const T* x,
    const float* rois,
    int* mask,
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
      mask[yi] = -1;
      continue;
    }

    const int roi_start_w = round(roi[1] * spatial_scale);
    const int roi_start_h = round(roi[2] * spatial_scale);
    const int roi_end_w = round(roi[3] * spatial_scale);
    const int roi_end_h = round(roi[4] * spatial_scale);

    const int roi_w = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_h = max(roi_end_h - roi_start_h + 1, 1);
    const float bin_h = (float)roi_h / (float)out_h;
    const float bin_w = (float)roi_w / (float)out_w;

    int hstart = floor(bin_h * oh);
    int wstart = floor(bin_w * ow);
    int hend = ceil(bin_h * (oh + 1));
    int wend = ceil(bin_w * (ow + 1));

    hstart = min(max(hstart + roi_start_h, 0), H);
    hend = min(max(hend + roi_start_h, 0), H);
    wstart = min(max(wstart + roi_start_w, 0), W);
    wend = min(max(wend + roi_start_w, 0), W);
    const bool empty = (hend <= hstart) || (wend <= wstart);

    int max_idx = empty ? -1 : 0;
    const T* offset_x = x + (batch_ind * C + c) * H * W;
    T val = empty ? T(0) : offset_x[0];

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = h * W + w;
#if __CUDA_ARCH__ >= 350
        if (__ldg(offset_x + xi) > val) {
          val = __ldg(offset_x + xi);
          max_idx = xi;
        }
#else
        if (offset_x[xi] > val) {
          val = offset_x[xi];
          max_idx = xi;
        }
#endif
      }
    }

    y[yi] = val;
    mask[yi] = max_idx;
  }
}

template <>
__global__ void _RoiPool<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const half* x,
    const float* rois,
    int* mask,
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
      mask[yi] = -1;
      continue;
    }

    const int roi_start_w = round(roi[1] * spatial_scale);
    const int roi_start_h = round(roi[2] * spatial_scale);
    const int roi_end_w = round(roi[3] * spatial_scale);
    const int roi_end_h = round(roi[4] * spatial_scale);

    const int roi_w = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_h = max(roi_end_h - roi_start_h + 1, 1);
    const float bin_h = (float)roi_h / (float)out_h;
    const float bin_w = (float)roi_w / (float)out_w;

    int hstart = floor(bin_h * oh);
    int wstart = floor(bin_w * ow);
    int hend = ceil(bin_h * (oh + 1));
    int wend = ceil(bin_w * (ow + 1));

    hstart = min(max(hstart + roi_start_h, 0), H);
    hend = min(max(hend + roi_start_h, 0), H);
    wstart = min(max(wstart + roi_start_w, 0), W);
    wend = min(max(wend + roi_start_w, 0), W);
    const bool empty = (hend <= hstart) || (wend <= wstart);

    int max_idx = empty ? -1 : 0;
    const half* offset_x = x + ((batch_ind * C + c) * H * W);
#if __CUDA_ARCH__ >= 530
    half val = empty ? __float2half(0.f) : __ldg(offset_x);
#else
    float val = empty ? 0.f : __half2float(*offset_x);
#endif

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = h * W + w;
#if __CUDA_ARCH__ >= 530
        if (__hgt(__ldg(offset_x + xi), val)) {
          val = __ldg(offset_x + xi);
          max_idx = xi;
        }
#elif __CUDA_ARCH__ >= 350
        if (__half2float(__ldg(offset_x + xi)) > val) {
          val = __half2float(__ldg(offset_x + xi));
          max_idx = xi;
        }
#else
        if (__half2float(offset_x[xi]) > val) {
          val = __half2float(offset_x[xi]);
          max_idx = xi;
        }
#endif
      }
    }

#if __CUDA_ARCH__ >= 530
    y[yi] = val;
#else
    y[yi] = __float2half(val);
#endif
    mask[yi] = max_idx;
  }
}

template <typename T>
__global__ void _RoiPoolGrad(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const T* dy,
    const float* rois,
    const int* mask,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) continue;

    float* offset_dx = dx + (batch_ind * C + c) * H * W;
#if __CUDA_ARCH__ >= 350
    if (__ldg(mask + yi) != -1) {
      atomicAdd(offset_dx + __ldg(mask + yi), (float)dy[yi]);
    }
#else
    if (mask[yi] != -1) {
      atomicAdd(offset_dx + mask[yi], (float)dy[yi]);
    }
#endif
  }
}

template <>
__global__ void _RoiPoolGrad<half>(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const float spatial_scale,
    const half* dy,
    const float* rois,
    const int* mask,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;
    const float* roi = rois + n * 5;

    const int batch_ind = roi[0];
    if (batch_ind < 0) continue;

    float* offset_dx = dx + (batch_ind * C + c) * H * W;
#if __CUDA_ARCH__ >= 350
    if (__ldg(mask + yi) != -1) {
      atomicAdd(offset_dx + __ldg(mask + yi), __half2float(dy[yi]));
    }
#else
    if (mask[yi] != -1) {
      atomicAdd(offset_dx + mask[yi], __half2float(dy[yi]));
    }
#endif
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void RoiPool<float16, CUDAContext>(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const float16* x,
    const float* rois,
    int* mask,
    float16* y,
    CUDAContext* ctx) {
  auto nthreads = num_rois * C * out_h * out_w;
  _RoiPool<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      C,
      H,
      W,
      out_h,
      out_w,
      spatial_scale,
      reinterpret_cast<const half*>(x),
      rois,
      mask,
      reinterpret_cast<half*>(y));
}

template <>
void RoiPoolGrad<float16, CUDAContext>(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const float16* dy,
    const float* rois,
    const int* mask,
    float* dx,
    CUDAContext* ctx) {
  auto nthreads = num_rois * C * out_h * out_w;
  _RoiPoolGrad<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>(
      nthreads,
      C,
      H,
      W,
      out_h,
      out_w,
      spatial_scale,
      reinterpret_cast<const half*>(dy),
      rois,
      mask,
      dx);
} // RoiPoolGrad

#define DEFINE_KERNEL_LAUNCHER(T)                                             \
  template <>                                                                 \
  void RoiPool<T, CUDAContext>(                                               \
      const int C,                                                            \
      const int H,                                                            \
      const int W,                                                            \
      const int out_h,                                                        \
      const int out_w,                                                        \
      const int num_rois,                                                     \
      const float spatial_scale,                                              \
      const T* x,                                                             \
      const float* rois,                                                      \
      int* mask,                                                              \
      T* y,                                                                   \
      CUDAContext* ctx) {                                                     \
    auto nthreads = num_rois * C * out_h * out_w;                             \
    _RoiPool<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads, C, H, W, out_h, out_w, spatial_scale, x, rois, mask, y);    \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T)                                       \
  template <>                                                                \
  void RoiPoolGrad<T, CUDAContext>(                                          \
      const int C,                                                           \
      const int H,                                                           \
      const int W,                                                           \
      const int out_h,                                                       \
      const int out_w,                                                       \
      const int num_rois,                                                    \
      const float spatial_scale,                                             \
      const T* dy,                                                           \
      const float* rois,                                                     \
      const int* mask,                                                       \
      float* dx,                                                             \
      CUDAContext* ctx) {                                                    \
    auto nthreads = num_rois * C * out_h * out_w;                            \
    _RoiPoolGrad<<<                                                          \
        CUDA_BLOCKS(nthreads),                                               \
        CUDA_THREADS,                                                        \
        0,                                                                   \
        ctx->cuda_stream()>>>(                                               \
        nthreads, C, H, W, out_h, out_w, spatial_scale, dy, rois, mask, dx); \
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
