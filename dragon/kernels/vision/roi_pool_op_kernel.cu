#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

#if __CUDA_ARCH__ >= 350
#define LDG(x, i) __ldg(x + i)
#else
#define LDG(x, i) x[i]
#endif

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
  auto Greater = math::GreaterFunctor<T>();
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];

    if (batch_ind < 0) {
      y[yi] = convert::To<T>(0.f);
      mask[yi] = -1;
      continue;
    }

    const int roi_wstart = round(roi[1] * spatial_scale);
    const int roi_hstart = round(roi[2] * spatial_scale);
    const int roi_wend = round(roi[3] * spatial_scale);
    const int roi_hend = round(roi[4] * spatial_scale);

    const int roi_w = max(roi_wend - roi_wstart + 1, 1);
    const int roi_h = max(roi_hend - roi_hstart + 1, 1);
    const float bin_h = (float)roi_h / (float)out_h;
    const float bin_w = (float)roi_w / (float)out_w;

    int hstart = floor(bin_h * h_out);
    int wstart = floor(bin_w * w_out);
    int hend = ceil(bin_h * (h_out + 1));
    int wend = ceil(bin_w * (w_out + 1));

    hstart = min(max(hstart + roi_hstart, 0), H);
    hend = min(max(hend + roi_hstart, 0), H);
    wstart = min(max(wstart + roi_wstart, 0), W);
    wend = min(max(wend + roi_wstart, 0), W);
    const bool empty = (hend <= hstart) || (wend <= wstart);

    int max_idx = empty ? -1 : 0;
    const T* offset_x = x + (batch_ind * C + c) * H * W;
    T val = empty ? convert::To<T>(0.f) : offset_x[0];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = h * W + w;
        if (Greater(LDG(offset_x, xi), val)) {
          val = LDG(offset_x, xi);
          max_idx = xi;
        }
      }
    }
    y[yi] = val;
    mask[yi] = max_idx;
  }
}

template <typename T, typename AccT>
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
    AccT* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) continue;

    AccT* offset_dx = dx + (batch_ind * C + c) * H * W;
    if (LDG(mask, yi) != -1) {
      math::utils::AtomicAdd(
          offset_dx + LDG(mask, yi), convert::To<AccT>(dy[yi]));
    }
  }
}

#undef LDG

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(name, InputT, OutputT)                        \
  template <>                                                                \
  void name<InputT, CUDAContext>(                                            \
      const int C,                                                           \
      const int H,                                                           \
      const int W,                                                           \
      const int out_h,                                                       \
      const int out_w,                                                       \
      const int num_rois,                                                    \
      const float spatial_scale,                                             \
      const InputT* x,                                                       \
      const float* rois,                                                     \
      int* mask,                                                             \
      OutputT* y,                                                            \
      CUDAContext* ctx) {                                                    \
    auto nthreads = num_rois * C * out_h * out_w;                            \
    _##name<<<CUDA_BLOCKS(nthreads), CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
        nthreads,                                                            \
        C,                                                                   \
        H,                                                                   \
        W,                                                                   \
        out_h,                                                               \
        out_w,                                                               \
        spatial_scale,                                                       \
        reinterpret_cast<const math::ScalarType<InputT>::type*>(x),          \
        rois,                                                                \
        mask,                                                                \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));              \
  }

DEFINE_KERNEL_LAUNCHER(RoiPool, float16, float16);
DEFINE_KERNEL_LAUNCHER(RoiPool, float, float);
DEFINE_KERNEL_LAUNCHER(RoiPool, double, double);
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, float16, float); // RoiPoolGrad
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, float, float); // RoiPoolGrad
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, double, float); // RoiPoolGrad
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
