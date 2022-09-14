#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

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
    int* index,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = yi / out_w % out_h;
    const int c = yi / out_w / out_h % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) {
      y[yi] = convert::To<T>(0.f), index[yi] = -1;
      continue;
    }

    const int roi_wstart = round(roi[1] * spatial_scale);
    const int roi_hstart = round(roi[2] * spatial_scale);
    const int roi_wend = round(roi[3] * spatial_scale);
    const int roi_hend = round(roi[4] * spatial_scale);

    const int roi_h = max(roi_hend - roi_hstart + 1, 1);
    const int roi_w = max(roi_wend - roi_wstart + 1, 1);
    const float bin_h = float(roi_h) / float(out_h);
    const float bin_w = float(roi_w) / float(out_w);

    int hstart = floor(bin_h * h_out);
    int wstart = floor(bin_w * w_out);
    int hend = ceil(bin_h * (h_out + 1));
    int wend = ceil(bin_w * (w_out + 1));

    hstart = min(max(hstart + roi_hstart, 0), H);
    hend = min(max(hend + roi_hstart, 0), H);
    wstart = min(max(wstart + roi_wstart, 0), W);
    wend = min(max(wend + roi_wstart, 0), W);
    const bool is_empty = (hend <= hstart) || (wend <= wstart);

    int maxidx = -1;
    float maxval = is_empty ? 0.f : -FLT_MAX;
    const T* offset_x = x + (batch_ind * C + c) * H * W;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int xi = h * W + w;
        const float val = convert::To<float>(__ldg(offset_x + xi));
        if (val > maxval) {
          maxval = val, maxidx = xi;
        }
      }
    }
    y[yi] = T(maxval), index[yi] = maxidx;
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
    const float /* spatial_scale */,
    const T* dy,
    const float* rois,
    const int* index,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int c = yi / out_w / out_h % C;
    const int n = yi / out_w / out_h / C;
    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) continue;
    float* offset_dx = dx + (batch_ind * C + c) * H * W;
    const int maxidx = index[yi];
    if (maxidx != -1) {
      math::utils::AtomicAdd(offset_dx + maxidx, convert::To<float>(dy[yi]));
    }
  }
}

} // namespace

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
      int* index,                                                            \
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
        index,                                                               \
        reinterpret_cast<math::ScalarType<OutputT>::type*>(y));              \
  }

DEFINE_KERNEL_LAUNCHER(RoiPool, float16, float16);
DEFINE_KERNEL_LAUNCHER(RoiPool, float, float);
DEFINE_KERNEL_LAUNCHER(RoiPool, double, double);
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, float16, float); // RoiPoolGrad
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, float, float); // RoiPoolGrad
DEFINE_KERNEL_LAUNCHER(RoiPoolGrad, double, float); // RoiPoolGrad
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
