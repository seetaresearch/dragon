#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/conversions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
void _RoiPool(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const T* x,
    const float* rois,
    int* index,
    T* y) {
  const auto HxW = H * W;
  const auto HoxWo = out_h * out_w;
  const auto CxHxW = C * HxW;
  const auto CxHoxWo = C * HoxWo;

  for (int n = 0; n < num_rois; ++n) {
    auto* roi = rois + n * 5;
    auto* offset_y = y + n * CxHoxWo;
    auto* offset_index = index + n * CxHoxWo;

    const int batch_ind = (int)roi[0];
    if (batch_ind < 0) {
      memset(offset_y, 0, sizeof(T) * CxHoxWo);
      memset(offset_index, -1, sizeof(int) * CxHoxWo);
      continue;
    }

    const int roi_wstart = std::round(roi[1] * spatial_scale);
    const int roi_hstart = std::round(roi[2] * spatial_scale);
    const int roi_wend = std::round(roi[3] * spatial_scale);
    const int roi_hend = std::round(roi[4] * spatial_scale);

    const int roi_w = std::max(roi_wend - roi_wstart + 1, 1);
    const int roi_h = std::max(roi_hend - roi_hstart + 1, 1);
    const float bin_h = float(roi_h) / float(out_h);
    const float bin_w = float(roi_w) / float(out_w);

    const T* offset_x = x + batch_ind * CxHxW;
    for (int c = 0; c < C; ++c) {
      int yi = 0;
      for (int h_out = 0; h_out < out_h; ++h_out) {
        int hstart = int(bin_h * h_out);
        int hend = int(ceil(bin_h * (h_out + 1)));
        hstart = std::min(std::max(hstart + roi_hstart, 0), H);
        hend = std::min(std::max(hend + roi_hstart, 0), H);
        bool is_empty = hend == hstart;
        for (int w_out = 0; w_out < out_w; ++w_out) {
          int wstart = int(bin_w * w_out);
          int wend = int(ceil(bin_w * (w_out + 1)));
          wstart = std::min(std::max(wstart + roi_wstart, 0), W);
          wend = std::min(std::max(wend + roi_wstart, 0), W);
          is_empty = is_empty || (wend == wstart);
          int maxidx = -1;
          float maxval = is_empty ? 0.f : -FLT_MAX;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int xi = h * W + w;
              const float val = convert::To<float>(offset_x[xi]);
              if (val > maxval) {
                maxval = val, maxidx = xi;
              }
            } // End w
          } // End h
          offset_y[yi] = convert::To<T>(maxval);
          offset_index[yi++] = maxidx;
        }
      } // End h_out && w_out
      offset_x += HxW;
      offset_y += HoxWo;
      offset_index += HoxWo;
    } // End c
  } // End n
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void RoiPool<T, CPUContext>(                                              \
      const int C,                                                          \
      const int H,                                                          \
      const int W,                                                          \
      const int out_h,                                                      \
      const int out_w,                                                      \
      const int num_rois,                                                   \
      const float spatial_scale,                                            \
      const T* x,                                                           \
      const float* rois,                                                    \
      int* index,                                                           \
      T* y,                                                                 \
      CPUContext* ctx) {                                                    \
    _RoiPool(                                                               \
        C, H, W, out_h, out_w, num_rois, spatial_scale, x, rois, index, y); \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T) \
  template <>                          \
  void RoiPoolGrad<T, CPUContext>(     \
      const int C,                     \
      const int H,                     \
      const int W,                     \
      const int out_h,                 \
      const int out_w,                 \
      const int num_rois,              \
      const float spatial_scale,       \
      const T* dy,                     \
      const float* rois,               \
      int* index,                      \
      float* dx,                       \
      CPUContext* ctx) {               \
    NOT_IMPLEMENTED;                   \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
