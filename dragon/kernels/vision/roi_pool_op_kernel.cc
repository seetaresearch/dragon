#include "dragon/utils/op_kernels.h"

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
    int* mask,
    T* y) {
  auto x_inner_dim = H * W;
  auto y_inner_dim = out_h * out_w;
  auto x_cols = C * x_inner_dim;
  auto y_cols = C * y_inner_dim;

  for (int n = 0; n < num_rois; ++n) {
    auto* roi = rois + n * 5;
    auto* offset_y = y + n * y_cols;
    auto* offset_mask = mask + n * y_cols;

    const int batch_ind = (int)roi[0];

    if (batch_ind < 0) {
      memset(offset_y, 0, sizeof(T) * y_cols);
      memset(offset_mask, -1, sizeof(int) * y_cols);
      continue;
    }

    const int roi_wstart = std::round(roi[1] * spatial_scale);
    const int roi_hstart = std::round(roi[2] * spatial_scale);
    const int roi_wend = std::round(roi[3] * spatial_scale);
    const int roi_hend = std::round(roi[4] * spatial_scale);

    const int roi_w = std::max(roi_wend - roi_wstart + 1, 1);
    const int roi_h = std::max(roi_hend - roi_hstart + 1, 1);
    const float bin_h = (float)roi_h / (float)out_h;
    const float bin_w = (float)roi_w / (float)out_w;

    T val;
    bool empty;
    int xi, yi, mask_val;
    int hstart, wstart, hend, wend;
    const T* offset_x = x + batch_ind * x_cols;

    for (int c = 0; c < C; ++c) {
      yi = 0;
      for (int h_out = 0; h_out < out_h; ++h_out) {
        hstart = (int)(bin_h * h_out);
        hstart = std::min(std::max(hstart + roi_hstart, 0), H);
        hend = (int)ceil(bin_h * (h_out + 1));
        hend = std::min(std::max(hend + roi_hstart, 0), H);
        empty = hend == hstart;
        for (int w_out = 0; w_out < out_w; ++w_out) {
          wstart = (int)(bin_w * w_out);
          wstart = std::min(std::max(wstart + roi_wstart, 0), W);
          wend = (int)ceil(bin_w * (w_out + 1));
          wend = std::min(std::max(wend + roi_wstart, 0), W);
          empty = empty || (wend == wstart);
          mask_val = empty ? -1 : 0;
          val = empty ? T(0) : offset_x[0];
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              xi = h * W + w;
              if (offset_x[xi] > offset_y[yi]) {
                mask_val = xi;
                val = offset_x[xi];
              }
            } // End w
          } // End h
          offset_y[yi] = val;
          offset_mask[yi++] = mask_val;
        }
      } // End h_out && w_out
      offset_x += x_inner_dim;
      offset_y += y_inner_dim;
      offset_mask += y_inner_dim;
    } // End c
  } // End n
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void RoiPool<float16, CPUContext>(
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
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                                          \
  template <>                                                              \
  void RoiPool<T, CPUContext>(                                             \
      const int C,                                                         \
      const int H,                                                         \
      const int W,                                                         \
      const int out_h,                                                     \
      const int out_w,                                                     \
      const int num_rois,                                                  \
      const float spatial_scale,                                           \
      const T* x,                                                          \
      const float* rois,                                                   \
      int* mask,                                                           \
      T* y,                                                                \
      CPUContext* ctx) {                                                   \
    _RoiPool(                                                              \
        C, H, W, out_h, out_w, num_rois, spatial_scale, x, rois, mask, y); \
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
      int* mask,                       \
      float* dx,                       \
      CPUContext* ctx) {               \
    NOT_IMPLEMENTED;                   \
  }

DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
DEFINE_GRAD_KERNEL_LAUNCHER(float16);
DEFINE_GRAD_KERNEL_LAUNCHER(float);
DEFINE_GRAD_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER
#undef DEFINE_GRAD_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
