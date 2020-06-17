#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
T _RoiAlignIntp(const int H, const int W, float h, float w, const T* x) {
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

  const float tl = (float)x[ti * W + li];
  const float tr = (float)x[ti * W + ri];
  const float bl = (float)x[bi * W + li];
  const float br = (float)x[bi * W + ri];

  const float v = h - ti;
  const float u = w - li;
  const float t = tl + (tr - tl) * u;
  const float b = bl + (br - bl) * u;

  return static_cast<T>(t + (b - t) * v);
}

template <typename T>
void _RoiAlign(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const T* x,
    const float* rois,
    T* y) {
  auto x_inner_dim = H * W;
  auto y_inner_dim = out_h * out_w;
  auto x_cols = C * x_inner_dim;
  auto y_cols = C * y_inner_dim;

  for (int n = 0; n < num_rois; ++n) {
    auto* roi = rois + n * 5;
    int batch_ind = (int)roi[0];
    auto* offset_y = y + n * y_cols;

    if (batch_ind < 0) {
      memset(offset_y, 0, sizeof(T) * y_cols);
      continue;
    }

    const float roi_start_w = roi[1] * spatial_scale;
    const float roi_start_h = roi[2] * spatial_scale;
    const float roi_end_w = roi[3] * spatial_scale;
    const float roi_end_h = roi[4] * spatial_scale;

    const float roi_w = std::max(roi_end_w - roi_start_w, 1.f);
    const float roi_h = std::max(roi_end_h - roi_start_h, 1.f);
    const float bin_h = roi_h / (float)out_h;
    const float bin_w = roi_w / (float)out_w;

    const int grid_h =
        sampling_ratio > 0 ? sampling_ratio : (int)std::ceil(roi_h / out_h);
    const int grid_w =
        sampling_ratio > 0 ? sampling_ratio : (int)std::ceil(roi_w / out_w);
    const T num_grids = T(grid_h * grid_w);

    int yi;
    T val;
    float hstart, wstart, h, w;
    const T* offset_x = x + batch_ind * x_cols;

    for (int c = 0; c < C; ++c) {
      yi = 0;
      for (int oh = 0; oh < out_h; ++oh) {
        hstart = roi_start_h + oh * bin_h;
        for (int ow = 0; ow < out_w; ++ow) {
          wstart = roi_start_w + ow * bin_w;
          val = T(0);
          for (int i = 0; i < grid_h; ++i) {
            h = hstart + (i + .5f) * bin_h / (float)grid_h;
            for (int j = 0; j < grid_w; ++j) {
              w = wstart + (j + .5f) * bin_w / (float)grid_w;
              val += _RoiAlignIntp(H, W, h, w, offset_x);
            } // End j
          } // End i
          offset_y[yi++] = val / num_grids;
        }
      } // End oh && ow
      offset_x += x_inner_dim;
      offset_y += y_inner_dim;
    } // End c
  } // End n
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void RoiAlign<float16, CPUContext>(
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
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T) \
  template <>                     \
  void RoiAlign<T, CPUContext>(   \
      const int C,                \
      const int H,                \
      const int W,                \
      const int out_h,            \
      const int out_w,            \
      const int num_rois,         \
      const float spatial_scale,  \
      const int sampling_ratio,   \
      const T* x,                 \
      const float* rois,          \
      T* y,                       \
      CPUContext* ctx) {          \
    _RoiAlign(                    \
        C,                        \
        H,                        \
        W,                        \
        out_h,                    \
        out_w,                    \
        num_rois,                 \
        spatial_scale,            \
        sampling_ratio,           \
        x,                        \
        rois,                     \
        y);                       \
  }

#define DEFINE_GRAD_KERNEL_LAUNCHER(T) \
  template <>                          \
  void RoiAlignGrad<T, CPUContext>(    \
      const int C,                     \
      const int H,                     \
      const int W,                     \
      const int out_h,                 \
      const int out_w,                 \
      const int num_rois,              \
      const float spatial_scale,       \
      const int sampling_ratio,        \
      const T* dy,                     \
      const float* rois,               \
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

} // namespace kernel

} // namespace dragon
