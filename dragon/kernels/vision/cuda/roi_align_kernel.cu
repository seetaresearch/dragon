#include "dragon/kernels/vision/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
__device__ float
_RoiAlignIntp(const int H, const int W, float h, float w, const T* x) {
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
  const float tl = math::utils::LDGC<float>(x + (ti * W + li));
  const float tr = math::utils::LDGC<float>(x + (ti * W + ri));
  const float bl = math::utils::LDGC<float>(x + (bi * W + li));
  const float br = math::utils::LDGC<float>(x + (bi * W + ri));
  const float v = h - ti;
  const float u = w - li;
  const float t = tl + (tr - tl) * u;
  const float b = bl + (br - bl) * u;
  return t + (b - t) * v;
}

template <typename T>
__device__ void _RoiAlignIntpParam(
    const int H,
    const int W,
    float h,
    float w,
    int& ti,
    int& bi,
    int& li,
    int& ri,
    T& v,
    T& u) {
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
    const bool aligned,
    const T* x,
    const float* rois,
    T* y) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) {
      y[yi] = T(0.f);
      continue;
    }

    const float roi_offset = aligned ? 0.5f : 0.0f;
    const float roi_wstart = roi[1] * spatial_scale - roi_offset;
    const float roi_hstart = roi[2] * spatial_scale - roi_offset;
    const float roi_wend = roi[3] * spatial_scale - roi_offset;
    const float roi_hend = roi[4] * spatial_scale - roi_offset;

    // clang-format off
    const float roi_h = aligned ? roi_hend - roi_hstart : max(roi_hend - roi_hstart, 1.f);
    const float roi_w = aligned ? roi_wend - roi_wstart : max(roi_wend - roi_wstart, 1.f);
    const float bin_h = roi_h / float(out_h);
    const float bin_w = roi_w / float(out_w);
    const int grid_h = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_h / float(out_h)));
    const int grid_w = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_w / float(out_w)));
    // clang-format on

    const float hstart = roi_hstart + h_out * bin_h;
    const float wstart = roi_wstart + w_out * bin_w;
    const T* offset_x = x + (batch_ind * C + c) * H * W;
    float val = 0.f;
    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        val += _RoiAlignIntp(H, W, h, w, offset_x);
      }
    }
    y[yi] = val / float(max(grid_h * grid_w, 1));
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
    const bool aligned,
    const T* dy,
    const float* rois,
    float* dx) {
  CUDA_1D_KERNEL_LOOP(yi, nthreads) {
    const int w_out = yi % out_w;
    const int h_out = (yi / out_w) % out_h;
    const int c = (yi / out_w / out_h) % C;
    const int n = yi / out_w / out_h / C;

    const float* roi = rois + n * 5;
    const int batch_ind = roi[0];
    if (batch_ind < 0) continue;

    const float roi_offset = aligned ? 0.5f : 0.0f;
    const float roi_wstart = roi[1] * spatial_scale - roi_offset;
    const float roi_hstart = roi[2] * spatial_scale - roi_offset;
    const float roi_wend = roi[3] * spatial_scale - roi_offset;
    const float roi_hend = roi[4] * spatial_scale - roi_offset;

    // clang-format off
    const float roi_h = aligned ? roi_hend - roi_hstart : max(roi_hend - roi_hstart, 1.f);
    const float roi_w = aligned ? roi_wend - roi_wstart : max(roi_wend - roi_wstart, 1.f);
    const float bin_w = roi_w / float(out_w);
    const float bin_h = roi_h / float(out_h);
    const int grid_h = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_h / float(out_h)));
    const int grid_w = sampling_ratio > 0 ? sampling_ratio : int(ceil(roi_w / float(out_w)));
    // clang-format on

    const float hstart = roi_hstart + h_out * bin_h;
    const float wstart = roi_wstart + w_out * bin_w;
    float* offset_dx = dx + (batch_ind * C + c) * H * W;
    const float grad = convert::To<float>(dy[yi]) / float(grid_h * grid_w);
    for (int i = 0; i < grid_h; i++) {
      const float h = hstart + (i + .5f) * bin_h / grid_h;
      for (int j = 0; j < grid_w; j++) {
        const float w = wstart + (j + .5f) * bin_w / grid_w;
        int ti, bi, li, ri;
        float v, u;
        _RoiAlignIntpParam(H, W, h, w, ti, bi, li, ri, v, u);
        if (li >= 0 && ri >= 0 && ti >= 0 && bi >= 0) {
          const float db = grad * v;
          const float dt = grad * (1.f - v);
          math::utils::AtomicAdd(offset_dx + (ti * W + li), (1.f - u) * dt);
          math::utils::AtomicAdd(offset_dx + (ti * W + ri), u * dt);
          math::utils::AtomicAdd(offset_dx + (bi * W + li), (1.f - u) * db);
          math::utils::AtomicAdd(offset_dx + (bi * W + ri), u * db);
        }
      }
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
      const int sampling_ratio,                                              \
      const bool aligned,                                                    \
      const InputT* x,                                                       \
      const float* rois,                                                     \
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
        sampling_ratio,                                                      \
        aligned,                                                             \
        reinterpret_cast<const math::Traits<InputT>::scalar_type*>(x),       \
        rois,                                                                \
        reinterpret_cast<math::Traits<OutputT>::scalar_type*>(y));           \
  }

DEFINE_KERNEL_LAUNCHER(RoiAlign, float16, float16);
DEFINE_KERNEL_LAUNCHER(RoiAlign, bfloat16, bfloat16);
DEFINE_KERNEL_LAUNCHER(RoiAlign, float, float);
DEFINE_KERNEL_LAUNCHER(RoiAlign, double, double);
DEFINE_KERNEL_LAUNCHER(RoiAlignGrad, float16, float);
DEFINE_KERNEL_LAUNCHER(RoiAlignGrad, bfloat16, float);
DEFINE_KERNEL_LAUNCHER(RoiAlignGrad, float, float);
DEFINE_KERNEL_LAUNCHER(RoiAlignGrad, double, float);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
