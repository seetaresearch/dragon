#include "dragon/kernels/vision/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

const static string METAL_SHADERS = R"(
#include <metal_stdlib>
using namespace metal;

constant int int_arg1 [[function_constant(0)]];   // C
constant int int_arg2 [[function_constant(1)]];   // H
constant int int_arg3 [[function_constant(2)]];   // W
constant int int_arg4 [[function_constant(3)]];   // out_h
constant int int_arg5 [[function_constant(4)]];   // out_w
constant float float_arg1 [[function_constant(5)]]; // spatial_scale
constant int int_arg6 [[function_constant(6)]];     // sampling_ratio
constant bool bool_arg1 [[function_constant(7)]];   // aligned

template <typename T>
float RoiAlignIntp(
    const int H,
    const int W,
    float h,
    float w, 
    device const T* x) {
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
  const float tl = float(x[ti * W + li]);
  const float tr = float(x[ti * W + ri]);
  const float bl = float(x[bi * W + li]);
  const float br = float(x[bi * W + ri]);
  const float v = h - ti;
  const float u = w - li;
  const float t = tl + (tr - tl) * u;
  const float b = bl + (br - bl) * u;
  return t + (b - t) * v;
}

template <typename T>
kernel void RoiAlign(
    device const T* x,
    device const float* rois,
    device T* y,
    const uint yi [[thread_position_in_grid]]) {
  const int w_out = int(yi) % int_arg5;
  const int h_out = int(yi) / int_arg5 % int_arg4;
  const int c = int(yi) / int_arg5 / int_arg4 % int_arg1;
  const int n = int(yi) / int_arg5 / int_arg4 / int_arg1;

  device const float* roi = rois + n * 5;
  const int batch_ind = roi[0];
  if (batch_ind < 0) {
    y[yi] = T(0);
    return;
  }

  const float roi_offset = bool_arg1 ? 0.5f : 0.0f;
  const float roi_wstart = roi[1] * float_arg1 - roi_offset;
  const float roi_hstart = roi[2] * float_arg1 - roi_offset;
  const float roi_wend = roi[3] * float_arg1 - roi_offset;
  const float roi_hend = roi[4] * float_arg1 - roi_offset;

  const float roi_h =
      bool_arg1 ? roi_hend - roi_hstart : max(roi_hend - roi_hstart, 1.f);
  const float roi_w =
      bool_arg1 ? roi_wend - roi_wstart : max(roi_wend - roi_wstart, 1.f);
  const float bin_h = roi_h / float(int_arg4);
  const float bin_w = roi_w / float(int_arg5);

  const int grid_h = int_arg6 > 0 ? int_arg6 : int(ceil(roi_h / float(int_arg4)));
  const int grid_w = int_arg6 > 0 ? int_arg6 : int(ceil(roi_w / float(int_arg5)));

  const float hstart = roi_hstart + float(h_out) * bin_h;
  const float wstart = roi_wstart + float(w_out) * bin_w;

  device const T* offset_x = x + (batch_ind * int_arg1 + c) * int_arg2 * int_arg3;
  float val = 0.f;
  for (int i = 0; i < grid_h; i++) {
    const float h = hstart + (i + .5f) * bin_h / grid_h;
    for (int j = 0; j < grid_w; j++) {
      const float w = wstart + (j + .5f) * bin_w / grid_w;
      val += RoiAlignIntp(int_arg2, int_arg3, h, w, offset_x);
    }
  }
  y[yi] = T(val / float(max(grid_h * grid_w, 1)));
}

#define INSTANTIATE_KERNEL(T) \
  template [[host_name("RoiAlign_"#T)]] \
  kernel void RoiAlign(device const T*, device const float*, device T*, uint);

INSTANTIATE_KERNEL(half);
INSTANTIATE_KERNEL(float);
#if defined(__HAVE_NATIVE_DOUBLE__)
INSTANTIATE_KERNEL(double);
#endif // defined(__HAVE_NATIVE_DOUBLE__)
#undef INSTANTIATE_KERNEL

)";

} // namespace

#define DEFINE_KERNEL_LAUNCHER(name, InputT, OutputT)                 \
  template <>                                                         \
  void name<InputT, MPSContext>(                                      \
      const int C,                                                    \
      const int H,                                                    \
      const int W,                                                    \
      const int out_h,                                                \
      const int out_w,                                                \
      const int num_rois,                                             \
      const float spatial_scale,                                      \
      const int sampling_ratio,                                       \
      const bool aligned,                                             \
      const InputT* x,                                                \
      const float* rois,                                              \
      OutputT* y,                                                     \
      MPSContext* ctx) {                                              \
    auto nthreads = num_rois * C * out_h * out_w;                     \
    auto kernel = MPSKernel::TypedString<InputT>(#name);              \
    auto args = vector<MPSConstant>({                                 \
        MPSConstant(&C, MTLDataTypeInt, 0),                           \
        MPSConstant(&H, MTLDataTypeInt, 1),                           \
        MPSConstant(&W, MTLDataTypeInt, 2),                           \
        MPSConstant(&out_h, MTLDataTypeInt, 3),                       \
        MPSConstant(&out_w, MTLDataTypeInt, 4),                       \
        MPSConstant(&spatial_scale, MTLDataTypeFloat, 5),             \
        MPSConstant(&sampling_ratio, MTLDataTypeInt, 6),              \
        MPSConstant(&aligned, MTLDataTypeBool, 7),                    \
    });                                                               \
    auto* command_buffer = ctx->mps_stream()->command_buffer();       \
    auto* encoder = [command_buffer computeCommandEncoder];           \
    auto* pso = MPSKernel(kernel, METAL_SHADERS).GetState(ctx, args); \
    [encoder setComputePipelineState:pso];                            \
    [encoder setBuffer:id<MTLBuffer>(x) offset:0 atIndex:0];          \
    [encoder setBuffer:id<MTLBuffer>(rois) offset:0 atIndex:1];       \
    [encoder setBuffer:id<MTLBuffer>(y) offset:0 atIndex:2];          \
    MPSDispatchThreads(nthreads, encoder, pso);                       \
    [encoder endEncoding];                                            \
    [encoder release];                                                \
  }

DEFINE_KERNEL_LAUNCHER(RoiAlign, float16, float16);
DEFINE_KERNEL_LAUNCHER(RoiAlign, float, float);
DEFINE_KERNEL_LAUNCHER(RoiAlign, double, double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
