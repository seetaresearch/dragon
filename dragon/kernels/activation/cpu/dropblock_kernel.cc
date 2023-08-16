#include "dragon/kernels/activation/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

namespace kernels {

namespace {

void _MaskBlock2dNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int block_size,
    const float ratio,
    const float* r,
    uint8_t* mask) {
  const int seed_h = H - block_size + 1;
  const int seed_w = W - block_size + 1;
  const int num_seeds = N * seed_h * seed_w;
  const int HxW = H * W;
  const int CxHxW = C * HxW;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, seed_h, seed_w};
  for (int i = 0; i < num_seeds; ++i) {
    if (r[i] > ratio) {
      math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
      continue;
    }
    const int offset = index[0] * CxHxW + index[1] * W + index[2];
    for (int c = 0; c < C; ++c) {
      for (int h_b = 0; h_b < block_size; ++h_b) {
        for (int w_b = 0; w_b < block_size; ++w_b) {
          mask[offset + c * HxW + h_b * W + w_b] &= uint8_t(0);
        }
      }
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

void _MaskBlock2dNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int block_size,
    const float ratio,
    const float* r,
    uint8_t* mask) {
  const int seed_h = H - block_size + 1;
  const int seed_w = W - block_size + 1;
  const int num_seeds = N * seed_h * seed_w;
  const int WxC = W * C;
  const int HxWxC = H * WxC;
  std::array<int, 3> index = {0, 0, 0};
  std::array<int, 3> dims = {N, seed_h, seed_w};
  for (int i = 0; i < num_seeds; ++i) {
    if (r[i] > ratio) {
      math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
      continue;
    }
    const int offset = index[0] * HxWxC + index[1] * WxC + index[2] * C;
    for (int h_b = 0; h_b < block_size; ++h_b) {
      for (int w_b = 0; w_b < block_size; ++w_b) {
        for (int c = 0; c < C; ++c) {
          mask[offset + h_b * WxC + w_b * C + c] &= uint8_t(0);
        }
      }
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), index.data());
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T)                               \
  template <>                                                   \
  void DropBlock2d<T, CPUContext>(                              \
      const int N,                                              \
      const int C,                                              \
      const int H,                                              \
      const int W,                                              \
      const int block_size,                                     \
      const float ratio,                                        \
      const float scale,                                        \
      const string& data_format,                                \
      const float* r,                                           \
      const T* x,                                               \
      T* y,                                                     \
      uint8_t* mask,                                            \
      CPUContext* ctx) {                                        \
    const auto NxCxHxW = N * C * H * W;                         \
    math::Set(NxCxHxW, uint8_t(1), mask, ctx);                  \
    if (data_format == "NCHW") {                                \
      _MaskBlock2dNCHW(N, C, H, W, block_size, ratio, r, mask); \
    } else if (data_format == "NHWC") {                         \
      _MaskBlock2dNHWC(N, C, H, W, block_size, ratio, r, mask); \
    } else {                                                    \
      LOG(FATAL) << "Unknown DataFormat: " << data_format;      \
    }                                                           \
    math::ApplyMask(NxCxHxW, scale, mask, x, y, ctx);           \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
