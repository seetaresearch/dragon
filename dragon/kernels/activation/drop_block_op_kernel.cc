#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

void _DropBlock2dNCHW(
    const int N,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const uint32_t* r,
    int* mask) {
  const int HW = H * W;
  const int CHW = C * HW;
  const int count = N * seed_h * seed_w;
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {N, seed_h, seed_w};
  int offset;
  for (int i = 0; i < count; ++i) {
    if (r[i] > 0) {
      offset = idx[0] * CHW + idx[1] * W + idx[2];
      for (int c = 0; c < C; ++c) {
        for (int bh = 0; bh < block_size; ++bh) {
          for (int bw = 0; bw < block_size; ++bw) {
            mask[offset + c * HW + bh * W + bw] &= 0;
          }
        }
      } // Share the mask between channels
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

void _DropBlock2dNHWC(
    const int N,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const uint32_t* seed,
    int* mask) {
  const int WC = W * C;
  const int HWC = H * WC;
  const int count = N * seed_h * seed_w;
  std::array<int, 3> idx = {0, 0, 0};
  std::array<int, 3> dims = {N, seed_h, seed_w};
  int offset;
  for (int i = 0; i < count; ++i) {
    if (seed[i] > 0) {
      offset = idx[0] * HWC + idx[1] * WC + idx[2] * C;
      for (int bh = 0; bh < block_size; ++bh) {
        for (int bw = 0; bw < block_size; ++bw) {
          for (int c = 0; c < C; ++c) {
            mask[offset + bh * WC + bw * C + c] &= 0;
          }
        }
      } // Share the mask between channels
    }
    math::utils::IncreaseIndexInDims(3, dims.data(), idx.data());
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void DropBlock2d<CPUContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const float gamma,
    const string& data_format,
    uint32_t* r,
    int* mask,
    CPUContext* ctx) {
  const int count = N * seed_h * seed_w;
  math::RandomBernoulli(count, gamma, r, ctx);
  if (data_format == "NCHW") {
    _DropBlock2dNCHW(N, C, H, W, seed_h, seed_w, block_size, r, mask);
  } else if (data_format == "NHWC") {
    _DropBlock2dNHWC(N, C, H, W, seed_h, seed_w, block_size, r, mask);
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format;
  }
}

} // namespace kernel

} // namespace dragon
