#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

__global__ void _MaskBlock2dNCHW(
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int num_seeds,
    const int block_size,
    const uint32_t thresh,
    const uint32_t* r,
    uint8_t* mask) {
  CUDA_2D_KERNEL_LOOP1(i, num_seeds) {
    if (r[i] < thresh) {
      const int wstart = i % seed_w;
      const int hstart = (i / seed_w) % seed_h;
      const int n = i / seed_w / seed_h;
      const int hend = hstart + block_size;
      const int wend = wstart + block_size;
      CUDA_2D_KERNEL_LOOP2(j, C) {
        uint8_t* offset_mask = mask + (n * C + j) * H * W;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            math::utils::AtomicAnd(offset_mask + h * W + w, uint8_t(0));
          }
        }
      }
    }
  }
}

__global__ void _MaskBlock2dNHWC(
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int num_seeds,
    const int block_size,
    const uint32_t thresh,
    const uint32_t* r,
    uint8_t* mask) {
  CUDA_2D_KERNEL_LOOP1(i, num_seeds) {
    if (r[i] < thresh) {
      const int wstart = i % seed_w;
      const int hstart = (i / seed_w) % seed_h;
      const int n = i / seed_w / seed_h;
      const int hend = hstart + block_size;
      const int wend = wstart + block_size;
      CUDA_2D_KERNEL_LOOP2(j, C) {
        uint8_t* offset_mask = mask + n * H * W * C + j;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            math::utils::AtomicAnd(offset_mask + (h * W + w) * C, uint8_t(0));
          }
        }
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void DropBlock2d<T, CUDAContext>(                                         \
      const int N,                                                          \
      const int C,                                                          \
      const int H,                                                          \
      const int W,                                                          \
      const int block_size,                                                 \
      const float ratio,                                                    \
      const float scale,                                                    \
      const string& data_format,                                            \
      const T* x,                                                           \
      T* y,                                                                 \
      uint8_t* mask,                                                        \
      uint32_t* r,                                                          \
      CUDAContext* ctx) {                                                   \
    const auto seed_h = H - block_size + 1;                                 \
    const auto seed_w = W - block_size + 1;                                 \
    const auto num_seeds = N * seed_h * seed_w;                             \
    const auto NxCxHxW = N * C * H * W;                                     \
    math::Set(NxCxHxW, uint8_t(1), mask, ctx);                              \
    math::Random(num_seeds, r, ctx);                                        \
    if (data_format == "NCHW") {                                            \
      _MaskBlock2dNCHW<<<num_seeds, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          C,                                                                \
          H,                                                                \
          W,                                                                \
          seed_h,                                                           \
          seed_w,                                                           \
          num_seeds,                                                        \
          block_size,                                                       \
          uint32_t(UINT_MAX * ratio),                                       \
          r,                                                                \
          mask);                                                            \
    } else if (data_format == "NHWC") {                                     \
      _MaskBlock2dNHWC<<<num_seeds, CUDA_THREADS, 0, ctx->cuda_stream()>>>( \
          C,                                                                \
          H,                                                                \
          W,                                                                \
          seed_h,                                                           \
          seed_w,                                                           \
          num_seeds,                                                        \
          block_size,                                                       \
          uint32_t(UINT_MAX * ratio),                                       \
          r,                                                                \
          mask);                                                            \
    } else {                                                                \
      LOG(FATAL) << "Unknown DataFormat: " << data_format;                  \
    }                                                                       \
    math::ApplyMask(NxCxHxW, scale, mask, x, y, ctx);                       \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon

#endif // USE_CUDA
