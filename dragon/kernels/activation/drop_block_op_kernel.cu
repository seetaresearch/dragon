#ifdef USE_CUDA

#include "dragon/core/context_cuda.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

__global__ void _DropBlock2dNCHW(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const uint32_t thresh,
    const uint32_t* seed,
    int* mask) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    if (seed[idx] < thresh) {
      const int wstart = idx % seed_w;
      const int hstart = (idx / seed_w) % seed_h;
      const int n = idx / seed_w / seed_h;
      int* offset_mask = mask + n * C * H * W + hstart * W + wstart;
      for (int c = 0; c < C; ++c) {
        for (int bh = 0; bh < block_size; ++bh) {
          for (int bw = 0; bw < block_size; ++bw) {
            atomicAnd(offset_mask + c * H * W + bh * W + bw, 0);
          }
        }
      } // Share the mask between channels
    }
  }
}

__global__ void _DropBlock2dNHWC(
    const int nthreads,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const uint32_t thresh,
    const uint32_t* seed,
    int* mask) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    if (seed[idx] < thresh) {
      const int wstart = idx % seed_w;
      const int hstart = (idx / seed_w) % seed_h;
      const int n = idx / seed_w / seed_h;
      int* offset_mask = mask + n * H * W * C + hstart * W * C + wstart * C;
      for (int bh = 0; bh < block_size; ++bh) {
        for (int bw = 0; bw < block_size; ++bw) {
          for (int c = 0; c < C; ++c) {
            atomicAnd(offset_mask + bh * W * C + bw * C + c, 0);
          }
        }
      } // Share the mask between channels
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void DropBlock2d<CUDAContext>(
    const int N,
    const int C,
    const int H,
    const int W,
    const int seed_h,
    const int seed_w,
    const int block_size,
    const float gamma,
    const string& data_format,
    uint32_t* seed,
    int* mask,
    CUDAContext* ctx) {
  auto nthreads = N * seed_h * seed_w;
  math::RandomUniform(nthreads, 0.f, 1.f, seed, ctx);
  auto mask_thresh = (uint32_t)(UINT_MAX * gamma);
  if (data_format == "NCHW") {
    _DropBlock2dNCHW<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads, C, H, W, seed_h, seed_w, block_size, mask_thresh, seed, mask);
  } else if (data_format == "NHWC") {
    _DropBlock2dNHWC<<<
        CUDA_BLOCKS(nthreads),
        CUDA_THREADS,
        0,
        ctx->cuda_stream()>>>(
        nthreads, C, H, W, seed_h, seed_w, block_size, mask_thresh, seed, mask);
  } else {
    LOG(FATAL) << "Unknown DataFormat: " << data_format;
  }
}

} // namespace kernel

} // namespace dragon

#endif // USE_CUDA
