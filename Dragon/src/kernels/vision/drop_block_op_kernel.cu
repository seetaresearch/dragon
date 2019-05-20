#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/* <Device = CUDA> */

__global__ void _DropBlock2dNCHW(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t          thresh,
    const uint32_t*         seed,
    int*                    mask) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        if (seed[idx] < thresh) {
            const int x = idx % seed_w;
            const int y = (idx / seed_w) % seed_h;
            const int c = (idx / seed_w / seed_h) % C;
            const int n = (idx / seed_w / seed_h) / C;
            const int nc = (n * C + c) * H;
            for (int i = 0; i < block_size; ++i) {
                const int nch = (nc + y + i) * W;
                for (int j = 0; j < block_size; ++j)
                    atomicAnd(&mask[nch + x + j], 0);
            }
        }
    }
}

__global__ void _DropBlock2dNHWC(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t          thresh,
    const uint32_t*         seed,
    int*                    mask) {
    CUDA_1D_KERNEL_LOOP(idx, nthreads) {
        if (seed[idx] < thresh) {
            const int x = idx % seed_w;
            const int y = (idx / seed_w) % seed_h;
            const int c = (idx / seed_w / seed_h) % C;
            const int n = (idx / seed_w / seed_h) / C;
            for (int i = 0; i < block_size; ++i) {
                const int nh = (n * H + y + i) * W;
                for (int j = 0; j < block_size; ++j)
                    atomicAnd(&mask[(nh + x + j) * C + c], 0);
            }
        }
    }
}

template <> void DropBlock2d<CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const float             gamma,
    const string&           data_format,
    uint32_t*               seed,
    int*                    mask,
    CUDAContext*            ctx) {
    auto nthreads = N * C * seed_h * seed_w;
    math::RandomUniform(nthreads, 0.f, 1.f, seed, ctx);
    auto mask_thresh = (uint32_t)(UINT_MAX * gamma);
    if (data_format == "NCHW") {
        _DropBlock2dNCHW
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads,
            C, H, W,
            seed_h, seed_w,
            block_size,
            mask_thresh,
            seed, mask
        );
    } else if(data_format == "NHWC") {
        _DropBlock2dNHWC
            <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
                0, ctx->cuda_stream() >>>(
            nthreads,
            C, H, W,
            seed_h, seed_w,
            block_size,
            mask_thresh,
            seed, mask
        );
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format;
    }
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA