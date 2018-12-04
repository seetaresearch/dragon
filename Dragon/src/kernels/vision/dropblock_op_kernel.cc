#include "utils/op_kernel.h"
#include "utils/math_functions.h"

namespace dragon {

namespace kernel {

/*! DropBlock2d <T = float32, Device = CPU> */

void _DropBlock2d_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t*         seed,
    int*                    mask) {
    TIndex seed_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int nc = (n * C + c) * H;
            for (int y = 0; y < seed_h; ++y) {
                for (int x = 0; x < seed_w; ++x) {
                    if (seed[seed_idx] > 0) {
                        for (int i = 0; i < block_size; ++i) {
                            const int nch = (nc + y + i) * W;
                            for (int j = 0; j < block_size; ++j) {
                                mask[nch + x + j] &= 0;
                            }  // End j
                        }  // End i
                    }
                    seed_idx++;
                }  // End x
            }  // End y
        }  // End c
    }  // End n
}

void _DropBlock2d_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               seed_h,
    const int               seed_w,
    const int               block_size,
    const uint32_t*         seed,
    int*                    mask) {
    TIndex seed_idx = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < seed_h; ++y) {
                for (int x = 0; x < seed_w; ++x) {
                    if (seed[seed_idx] > 0) {
                        for (int i = 0; i < block_size; ++i) {
                            const int nh = (n * H + y + i) * W;
                            for (int j = 0; j < block_size; ++j) {
                                mask[(nh + x + j) * C + c] &= 0;
                            }  // End j
                        }  // End i
                    }
                    seed_idx++;
                }  // End x
            }  // End y
        }  // End c
    }  // End n
}

template <> void DropBlock2d<CPUContext>(
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
    CPUContext*             ctx) {
    const int count = N * C * seed_h * seed_w;
    math::RandomBernoulli<uint32_t, CPUContext>(
        count, gamma, seed, ctx);
    if (data_format == "NCHW") {
        _DropBlock2d_NCHW(N, C, H, W,
            seed_h, seed_w, block_size, seed, mask);
    } else if (data_format == "NHWC") {
        _DropBlock2d_NHWC(N, C, H, W,
            seed_h, seed_w, block_size, seed, mask);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

}  // namespace kernel

}  // namepsace dragon