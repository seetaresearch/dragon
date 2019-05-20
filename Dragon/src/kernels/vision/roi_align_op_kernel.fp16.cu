#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float16, Device = CUDA> */

__device__ float _ROIAlignIntp(
    const half*             X,
    const int               H,
    const int               W,
    float                   y,
    float                   x) {
    if (y < -1.0 || y > H || x < -1.0 || x > W) return 0.f;
#if __CUDA_ARCH__ >= 530
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= H - 1) {
        y_high = y_low = H - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= W - 1) {
        x_high = x_low = W - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    const float ly = y - y_low;
    const float lx = x - x_low;
    const float hy = 1.f - ly, hx = 1.f - lx;
    const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    const float v1 = __half2float(__ldg(X + (y_low * W + x_low)));
    const float v2 = __half2float(__ldg(X + (y_low * W + x_high)));
    const float v3 = __half2float(__ldg(X + (y_high * W + x_low)));
    const float v4 = __half2float(__ldg(X + (y_high * W + x_high)));
    const float value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
#else
    const float value = 0.f;
#endif
    return value;
}

__global__ void _ROIAlignHalf(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const float             spatial_scale,
    const half*             xdata,
    const float*            rois,
    half*                   ydata) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
        int pw = yi % pool_w;
        int ph = (yi / pool_w) % pool_h;
        int c = (yi / pool_w / pool_h) % C;
        int n = yi / pool_w / pool_h / C;

        const float* roi = rois + n * 5;
        int batch_ind = roi[0];

        if (batch_ind < 0) {
            ydata[yi] = __float2half(0.f);
            continue;
        }

        float roi_wstart = roi[1] * spatial_scale;
        float roi_hstart = roi[2] * spatial_scale;
        float roi_wend = roi[3] * spatial_scale;
        float roi_hend = roi[4] * spatial_scale;

        float roi_w = max(roi_wend - roi_wstart, 1.f);
        float roi_h = max(roi_hend - roi_hstart, 1.f);
        float bin_h = roi_h / (float)pool_h;
        float bin_w = roi_w / (float)pool_w;

        const half* X = xdata + (batch_ind * C + c) * H * W;

        int grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_h / pool_h);
        int grid_w = (sampling_ratio > 0) ? 
            sampling_ratio : ceil(roi_w / pool_w);
        
        float intp_val = 0.f;

        for (int iy = 0; iy < grid_h; iy++) {
            const float y = roi_hstart + ph * bin_h +
                (float)(iy + .5f) * bin_h / (float)grid_h;
            for (int ix = 0; ix < grid_w; ix++) {
                const float x = roi_wstart + pw * bin_w +
                    (float)(ix + .5f) * bin_w / (float)grid_w;
                intp_val += _ROIAlignIntp(X, H, W, y, x);
            }
        }
        ydata[yi] = __float2half(
            intp_val / float(grid_h * grid_w)
        );
#endif
    }
}

template<> void ROIAlign<float16, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float16*          x,
    const float*            rois,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = num_rois * C  * pool_h * pool_w;
    _ROIAlignHalf
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        C, H, W,
        pool_h, pool_w,
        sampling_ratio,
        spatial_scale,
        reinterpret_cast<const half*>(x),
        rois,
        reinterpret_cast<half*>(y)
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA