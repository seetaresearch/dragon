#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ROIAlign <T = float16, Device = CUDA> */

__device__ float _ROIAlignInterpolate(
    const half*             Xdata,
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
    const float hy = 1. - ly, hx = 1. - lx;
    const float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    const float v1 = __half2float(__ldg(Xdata + (y_low * W + x_low)));
    const float v2 = __half2float(__ldg(Xdata + (y_low * W + x_high)));
    const float v3 = __half2float(__ldg(Xdata + (y_high * W + x_low)));
    const float v4 = __half2float(__ldg(Xdata + (y_high * W + x_high)));
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
    const half*             Xdata,
    const float*            rois,
    half*                   Ydata) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        int pw = y_idx % pool_w;
        int ph = (y_idx / pool_w) % pool_h;
        int c = (y_idx / pool_w / pool_h) % C;
        int n = y_idx / pool_w / pool_h / C;

        const float* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            Ydata[y_idx] = __float2half(0.f);
            continue;
        }

        float roi_start_w = offset_rois[1] * spatial_scale;
        float roi_start_h = offset_rois[2] * spatial_scale;
        float roi_end_w = offset_rois[3] * spatial_scale;
        float roi_end_h = offset_rois[4] * spatial_scale;

        float roi_width = max(roi_end_w - roi_start_w, 1.f);
        float roi_height = max(roi_end_h - roi_start_h, 1.f);
        float bin_size_h = (float)roi_height / (float)pool_h;
        float bin_size_w = (float)roi_width / (float)pool_w;

        const half* offset_Xdata = Xdata + (roi_batch_ind * C + c) * H * W;

        int roi_bin_grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_height / pool_h);
        int roi_bin_grid_w = (sampling_ratio > 0) ? 
            sampling_ratio : ceil(roi_width / pool_w);
        
        float output_val = 0.;
        const float num_bin_grids = roi_bin_grid_h * roi_bin_grid_w;

        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            const float y = roi_start_h + ph * bin_size_h +
                static_cast<float>(iy + .5f) * bin_size_h /
                    static_cast<float>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const float x = roi_start_w + pw * bin_size_w +
                    static_cast<float>(ix + .5f) * bin_size_w /
                        static_cast<float>(roi_bin_grid_w);
                output_val += _ROIAlignInterpolate(
                    offset_Xdata, H, W, y, x);
            }
        }
        output_val /= num_bin_grids;
        Ydata[y_idx] = __float2half(output_val);
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
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, C, H, W, pool_h, pool_w,
            sampling_ratio, spatial_scale,
                reinterpret_cast<const half*>(x), rois,
                    reinterpret_cast<half*>(y));
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA