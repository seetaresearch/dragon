#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ROIPool <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ROIPool(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const float             spatial_scale,
    const T*                x,
    const float*            rois,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
        int pw = y_idx % pool_w;
        int ph = (y_idx / pool_w) % pool_h;
        int c = (y_idx / pool_w / pool_h) % C;
        int n = y_idx / pool_w / pool_h / C;

        const float* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            y[y_idx] = 0; mask[y_idx] = -1; continue;
        }

        int roi_start_w = round(offset_rois[1] * spatial_scale);
        int roi_start_h = round(offset_rois[2] * spatial_scale);
        int roi_end_w = round(offset_rois[3] * spatial_scale);
        int roi_end_h = round(offset_rois[4] * spatial_scale);

        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        const float bin_size_h = (float)roi_height / (float)pool_h;
        const float bin_size_w = (float)roi_width / (float)pool_w;

        int hstart = floor(bin_size_h * ph);
        int wstart = floor(bin_size_w * pw);
        int hend = ceil(bin_size_h * (ph + 1));
        int wend = ceil(bin_size_w * (pw + 1));

        hstart = min(max(hstart + roi_start_h, 0), H);
        hend = min(max(hend + roi_start_h, 0), H);
        wstart = min(max(wstart + roi_start_w, 0), W);
        wend = min(max(wend + roi_start_w, 0), W);

        bool is_empty = (hend <= hstart) || (wend <= wstart);
        float max_val = is_empty ? 0 : -FLT_MAX;
        int max_idx = -1;
        x += ((roi_batch_ind * C + c) * H * W);
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                const int x_idx = h * W + w;
#if __CUDA_ARCH__ >= 350
                if (__ldg(x + x_idx) > max_val) {
                    max_val = __ldg(x + x_idx);
                    max_idx = x_idx;
                }
#else
                if (x[x_idx] > max_val) {
                    max_val = x[x_idx];
                    max_idx = x_idx;
                }
#endif
            }
        }
        y[y_idx] = max_val;
        mask[y_idx] = max_idx;
    }
}

template<> void ROIPool<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            x,
    const float*            rois,
    int*                    mask,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = num_rois * C * pool_h * pool_w;
    _ROIPool<float>
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, C, H, W, pool_h, pool_w,
            spatial_scale, x, rois, mask, y);
}

/*! ROIPool <T = float16, Device = CUDA> */

__global__ void _ROIPoolHalf(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const float             spatial_scale,
    const half*             x,
    const float*            rois,
    int*                    mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(y_idx, nthreads) {
#if __CUDA_ARCH__ >= 530
        int pw = y_idx % pool_w;
        int ph = (y_idx / pool_w) % pool_h;
        int c = (y_idx / pool_w / pool_h) % C;
        int n = y_idx / pool_w / pool_h / C;

        const float* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            y[y_idx] = __float2half(0.f);
            mask[y_idx] = -1; continue;
        }

        int roi_start_w = round(offset_rois[1] * spatial_scale);
        int roi_start_h = round(offset_rois[2] * spatial_scale);
        int roi_end_w = round(offset_rois[3] * spatial_scale);
        int roi_end_h = round(offset_rois[4] * spatial_scale);

        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        const float bin_size_h = (float)roi_height / (float)pool_h;
        const float bin_size_w = (float)roi_width / (float)pool_w;

        int hstart = floor(bin_size_h * ph);
        int wstart = floor(bin_size_w * pw);
        int hend = ceil(bin_size_h * (ph + 1));
        int wend = ceil(bin_size_w * (pw + 1));

        hstart = min(max(hstart + roi_start_h, 0), H);
        hend = min(max(hend + roi_start_h, 0), H);
        wstart = min(max(wstart + roi_start_w, 0), W);
        wend = min(max(wend + roi_start_w, 0), W);

        bool is_empty = (hend <= hstart) || (wend <= wstart);
        x += ((roi_batch_ind * C + c) * H * W);

        int max_idx = -1;
        half max_val = is_empty ? __float2half(0.f) : x[0];

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                const int x_idx = h * W + w;
                if (__hgt(__ldg(x + x_idx), max_val)) {
                    max_val = __ldg(x + x_idx);
                    max_idx = x_idx;
                }
            }
        }
        y[y_idx] = max_val;
        mask[y_idx] = max_idx;
#endif
    }
}

template<> void ROIPool<float16, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float16*          x,
    const float*            rois,
    int*                    mask,
    float16*                y,
    CUDAContext*            ctx) {
    auto nthreads = num_rois * C * pool_h * pool_w;
    _ROIPoolHalf
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, C, H, W, pool_h, pool_w, spatial_scale,
            reinterpret_cast<const half*>(x), rois,
                mask, reinterpret_cast<half*>(y));
}

/*! ROIPoolGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ROIPoolGrad(
    const int               nthreads,
    const int               num_rois,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const T                 spatial_scale,
    const T*                dy,
    const T*                rois,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(x_idx, nthreads) {
        int w = x_idx % W;
        int h = (x_idx / W) % H;
        int c = (x_idx / W / H) % C;
        int n = x_idx / W / H / C;

        T gradient = 0;

        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const T* offset_rois = rois + roi_n * 5;
            int roi_batch_ind = offset_rois[0];

            if (n != roi_batch_ind) continue;

            int roi_start_w = round(offset_rois[1] * spatial_scale);
            int roi_start_h = round(offset_rois[2] * spatial_scale);
            int roi_end_w = round(offset_rois[3] * spatial_scale);
            int roi_end_h = round(offset_rois[4] * spatial_scale);

            const bool in_roi = (w >= roi_start_w &&
                                 w <= roi_end_w &&
                                 h >= roi_start_h &&
                                 h <= roi_end_h);

            if (!in_roi) continue;

            int y_offset = (roi_n * C + c) * pool_h * pool_w;
            const T* offset_dy = dy + y_offset;
            const int* offset_mask = mask + y_offset;

            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            const T bin_size_h = (T)roi_height / (T)pool_h;
            const T bin_size_w = (T)roi_width / (T)pool_w;

            int phstart = floor(static_cast<T>(h - roi_start_h) / bin_size_h);
            int phend = ceil(static_cast<T>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<T>(w - roi_start_w) / bin_size_w);
            int pwend = ceil(static_cast<T>(w - roi_start_w + 1) / bin_size_w);

            phstart = min(max(phstart, 0), pool_h);
            phend = min(max(phend, 0), pool_h);
            pwstart = min(max(pwstart, 0), pool_w);
            pwend = min(max(pwend, 0), pool_w);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int pool_idx = ph * pool_w + pw;
                    if (offset_mask[pool_idx] == (h * W + w)) {
                        gradient += offset_dy[pool_idx];
                    }
                }
            }
        }
        dx[x_idx] = gradient;
    }
}

template<> void ROIPoolGrad<float, CUDAContext>(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            dy,
    const float*            rois,
    const int*              mask,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = N * C * H * W;
    _ROIPoolGrad<float>
        << < CUDA_BLOCKS(nthreads), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (nthreads, num_rois, C, H, W, pool_h, pool_w,
            spatial_scale, dy, rois, mask, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA