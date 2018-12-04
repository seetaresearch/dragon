#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/*! ROIPooling <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ROIPooling(
    const int               count,
    const float             spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const T*                x,
    const float*            rois,
    int*                    mask,
    T*                      y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const float* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            y[idx] = 0;
            mask[idx] = -1;
            continue;
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

        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);
        float max_val = is_empty ? 0 : -FLT_MAX;
        int max_idx = -1;
        x += ((roi_batch_ind * channels + c) * height * width);
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                const int x_idx = h * width + w;
                if (x[x_idx] > max_val) {
                    max_val = x[x_idx];
                    max_idx = x_idx;
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
    }
}

template<> void ROIPooling<float, CUDAContext>(
    const int               count,
    const int               N,
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
    _ROIPooling<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, spatial_scale, C, H, W,
            pool_h, pool_w, x, rois, mask, y);
}

/*! ROIPooling <T = float16, Device = CUDA> */

__global__ void _ROIPoolingHalf(
    const int               count,
    const float             spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const half*             x,
    const float*            rois,
    int*                    mask,
    half*                   y) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
#if __CUDA_ARCH__ >= 530
        int pw = idx % pool_w;
        int ph = (idx / pool_w) % pool_h;
        int c = (idx / pool_w / pool_h) % channels;
        int n = idx / pool_w / pool_h / channels;

        const float* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        if (roi_batch_ind < 0) {
            y[idx] = __float2half(0.f);
            mask[idx] = -1;
            continue;
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

        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);
        x += ((roi_batch_ind * channels + c) * height * width);

        int max_idx = -1;
        half max_val = is_empty ? __float2half(0.f) : x[0];

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                const int x_idx = h * width + w;
                if (__hgt(x[x_idx], max_val)) {
                    max_val = x[x_idx];
                    max_idx = x_idx;
                }
            }
        }
        y[idx] = max_val;
        mask[idx] = max_idx;
#endif
    }
}

template<> void ROIPooling<float16, CUDAContext>(
    const int               count,
    const int               N,
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
    _ROIPoolingHalf
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, spatial_scale, C, H, W, pool_h, pool_w,
            reinterpret_cast<const half*>(x), rois,
                mask, reinterpret_cast<half*>(y));
}

/*! ROIPoolingGrad <T = float32, Device = CUDA> */

template <typename T>
__global__ void _ROIPoolingGrad(
    const int               count,
    const int               num_rois,
    const T                 spatial_scale,
    const int               channels,
    const int               height,
    const int               width,
    const int               pool_h,
    const int               pool_w,
    const T*                dy,
    const T*                rois,
    const int*              mask,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(idx, count) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / width / height) % channels;
        int n = idx / width / height / channels;

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

            int y_offset = (roi_n * channels + c) * pool_h * pool_w;
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
                    if (offset_mask[pool_idx] == (h * width + w)) {
                        gradient += offset_dy[pool_idx];
                    }
                }
            }
        }
        dx[idx] = gradient;
    }
}

template<> void ROIPoolingGrad<float, CUDAContext>(
    const int               count,
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
    _ROIPoolingGrad<float>
        << < CUDA_BLOCKS(count), CUDA_THREADS,
             0, ctx->cuda_stream() >> >
        (count, num_rois, spatial_scale, C, H, W,
            pool_h, pool_w, dy, rois, mask, dx);
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA