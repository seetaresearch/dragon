#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        int pw = yi % pool_w;
        int ph = (yi / pool_w) % pool_h;
        int c = (yi / pool_w / pool_h) % C;
        int n = yi / pool_w / pool_h / C;

        const float* roi = rois + n * 5;
        const int batch_ind = roi[0];
        const T* X = x + ((batch_ind * C + c) * H * W);

        if (batch_ind < 0) {
            y[yi] = 0; mask[yi] = -1; continue;
        }

        const int roi_wstart = round(roi[1] * spatial_scale);
        const int roi_hstart = round(roi[2] * spatial_scale);
        const int roi_wend = round(roi[3] * spatial_scale);
        const int roi_hend = round(roi[4] * spatial_scale);

        const int roi_w = max(roi_wend - roi_wstart + 1, 1);
        const int roi_h = max(roi_hend - roi_hstart + 1, 1);
        const float bin_h = (float)roi_h / (float)pool_h;
        const float bin_w = (float)roi_w / (float)pool_w;

        int hstart = floor(bin_h * ph);
        int wstart = floor(bin_w * pw);
        int hend = ceil(bin_h * (ph + 1));
        int wend = ceil(bin_w * (pw + 1));

        hstart = min(max(hstart + roi_hstart, 0), H);
        hend = min(max(hend + roi_hstart, 0), H);
        wstart = min(max(wstart + roi_wstart, 0), W);
        wend = min(max(wend + roi_wstart, 0), W);
        const bool empty = (hend <= hstart) || (wend <= wstart);

        int xi, maxi = -1;
        T maxv = empty ? T(0) : -FLT_MAX;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                xi = h * W + w;
#if __CUDA_ARCH__ >= 350
                if (__ldg(X + xi) > maxv) {
                    maxi = xi; maxv = __ldg(X + xi);
                }
#else
                if (X[xi] > maxv) {
                    maxi = xi; maxv = X[xi];
                }
#endif
            }
        }
        y[yi] = maxv; mask[yi] = maxi;
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
    _ROIPool
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        C, H, W,
        pool_h, pool_w,
        spatial_scale,
        x, rois,
        mask, y
    );
}

/* <T = float16, Device = CUDA> */

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
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
#if __CUDA_ARCH__ >= 530
        int pw = yi % pool_w;
        int ph = (yi / pool_w) % pool_h;
        int c = (yi / pool_w / pool_h) % C;
        int n = yi / pool_w / pool_h / C;

        const float* roi = rois + n * 5;
        const int batch_ind = roi[0];
        const half* X = x + ((batch_ind * C + c) * H * W);

        if (batch_ind < 0) {
            y[yi] = __float2half(0.f);
            mask[yi] = -1; continue;
        }

        const int roi_wstart = round(roi[1] * spatial_scale);
        const int roi_hstart = round(roi[2] * spatial_scale);
        const int roi_wend = round(roi[3] * spatial_scale);
        const int roi_hend = round(roi[4] * spatial_scale);

        const int roi_w = max(roi_wend - roi_wstart + 1, 1);
        const int roi_h = max(roi_hend - roi_hstart + 1, 1);
        const float bin_h = (float)roi_h / (float)pool_h;
        const float bin_w = (float)roi_w / (float)pool_w;

        int hstart = floor(bin_h * ph);
        int wstart = floor(bin_w * pw);
        int hend = ceil(bin_h * (ph + 1));
        int wend = ceil(bin_w * (pw + 1));

        hstart = min(max(hstart + roi_hstart, 0), H);
        hend = min(max(hend + roi_hstart, 0), H);
        wstart = min(max(wstart + roi_wstart, 0), W);
        wend = min(max(wend + roi_wstart, 0), W);
        const bool empty = (hend <= hstart) || (wend <= wstart);

        int xi, maxi = -1;
        half maxv = empty ? __float2half(0.f) : X[0];

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                xi = h * W + w;
                if (__hgt(__ldg(X + xi), maxv)) {
                    maxi = xi; maxv = __ldg(X + xi);
                }
            }
        }
        y[yi] = maxv; mask[yi] = maxi;
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
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        C, H, W,
        pool_h, pool_w,
        spatial_scale,
        reinterpret_cast<const half*>(x),
        rois, mask,
        reinterpret_cast<half*>(y)
    );
}

/* <T = float32, Device = CUDA> */

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
    CUDA_1D_KERNEL_LOOP(xi, nthreads) {
        const int w = xi % W;
        const int h = (xi / W) % H;
        const int c = (xi / W / H) % C;
        const int n = xi / W / H / C;

        T grad = 0;

        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const T* roi = rois + roi_n * 5;
            if (n != roi[0]) continue;

            const int roi_wstart = round(roi[1] * spatial_scale);
            const int roi_hstart = round(roi[2] * spatial_scale);
            const int roi_wend = round(roi[3] * spatial_scale);
            const int roi_hend = round(roi[4] * spatial_scale);

            const bool in_roi = (w >= roi_wstart &&
                                 w <= roi_wend &&
                                 h >= roi_hstart &&
                                 h <= roi_hend);

            if (!in_roi) continue;

            const int y_ofs = (roi_n * C + c) * pool_h * pool_w;
            const T* dY = dy + y_ofs;
            const int* M = mask + y_ofs;

            const int roi_w = max(roi_wend - roi_wstart + 1, 1);
            const int roi_h = max(roi_hend - roi_hstart + 1, 1);

            const T bin_h = (T)roi_h / (T)pool_h;
            const T bin_w = (T)roi_w / (T)pool_w;

            int phstart = floor((T)(h - roi_hstart) / (T)bin_h);
            int phend = ceil((T)(h - roi_hstart + 1) / (T)bin_h);
            int pwstart = floor((T)(w - roi_wstart) / (T)bin_w);
            int pwend = ceil((T)(w - roi_wstart + 1) / (T)bin_w);

            phstart = min(max(phstart, 0), pool_h);
            phend = min(max(phend, 0), pool_h);
            pwstart = min(max(pwstart, 0), pool_w);
            pwend = min(max(pwend, 0), pool_w);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    int yi = ph * pool_w + pw;
                    if (M[yi] == (h * W + w)) {
                        grad += dY[yi];
                    }
                }
            }
        }
        dx[xi] = grad;
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
    _ROIPoolGrad
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        num_rois,
        C, H, W,
        pool_h, pool_w,
        spatial_scale,
        dy, rois,
        mask, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA