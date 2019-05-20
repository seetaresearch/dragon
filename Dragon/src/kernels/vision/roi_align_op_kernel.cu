#ifdef WITH_CUDA

#include "core/context_cuda.h"
#include "utils/op_kernel.h"

namespace dragon {

namespace kernel {

/* <T = float32, Device = CUDA> */

template <typename T>
__device__ T _ROIAlignIntp(
    const T*                X,
    const int               H,
    const int               W,
    T                       y,
    T                       x) {
    if (y < -1.0 || y > H || x < -1.0 || x > W) return 0;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= H - 1) {
        y_high = y_low = H - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= W - 1) {
        x_high = x_low = W - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    const T ly = y - y_low;
    const T lx = x - x_low;
    const T hy = T(1) - ly;
    const T hx = T(1) - lx;
#if __CUDA_ARCH__ >= 350
    T v1 = __ldg(X + (y_low * W + x_low));
    T v2 = __ldg(X + (y_low * W + x_high));
    T v3 = __ldg(X + (y_high * W + x_low));
    T v4 = __ldg(X + (y_high * W + x_high));
#else
    T v1 = X[y_low * W + x_low];
    T v2 = X[y_low * W + x_high];
    T v3 = X[y_high * W + x_low];
    T v4 = X[y_high * W + x_high];
#endif
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
__global__ void _ROIAlign(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const float             spatial_scale,
    const T*                xdata,
    const float*            rois,
    T*                      ydata) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        int pw = yi % pool_w;
        int ph = (yi / pool_w) % pool_h;
        int c = (yi / pool_w / pool_h) % C;
        int n = yi / pool_w / pool_h / C;

        const T* roi = rois + n * 5;
        const int batch_ind = roi[0];

        if (batch_ind < 0) {
            ydata[yi] = T(0);
            continue;
        }

        const T roi_wstart = roi[1] * spatial_scale;
        const T roi_hstart = roi[2] * spatial_scale;
        const T roi_wend = roi[3] * spatial_scale;
        const T roi_hend = roi[4] * spatial_scale;

        const T roi_w = max(roi_wend - roi_wstart, T(1));
        const T roi_h = max(roi_hend - roi_hstart, T(1));
        const T bin_h = roi_h / (T)pool_h;
        const T bin_w = roi_w / (T)pool_w;

        const T* X = xdata + (batch_ind * C + c) * H * W;

        const int grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_h / pool_h);
        const int grid_w = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_w / pool_w);
        
        T intp_val = T(0);

        for (int iy = 0; iy < grid_h; iy++) {
            const T y = roi_hstart + ph * bin_h +
                (T)(iy + .5f) * bin_h / (T)grid_h;
            for (int ix = 0; ix < grid_w; ix++) {
                const T x = roi_wstart + pw * bin_w +
                    (T)(ix + .5f) * bin_w / (T)grid_w;
                intp_val += _ROIAlignIntp(X, H, W, y, x);
            }
        }
        ydata[yi] = intp_val / T(grid_h * grid_w);
    }
}

template<> void ROIAlign<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            x,
    const float*            rois,
    float*                  y,
    CUDAContext*            ctx) {
    auto nthreads = num_rois * C  * pool_h * pool_w;
    _ROIAlign
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        C, H, W,
        pool_h, pool_w,
        sampling_ratio,
        spatial_scale,
        x, rois, y
    );
}

/* <T = float32, Device = CUDA> */

template <typename T>
__device__ void _ROIAlignIntpGrad(
    const int               H,
    const int               W,
    T                       y,
    T                       x,
    T&                      w1,
    T&                      w2,
    T&                      w3,
    T&                      w4,
    int&                    x_low,
    int&                    x_high,
    int&                    y_low,
    int&                    y_high) {
    if (y < -1.0 || y > H || x < -1.0 || x > W) {
        w1 = w2 = w3 = w4 = 0.;
        x_low = x_high = y_low = y_high = -1;
        return;
    }

    if (y <= 0) y = 0;
    if (x <= 0) x = 0;

    y_low = (int)y;
    x_low = (int)x;

    if (y_low >= H - 1) {
        y_high = y_low = H - 1;
        y = (T)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= W - 1) {
        x_high = x_low = W - 1;
        x = (T)x_low;
    } else {
        x_high = x_low + 1;
    }

    const T ly = y - y_low;
    const T lx = x - x_low;
    const T hy = T(1) - ly;
    const T hx = T(1) - lx;
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename T>
__global__ void _ROIAlignGrad(
    const int               nthreads,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               sampling_ratio,
    const T                 spatial_scale,
    const T*                dy,
    const T*                rois,
    T*                      dx) {
    CUDA_1D_KERNEL_LOOP(yi, nthreads) {
        int pw = yi % pool_w;
        int ph = (yi / pool_w) % pool_h;
        int c = (yi / pool_w / pool_h) % C;
        int n = yi / pool_w / pool_h / C;

        const T* roi = rois + n * 5;
        int batch_ind = roi[0];

        if (batch_ind < 0) continue;

        const T roi_wstart = roi[1] * spatial_scale;
        const T roi_hstart = roi[2] * spatial_scale;
        const T roi_wend = roi[3] * spatial_scale;
        const T roi_hend = roi[4] * spatial_scale;

        const T roi_w = max(roi_wend - roi_wstart, T(1));
        const T roi_h = max(roi_hend - roi_hstart, T(1));
        const T bin_h = roi_h / (T)pool_h;
        const T bin_w = roi_w / (T)pool_w;

        T* dX = dx + (batch_ind * C + c) * H * W;
        const T dY = dy[((n * C + c) * pool_h + ph) * pool_w + pw];

        const int grid_h = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_h / pool_h);
        const int grid_w = (sampling_ratio > 0) ?
            sampling_ratio : ceil(roi_w / pool_w);

        const T num_grids = grid_h * grid_w;

        for (int iy = 0; iy < grid_h; iy++) {
            const T y = roi_hstart + ph * bin_h +
                (T)(iy + .5f) * bin_h / (T)(grid_h);
            for (int ix = 0; ix < grid_w; ix++) {
                const T x = roi_wstart + pw * bin_w +
                    (T)(ix + .5f) * bin_w / (T)(grid_w);

                T w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;

                _ROIAlignIntpGrad(
                    H, W, y, x, w1, w2, w3, w4,
                    x_low, x_high, y_low, y_high
                );

                T g1 = dY * w1 / num_grids;
                T g2 = dY * w2 / num_grids;
                T g3 = dY * w3 / num_grids;
                T g4 = dY * w4 / num_grids;

                if (x_low >= 0 && x_high >= 0 &&
                    y_low >= 0 && y_high >= 0) {
                    atomicAdd(dX + y_low * W + x_low, g1);
                    atomicAdd(dX + y_low * W + x_high, g2);
                    atomicAdd(dX + y_high * W + x_low, g3);
                    atomicAdd(dX + y_high * W + x_high, g4);
                }
            }  // End iy
        }  // End ix
    }
}

template<> void ROIAlignGrad<float, CUDAContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            dy,
    const float*            rois,
    float*                  dx,
    CUDAContext*            ctx) {
    auto nthreads = num_rois * C  * pool_h * pool_w;
    _ROIAlignGrad
        <<< CUDA_BLOCKS(nthreads), CUDA_THREADS,
            0, ctx->cuda_stream() >>>(
        nthreads,
        C, H, W,
        pool_h, pool_w,
        sampling_ratio,
        spatial_scale,
        dy, rois, dx
    );
}

}  // namespace kernel

}  // namepsace dragon

#endif  // WITH_CUDA